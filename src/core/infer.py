# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List, Optional, Tuple, Union
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from ..common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from ..common.distributed import (
    get_device,
)
from ..optimization.performance import (
    optimized_channels_to_last,
    optimized_channels_to_second
)
from ..models.dit_3b import na


class VideoDiffusionInfer():
    def __init__(self, config: DictConfig, debug: 'Debug',
                 encode_tiled: bool = False, encode_tile_size: Tuple[int, int] = (512, 512), 
                 encode_tile_overlap: Tuple[int, int] = (64, 64),
                 decode_tiled: bool = False, decode_tile_size: Tuple[int, int] = (512, 512),
                 decode_tile_overlap: Tuple[int, int] = (64, 64),
                 tile_debug: str = "false"):
        self.config = config
        self.debug = debug
        # Store separate encode and decode tiling parameters
        self.encode_tiled = encode_tiled
        self.encode_tile_size = encode_tile_size
        self.encode_tile_overlap = encode_tile_overlap
        self.decode_tiled = decode_tiled
        self.decode_tile_size = decode_tile_size
        self.decode_tile_overlap = decode_tile_overlap
        self.tile_debug = tile_debug
        
    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError
    
    def configure_diffusion(self, device: Optional[torch.device] = None, dtype=torch.float32):
        """
        Configure diffusion schedule and sampler.
        
        Args:
            device: Device for schedule tensors. If None, uses get_device()
            dtype: Data type for computations
        """
        # Use provided device or fallback to standard detection
        if device is None:
            device = get_device()
        elif not isinstance(device, torch.device):
            device = torch.device(device)
            
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=device,
            dtype=dtype,
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=device,
            dtype=dtype,
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )
        # Propagate debug to sampler
        if hasattr(self, 'debug'):
            self.sampler.debug = self.debug

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        """VAE encode with configured dtype - converts samples to latents with optional tiling"""
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            # Use VAE model's current device
            # This ensures consistency with where the VAE model is loaded
            try:
                device = next(self.vae.parameters()).device
            except StopIteration:
                # Fallback if VAE has no parameters (shouldn't happen)
                device = get_device()
            
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # VAE process by each group.
            for sample in batches:
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)

                # Detect VAE model dtype
                try:
                    vae_dtype = next(self.vae.parameters()).dtype
                except StopIteration:
                    vae_dtype = dtype  # Fallback

                # Use autocast if VAE dtype differs from input dtype
                # Skip autocast on MPS (only supports bf16, unified memory = no benefit)
                # Instead, explicitly convert input to model dtype
                if vae_dtype != sample.dtype:
                    if device.type == 'mps':
                        # MPS: explicit dtype conversion instead of autocast
                        sample = sample.to(vae_dtype)
                        if use_sample:
                            latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                    tile_overlap=self.encode_tile_overlap).latent
                        else:
                            latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                                tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)
                    else:
                        with torch.autocast(device.type, sample.dtype, enabled=True):
                            if use_sample:
                                latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                        tile_overlap=self.encode_tile_overlap).latent
                            else:
                                latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                                    tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)
                else:
                    if use_sample:
                        latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                tile_overlap=self.encode_tile_overlap).latent
                    else:
                        # Deterministic vae encode, only used for i2v inference (optionally)
                        latent = self.vae.encode(sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                            tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)

                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = optimized_channels_to_last(latent)
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]
            
            self.debug.log(f"Latents shape: {latents[0].shape}", category="info", indent_level=1)

        return latents
    

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        """VAE decode with configured dtype - converts latents to samples with optional tiling.

        For large video batches (T_lat > 9) on CUDA, temporal chunking is used to reduce
        peak VRAM: the latent is split into overlapping 5-latent-frame chunks that are
        decoded independently and moved to CPU immediately.  Chunk boundaries are
        smoothed via linear blending over the overlap region, eliminating flickering.
        """
        samples = []
        if len(latents) > 0:
            # Use VAE model's current device
            # This ensures consistency with where the VAE model is loaded
            try:
                device = next(self.vae.parameters()).device
            except StopIteration:
                # Fallback if VAE has no parameters (shouldn't happen)
                device = get_device()
            
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            self.debug.log(f"Latents shape: {latents[0].shape}", category="info", indent_level=1)

            for i, latent in enumerate(latents):
                latent = latent / scale + shift
                latent = optimized_channels_to_second(latent)
                latent = latent.squeeze(2)

                # Detect VAE model dtype
                try:
                    vae_dtype = next(self.vae.parameters()).dtype
                except StopIteration:
                    vae_dtype = dtype  # Fallback

                # Temporal chunking: enabled for 5-D video tensors on CUDA when the
                # temporal dimension exceeds the fallback threshold (T_lat > 9, i.e.,
                # > 33 video frames in 4n+1 format).  MPS uses unified memory so
                # offloading to CPU brings no VRAM benefit there.
                temporal_downsample_factor = self.config.vae.model.get(
                    "temporal_downsample_factor", 4
                )
                use_temporal_chunks = (
                    latent.ndim == 5
                    and latent.shape[2] > 9
                    and device.type == 'cuda'
                )

                if use_temporal_chunks:
                    sample = self._decode_with_temporal_chunks(
                        latent, device, vae_dtype, temporal_downsample_factor
                    )
                else:
                    # Single-pass decode (original path)
                    if vae_dtype != latent.dtype:
                        if device.type == 'mps':
                            # MPS: explicit dtype conversion instead of autocast
                            latent = latent.to(vae_dtype)
                            sample = self.vae.decode(
                                latent,
                                tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                                tile_overlap=self.decode_tile_overlap
                            ).sample
                        else:
                            with torch.autocast(device.type, latent.dtype, enabled=True):
                                sample = self.vae.decode(
                                    latent,
                                    tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                                    tile_overlap=self.decode_tile_overlap
                                ).sample
                    else:
                        sample = self.vae.decode(
                            latent,
                            tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                            tile_overlap=self.decode_tile_overlap
                        ).sample

                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)

                samples.append(sample)

            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples

    def _decode_single_chunk(
        self,
        chunk_lat: torch.Tensor,
        device: torch.device,
        vae_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decode a single latent chunk, handling dtype/autocast automatically.

        Args:
            chunk_lat: Latent chunk [1, C, T_chunk, H, W] already on *device*.
            device: VAE device.
            vae_dtype: Actual dtype of the VAE model weights.

        Returns:
            Decoded sample [1, C, T_vid_chunk, H_out, W_out] on *device*.
        """
        if vae_dtype != chunk_lat.dtype:
            with torch.autocast(device.type, chunk_lat.dtype, enabled=True):
                return self.vae.decode(
                    chunk_lat,
                    tiled=self.decode_tiled,
                    tile_size=self.decode_tile_size,
                    tile_overlap=self.decode_tile_overlap,
                ).sample
        return self.vae.decode(
            chunk_lat,
            tiled=self.decode_tiled,
            tile_size=self.decode_tile_size,
            tile_overlap=self.decode_tile_overlap,
        ).sample

    def _decode_with_temporal_chunks(
        self,
        latent: torch.Tensor,
        device: torch.device,
        vae_dtype: torch.dtype,
        temporal_downsample_factor: int,
    ) -> torch.Tensor:
        """Decode a video latent tensor using temporal chunking to cut peak VRAM.

        The latent [1, C, T_lat, H, W] is split into overlapping 5-latent-frame chunks
        (17 video frames each at 4n+1).  Each chunk is decoded independently, moved to
        CPU memory immediately, and CUDA cache is freed.  Boundaries between adjacent
        chunks are smoothed via a linear alpha fade over the overlap region to prevent
        luminance jumps or flickering.

        Chunk geometry (4n+1 temporal format, factor=4):
          CHUNK_LAT  = 5  → 17 video frames per chunk
          OVERLAP_LAT = 2  → 5 video frames of blend zone at each boundary
          STRIDE_LAT  = 3  → 12 new video frames per chunk advance

        Args:
            latent: Video latent [1, C, T_lat, H, W] on CUDA.
            device: CUDA device of the VAE model.
            vae_dtype: Weight dtype of the VAE model.
            temporal_downsample_factor: Temporal downsampling factor (typically 4).

        Returns:
            Decoded video tensor [1, C, T_vid, H_out, W_out] back on *device*.
        """
        CHUNK_LAT = 5    # latent frames per chunk → (5-1)*4+1 = 17 video frames
        OVERLAP_LAT = 2  # latent frames shared between adjacent chunks
        STRIDE_LAT = CHUNK_LAT - OVERLAP_LAT  # = 3

        T_lat = latent.shape[2]
        T_vid_total = (T_lat - 1) * temporal_downsample_factor + 1

        # Build chunk start positions along the latent temporal dimension
        chunk_starts: List[int] = []
        s = 0
        while s + CHUNK_LAT <= T_lat:
            chunk_starts.append(s)
            s += STRIDE_LAT
        # Append an explicit last chunk if it would not be fully covered
        if not chunk_starts or chunk_starts[-1] + CHUNK_LAT < T_lat:
            last_start = T_lat - CHUNK_LAT
            if not chunk_starts or last_start > chunk_starts[-1]:
                chunk_starts.append(last_start)

        if not chunk_starts:
            # Safety fallback: T_lat < CHUNK_LAT – single-pass decode
            return self._decode_single_chunk(latent, device, vae_dtype)

        self.debug.log(
            f"Temporal chunked decode: {T_lat} latent frames, "
            f"{len(chunk_starts)} chunks (chunk={CHUNK_LAT}, overlap={OVERLAP_LAT})",
            category="vae", indent_level=1,
        )

        # ── Decode each chunk, offload to CPU immediately ──────────────────
        # Each entry: (cpu_tensor [1,C,T_vid_chunk,H,W], global_vid_start)
        decoded_chunks: List[Tuple[torch.Tensor, int]] = []

        for chunk_idx, s in enumerate(chunk_starts):
            e = min(s + CHUNK_LAT, T_lat)
            chunk_lat = latent[:, :, s:e, :, :]  # [1, C, T_chunk, H, W]

            chunk_sample = self._decode_single_chunk(chunk_lat, device, vae_dtype)

            global_vid_start = s * temporal_downsample_factor
            chunk_sample_cpu = chunk_sample.cpu()
            del chunk_sample
            torch.cuda.empty_cache()

            decoded_chunks.append((chunk_sample_cpu, global_vid_start))

            self.debug.log(
                f"  Chunk {chunk_idx + 1}/{len(chunk_starts)}: "
                f"lat [{s},{e}) → vid [{global_vid_start},"
                f"{global_vid_start + chunk_sample_cpu.shape[2]})",
                category="vae", indent_level=2,
            )

        # ── Assemble on CPU with linear blend at overlap boundaries ────────
        sample_ref = decoded_chunks[0][0]
        C_out, H_out, W_out = sample_ref.shape[1], sample_ref.shape[3], sample_ref.shape[4]

        output_cpu = torch.zeros(
            (1, C_out, T_vid_total, H_out, W_out),
            dtype=sample_ref.dtype,
            device='cpu',
        )

        prev_vid_end = 0  # exclusive end of the already-written region

        for chunk_sample_cpu, vid_start in decoded_chunks:
            T_vid_chunk = chunk_sample_cpu.shape[2]
            vid_end = vid_start + T_vid_chunk  # exclusive

            overlap_frames = max(0, prev_vid_end - vid_start)

            if overlap_frames > 0:
                # Linear alpha: 0 (keep previous) → 1 (take current) over overlap zone
                blend_len = min(overlap_frames, T_vid_total - vid_start)
                alpha = torch.linspace(0.0, 1.0, blend_len, dtype=sample_ref.dtype)
                alpha = alpha.view(1, 1, blend_len, 1, 1)  # broadcast over B,C,H,W

                prev_vals = output_cpu[:, :, vid_start : vid_start + blend_len]
                cur_vals = chunk_sample_cpu[:, :, :blend_len]
                output_cpu[:, :, vid_start : vid_start + blend_len] = (
                    prev_vals * (1.0 - alpha) + cur_vals * alpha
                )

            # Write non-overlapping tail of this chunk
            write_start = max(vid_start + overlap_frames, prev_vid_end)
            local_offset = write_start - vid_start
            write_end = min(vid_end, T_vid_total)
            if write_start < write_end:
                output_cpu[:, :, write_start:write_end] = (
                    chunk_sample_cpu[:, :, local_offset : local_offset + (write_end - write_start)]
                )

            prev_vid_end = max(prev_vid_end, vid_end)

        # Return on the original CUDA device
        return output_cpu.to(device)


    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps


    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []
        
        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale
        
        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)
        
        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)
        
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        latents = na.unflatten(latents, latents_shapes)

        # Clean up temporary tensors
        del latents_cond
        del latents_shapes
        del text_pos_embeds
        del text_neg_embeds
        del text_pos_shapes
        del text_neg_shapes
            
        return latents