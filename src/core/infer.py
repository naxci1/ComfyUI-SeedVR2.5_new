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
        """Decode a video latent tensor using macro-chunk temporal splitting.

        The latent [1, C, T_lat, H, W] is split into at most 3 large macro-chunks
        (2 chunks for shorter sequences, 3 for longer ones).  All decoding and
        blending is done natively on GPU; no CPU offloading or cache-clearing occurs
        inside the decode loop, keeping the CUDA allocator efficient.

        Macro-chunk geometry:
          num_chunks  = 2  (T_lat ≤ 25) or 3  (T_lat > 25)
          OVERLAP_LAT = 2  latent frames (≈ 4 video frames) at each boundary
          chunk_size  = ceil(T_lat / num_chunks) with overlap added to each boundary

        Pipeline:
          1. Enable VAE slicing if the model supports it.
          2. Wrap all decoding in torch.inference_mode() and force bfloat16.
          3. Clone each slice to break the parent-tensor reference before passing to VAE.
          4. Delete intermediate tensors explicitly; let the allocator reclaim freely.
          5. Blend overlap boundaries with a linear alpha ramp (0 → 1) on GPU.
          6. Concatenate and return on the original CUDA device.

        Args:
            latent: Video latent [1, C, T_lat, H, W] on CUDA.
            device: CUDA device of the VAE model.
            vae_dtype: Weight dtype of the VAE model.
            temporal_downsample_factor: Temporal downsampling factor (typically 4).

        Returns:
            Decoded video tensor [1, C, T_vid, H_out, W_out] on *device*.
        """
        OVERLAP_LAT = 2  # latent frames shared at each boundary → ~4 video frames

        T_lat = latent.shape[2]

        # Enable VAE slicing for lower-level VRAM savings when supported
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()

        # Choose 2 or 3 macro-chunks based on sequence length
        num_chunks = 3 if T_lat > 25 else 2

        # Compute evenly-sized chunk boundaries (start, end) in latent frames
        # Each boundary except the last shares OVERLAP_LAT frames with its neighbour
        base = T_lat // num_chunks
        remainder = T_lat % num_chunks
        boundaries: List[Tuple[int, int]] = []
        s = 0
        for i in range(num_chunks):
            chunk_len = base + (1 if i < remainder else 0)
            e = s + chunk_len
            # Extend right boundary by overlap (clamp to T_lat)
            e_ext = min(e + OVERLAP_LAT, T_lat) if i < num_chunks - 1 else T_lat
            # Extend left boundary by overlap so blend zone is covered
            s_ext = max(s - OVERLAP_LAT, 0) if i > 0 else 0
            boundaries.append((s_ext, e_ext))
            s = e

        if len(boundaries) == 0:
            return self._decode_single_chunk(latent, device, vae_dtype)

        self.debug.log(
            f"Macro-chunk decode: {T_lat} latent frames → {len(boundaries)} chunks "
            f"(overlap={OVERLAP_LAT}): {boundaries}",
            category="vae", indent_level=1,
        )

        # ── Decode each macro-chunk entirely on GPU ─────────────────────────
        decoded_chunks: List[Tuple[torch.Tensor, int]] = []  # (tensor on GPU, vid_start)

        with torch.inference_mode():
            decode_dtype = torch.bfloat16

            for chunk_idx, (cs, ce) in enumerate(boundaries):
                # .clone() breaks the view relationship with the parent latent tensor,
                # allowing the graph / reference count on older slices to drop to zero.
                chunk_lat = latent[:, :, cs:ce, :, :].clone()

                if chunk_lat.dtype != decode_dtype:
                    chunk_lat = chunk_lat.to(decode_dtype)

                chunk_vae_dtype = decode_dtype
                if chunk_vae_dtype != vae_dtype:
                    with torch.autocast(device.type, decode_dtype, enabled=True):
                        chunk_sample = self.vae.decode(
                            chunk_lat,
                            tiled=self.decode_tiled,
                            tile_size=self.decode_tile_size,
                            tile_overlap=self.decode_tile_overlap,
                        ).sample
                else:
                    chunk_sample = self.vae.decode(
                        chunk_lat,
                        tiled=self.decode_tiled,
                        tile_size=self.decode_tile_size,
                        tile_overlap=self.decode_tile_overlap,
                    ).sample

                global_vid_start = cs * temporal_downsample_factor
                decoded_chunks.append((chunk_sample, global_vid_start))

                self.debug.log(
                    f"  Chunk {chunk_idx + 1}/{len(boundaries)}: "
                    f"lat [{cs},{ce}) → vid [{global_vid_start},"
                    f"{global_vid_start + chunk_sample.shape[2]})",
                    category="vae", indent_level=2,
                )

                # Explicitly release the cloned latent slice; allocator reclaims freely
                del chunk_lat

        # ── Assemble on GPU with linear blend at overlap boundaries ─────────
        T_vid_total = (T_lat - 1) * temporal_downsample_factor + 1
        sample_ref = decoded_chunks[0][0]
        C_out = sample_ref.shape[1]
        H_out = sample_ref.shape[3]
        W_out = sample_ref.shape[4]

        output = torch.zeros(
            (1, C_out, T_vid_total, H_out, W_out),
            dtype=sample_ref.dtype,
            device=device,
        )

        prev_vid_end = 0  # exclusive end of the already-written region

        for chunk_sample, vid_start in decoded_chunks:
            T_vid_chunk = chunk_sample.shape[2]
            vid_end = min(vid_start + T_vid_chunk, T_vid_total)

            overlap_frames = max(0, prev_vid_end - vid_start)

            if overlap_frames > 0:
                blend_len = min(overlap_frames, T_vid_total - vid_start)
                # Linear alpha 0→1 ramp on GPU
                alpha = torch.linspace(
                    0.0, 1.0, blend_len, dtype=output.dtype, device=device
                ).view(1, 1, blend_len, 1, 1)

                prev_vals = output[:, :, vid_start : vid_start + blend_len].clone()
                cur_vals = chunk_sample[:, :, :blend_len]
                output[:, :, vid_start : vid_start + blend_len] = (
                    prev_vals * (1.0 - alpha) + cur_vals * alpha
                )
                del prev_vals, alpha

            write_start = max(vid_start + overlap_frames, prev_vid_end)
            local_offset = write_start - vid_start
            write_end = min(vid_end, T_vid_total)
            if write_start < write_end:
                output[:, :, write_start:write_end] = (
                    chunk_sample[:, :, local_offset : local_offset + (write_end - write_start)]
                )

            prev_vid_end = max(prev_vid_end, vid_end)
            del chunk_sample

        return output


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