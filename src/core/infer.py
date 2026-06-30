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
    classifier_free_guidance,
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

    def _configure_vae_runtime(self):
        dtype = getattr(torch, self.config.vae.dtype)
        try:
            vae_param = next(self.vae.parameters())
            device = vae_param.device
            vae_dtype = vae_param.dtype
        except StopIteration:
            device = get_device()
            vae_dtype = dtype

        signature = (id(self.vae), device, vae_dtype, dtype)
        if getattr(self, "_vae_runtime_signature", None) == signature:
            return self._vae_runtime

        scale = self.config.vae.scaling_factor
        shift = self.config.vae.get("shifting_factor", 0.0)

        if isinstance(scale, ListConfig):
            scale = torch.as_tensor(scale, device=device, dtype=dtype)
        if isinstance(shift, ListConfig):
            shift = torch.as_tensor(shift, device=device, dtype=dtype)

        self._vae_runtime_signature = signature
        self._vae_runtime = {
            "device": device,
            "dtype": dtype,
            "vae_dtype": vae_dtype,
            "scale": scale,
            "shift": shift,
        }
        return self._vae_runtime

    @staticmethod
    def _concat_cfg_inputs(pos, neg):
        if isinstance(pos, list):
            return [torch.cat([pos_item, neg_item], dim=0) for pos_item, neg_item in zip(pos, neg)]
        return torch.cat([pos, neg], dim=0)

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        """VAE encode with configured dtype - converts samples to latents with optional tiling"""
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            vae_runtime = self._configure_vae_runtime()
            device = vae_runtime["device"]
            dtype = vae_runtime["dtype"]
            vae_dtype = vae_runtime["vae_dtype"]
            scale = vae_runtime["scale"]
            shift = vae_runtime["shift"]

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # VAE process by each group.
            for sample in batches:
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)

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
            vae_runtime = self._configure_vae_runtime()
            device = vae_runtime["device"]
            dtype = vae_runtime["dtype"]
            vae_dtype = vae_runtime["vae_dtype"]
            scale = vae_runtime["scale"]
            shift = vae_runtime["shift"]

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

                use_temporal_chunks = (
                    latent.ndim == 5
                    and latent.shape[2] > 9
                    and device.type == 'cuda'
                )

                if use_temporal_chunks:
                    sample = self._decode_with_temporal_chunks(latent).sample
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

    def _decode_with_temporal_chunks(
        self,
        latent: torch.Tensor,
    ):
        """Decode the full latent tensor in a single pass."""
        with torch.autocast(latent.device.type, dtype=torch.float16, enabled=(latent.device.type == "cuda")):
            return self.vae.decode(
                latent,
                tiled=self.decode_tiled,
                tile_size=self.decode_tile_size,
                tile_overlap=self.decode_tile_overlap,
            )


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

        cfg_partial = self.config.diffusion.cfg.get("partial", 1)
        cfg_rescale = self.config.diffusion.cfg.rescale
        sampler_steps = len(self.sampler.timesteps)
        cfg_text_embeds = self._concat_cfg_inputs(text_pos_embeds, text_neg_embeds)
        cfg_text_shapes = self._concat_cfg_inputs(text_pos_shapes, text_neg_shapes)
        cfg_latents_shapes = torch.cat([latents_shapes, latents_shapes], dim=0)

        def diffusion_step(args):
            scale = (
                cfg_scale
                if (args.i + 1) / sampler_steps <= cfg_partial
                else 1.0
            )
            timestep = args.t.repeat(batch_size)
            model_input = torch.cat([args.x_t, latents_cond], dim=-1)
            if scale == 1.0:
                return self.dit(
                    vid=model_input,
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=timestep,
                ).vid_sample

            cfg_output = self.dit(
                vid=torch.cat([model_input, model_input], dim=0),
                txt=cfg_text_embeds,
                vid_shape=cfg_latents_shapes,
                txt_shape=cfg_text_shapes,
                timestep=timestep.repeat(2),
            ).vid_sample
            pos, neg = torch.chunk(cfg_output, 2, dim=0)
            return classifier_free_guidance(
                pos=pos,
                neg=neg,
                scale=scale,
                rescale=cfg_rescale,
            )

        latents = self.sampler.sample(
            x=latents,
            f=diffusion_step,
        )

        latents = na.unflatten(latents, latents_shapes)
        return latents