import copy
import math
import types

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils


def is_wan_upscale_vae(vae) -> bool:
    output_channels = getattr(vae, "output_channels", None)
    conv_out_channels = getattr(vae, "conv_out_channels", None)
    if getattr(vae, "latent_dim", None) != 3:
        return False
    if not isinstance(output_channels, int) or not isinstance(conv_out_channels, int):
        return False
    if output_channels <= 0 or conv_out_channels <= output_channels:
        return False
    ratio_squared, remainder = divmod(conv_out_channels, output_channels)
    ratio = math.isqrt(ratio_squared)
    return remainder == 0 and ratio > 1 and ratio * ratio == ratio_squared


def _finish_decode(vae, images):
    if images.numel() > 0 and images.min() < -0.1:
        images = torch.clamp((images.float() + 1.0) / 2.0, min=0.0, max=1.0)

    channels = images.shape[-1]
    if channels != vae.output_channels:
        ratio_squared, remainder = divmod(channels, vae.output_channels)
        ratio = math.isqrt(ratio_squared)
        if remainder != 0 or ratio * ratio != ratio_squared:
            raise RuntimeError(
                f"Cannot unpack {channels} decoder channels into "
                f"{vae.output_channels} image channels."
            )
        images = F.pixel_shuffle(images.movedim(-1, -3), ratio).movedim(-3, -1)

    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images


def _decode_tiled_3d(vae, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
    decode_fn = lambda value: vae.first_stage_model.decode(
        value.to(vae.vae_dtype).to(vae.device)
    ).to(dtype=vae.vae_output_dtype())
    return vae.process_output(
        comfy.utils.tiled_scale_multidim(
            samples,
            decode_fn,
            tile=(tile_t, tile_x, tile_y),
            overlap=overlap,
            upscale_amount=vae.upscale_ratio,
            out_channels=vae.conv_out_channels,
            index_formulas=vae.upscale_index_formula,
            output_device=vae.output_device,
        )
    )


def _decode(vae, samples, vae_options={}):
    images = vae._vae_utils_original_decode(vae, samples, vae_options)
    return _finish_decode(vae, images)


def _decode_tiled(vae, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
    images = vae._vae_utils_original_decode_tiled(
        vae,
        samples,
        tile_x=tile_x,
        tile_y=tile_y,
        overlap=overlap,
        tile_t=tile_t,
        overlap_t=overlap_t,
    )
    return _finish_decode(vae, images)


def patch_wan_upscale_vae(vae):
    if not is_wan_upscale_vae(vae):
        raise ValueError(
            "Patch Wan Upscale VAE requires a Core-loaded 3D VAE with packed decoder channels."
        )
    if getattr(vae, "_vae_utils_wan_upscale_patch", False):
        return copy.copy(vae)

    patched = copy.copy(vae)
    patched._vae_utils_wan_upscale_patch = True
    patched._vae_utils_original_decode = vae.decode.__func__
    patched._vae_utils_original_decode_tiled = vae.decode_tiled.__func__
    patched.decode = types.MethodType(_decode, patched)
    patched.decode_tiled = types.MethodType(_decode_tiled, patched)
    patched.decode_tiled_3d = types.MethodType(_decode_tiled_3d, patched)
    return patched


def set_vae_offload_policy(vae, disable_offload):
    patched = copy.copy(vae)
    patched.patcher = vae.patcher.clone()
    patched.patcher.offload_device = (
        patched.patcher.load_device
        if disable_offload
        else comfy.model_management.vae_offload_device()
    )
    patched.disable_offload = disable_offload
    return patched
