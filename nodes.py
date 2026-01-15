import torch
import torch.nn.functional as F
import copy
import math
from tqdm.auto import tqdm

import comfy.utils
import comfy.model_management
import comfy.latent_formats
import folder_paths
from nodes import VAELoader
from .src.sd import CustomVAE
from .latent_upscale.model import latent_upscale_models


class VAEUtils_CustomVAELoader(VAELoader):
    @staticmethod
    def vae_list():
        return folder_paths.get_filename_list("vae")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (s.vae_list(), ),
                "disable_offload": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "VAE-Utils"

    def load_vae(self, vae_name, disable_offload):
        if vae_name == "pixel_space":
            sd = {}
            sd["pixel_space_vae"] = torch.tensor(1.0)
        elif vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = CustomVAE(sd=sd)
        vae.throw_exception_if_invalid()
        vae.disable_offload = disable_offload
        return (vae, )


class VAEUtils_DisableVAEOffload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", ),
                "disable_offload": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "set_offload"
    CATEGORY = "VAE-Utils"

    def set_offload(self, vae, disable_offload):
        vae = copy.copy(vae)
        vae.disable_offload = disable_offload
        return (vae, )


class VAEUtils_VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "upscale": ("INT", {"default": -1, "min": -1, "tooltip": "Post upscale factor, -1=auto"}),
                "tile": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 4096, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                "temporal_overlap": ("INT", {"default": 64, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "VAE-Utils"

    def decode(self, samples, vae, upscale, tile, tile_size, overlap, temporal_size, temporal_overlap):
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        temporal_compression = vae.temporal_compression_decode()
        if temporal_compression is not None:
            temporal_size = max(2, temporal_size // temporal_compression)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
        else:
            temporal_size = None
            temporal_overlap = None

        compression = vae.spacial_compression_decode()
        
        if tile:
            images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
        else:
            images = vae.decode(samples["samples"])
        
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        
        if upscale < 1:
            ch = images.shape[-1]
            if ch == 3:
                upscale = 1
            else:
                if ch % 3 == 0:
                    upscale = round((ch // 3) ** 0.5)
                else:
                    raise Exception("Couldn't determine upscale factor, try setting the value manually instead")
        
        images = F.pixel_shuffle(images.movedim(-1, 1), upscale_factor=int(upscale)).movedim(1, -1)
        return (images, )


class VAEUtils_LatentUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "model": (list(latent_upscale_models.keys()), ),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "VAE-Utils"
    
    def upscale(self, samples, model):
        device = comfy.model_management.get_torch_device()
        model = latent_upscale_models[model]().to(device)
        
        latents = samples["samples"].to(dtype=torch.float32, device=device)
        upscaled_latents = model(latents).to(comfy.model_management.intermediate_device())
        
        samples = copy.deepcopy(samples)
        samples["samples"] = upscaled_latents
        
        return (samples, )


def get_tiles(length, tile_size, min_overlap):
    if length <= tile_size:
        return [(0, length)]
    
    max_step = tile_size - min_overlap
    total_shiftable = length - tile_size
    
    gaps_needed = math.ceil(total_shiftable / max_step) if total_shiftable > 0 else 0
    num_tiles = gaps_needed + 1
    
    if num_tiles == 1:
        raise Exception("this shouldn't happen")

    gap_base = total_shiftable // (num_tiles - 1)
    remainder = total_shiftable % (num_tiles - 1)

    starts = []
    acc = 0
    for i in range(num_tiles):
        starts.append(acc)
        if i < num_tiles - 1:
            acc += gap_base + (1 if i < remainder else 0)

    slices = [(s, s + tile_size) for s in starts]
    return slices


def get_1d_mask(idx, tiles, drop_first=0):
    tile_start, tile_end = tiles[idx]
    mask = torch.ones(tile_end - tile_start)
    
    if idx > 0:
        prev_end = tiles[idx - 1][1]
        size = prev_end - tile_start - drop_first
        ramp = (torch.arange(size) + 1) / size
        mask[drop_first : size + drop_first] *= ramp
        mask[:drop_first] *= 0
    
    if idx < (len(tiles) - 1):
        next_start = tiles[idx + 1][0]
        size = tile_end - next_start
        ramp = (torch.flip(torch.arange(size), [0]) + 1) / size
        mask[-size:] *= ramp
    
    return mask


class VAEUtils_TileModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "tile_t": ("INT", {"default": 13, "min": 3, "step": 1}),
                "tile_h": ("INT", {"default": 78, "min": 32, "step": 2}),
                "tile_w": ("INT", {"default": 78, "min": 32, "step": 2}),
                "min_overlap_t": ("INT", {"default": 5, "min": 1, "step": 1}),
                "min_overlap_h": ("INT", {"default": 16, "min": 0, "step": 2}),
                "min_overlap_w": ("INT", {"default": 16, "min": 0, "step": 2}),
                "drop_first_t": ("INT", {"default": 2, "min": 0, "step": 1}),
                "patch_memory_estimate": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "VAE-Utils"
    
    def patch(self, model, tile_t, tile_h, tile_w, min_overlap_t, min_overlap_h, min_overlap_w, drop_first_t, patch_memory_estimate):
        model = model.clone()
        
        def model_function_tile_wrapper(apply_model, args):
            # print(args["c"]["transformer_options"]["patches"]["noise_refiner"][0].encoded_image.shape)
            
            # collect inputs
            x = args["input"]
            t = args["timestep"]
            c = args["c"]
            c_concat = c.get("c_concat", None)
            # transformer_options = c.get("transformer_options", {})
            
            if c_concat is not None:
                assert c_concat.shape[2:] == x.shape[2:], f"c_concat shape {c_concat.shape} doesn't match x shape {x.shape}"
            
            t_tiles = get_tiles(x.shape[-3], tile_t, min_overlap_t) if x.ndim == 5 else None
            h_tiles = get_tiles(x.shape[-2], tile_h, min_overlap_h)
            w_tiles = get_tiles(x.shape[-1], tile_w, min_overlap_w)
            
            total_tiles = len(h_tiles) * len(w_tiles)
            message = f"{len(h_tiles)}h {len(w_tiles)}w"
            
            if t_tiles is not None:
                total_tiles *= len(t_tiles)
                message = f"{len(t_tiles)}t " + message
            
            x_out = torch.zeros_like(x)
            x_alpha = torch.zeros_like(x)
            progress_bar = tqdm(range(0, total_tiles), desc=message)
            
            for w_idx, (w_start, w_end) in enumerate(w_tiles):
                for h_idx, (h_start, h_end) in enumerate(h_tiles):
                    if t_tiles is not None:
                        for t_idx, (t_start, t_end) in enumerate(t_tiles):
                            # slice inputs (todo: any other conditions? controlnets etc)
                            x_tile = x[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                            c_concat_tile = c_concat[:, :, t_start:t_end, h_start:h_end, w_start:w_end] if c_concat is not None else None
                            
                            c_tile = c.copy()
                            c_tile["c_concat"] = c_concat_tile
                            
                            tile_out = apply_model(x_tile, t, **c_tile)
                            
                            # prepare mask for edge blending
                            mask_t = get_1d_mask(t_idx, t_tiles, drop_first_t).view(1, 1, -1, 1, 1)
                            mask_h = get_1d_mask(h_idx, h_tiles).view(1, 1, 1, -1, 1)
                            mask_w = get_1d_mask(w_idx, w_tiles).view(1, 1, 1, 1, -1)
                            
                            mask = torch.ones_like(tile_out)
                            mask *= mask_t.to(mask)
                            mask *= mask_h.to(mask)
                            mask *= mask_w.to(mask)
                            
                            # accumulate tile into full image
                            x_out[:, :, t_start:t_end, h_start:h_end, w_start:w_end] += tile_out * mask
                            x_alpha[:, :, t_start:t_end, h_start:h_end, w_start:w_end] += mask
                            progress_bar.update(1)
                    
                    else:
                        x_tile = x[:, :, h_start:h_end, w_start:w_end]
                        c_concat_tile = c_concat[:, :, h_start:h_end, w_start:w_end] if c_concat is not None else None
                        
                        c_tile = c.copy()
                        c_tile["c_concat"] = c_concat_tile
                        
                        tile_out = apply_model(x_tile, t, **c_tile)
                        
                        mask_h = get_1d_mask(h_idx, h_tiles).view(1, 1, -1, 1)
                        mask_w = get_1d_mask(w_idx, w_tiles).view(1, 1, 1, -1)
                        
                        mask = torch.ones_like(tile_out)
                        mask *= mask_h.to(mask)
                        mask *= mask_w.to(mask)
                        
                        x_out[:, :, h_start:h_end, w_start:w_end] += tile_out * mask
                        x_alpha[:, :, h_start:h_end, w_start:w_end] += mask
                        progress_bar.update(1)
            
            assert x_alpha.min() > 0, "missing tile coverage"
            x_out = x_out / x_alpha
            return x_out
        
        model.set_model_unet_function_wrapper(model_function_tile_wrapper)
        
        def update_input_shape(input_shape):
            new_input_shape = input_shape.copy()
            if len(new_input_shape) == 5:
                new_input_shape = new_input_shape[:2] + [tile_t, tile_h, tile_w]
            elif len(new_input_shape) == 4:
                new_input_shape = new_input_shape[:2] + [tile_h, tile_w]
            return new_input_shape
        
        original_memory_required = model.model.memory_required
        
        def memory_required_wrapper(input_shape, cond_shapes={}):
            new_input_shape = update_input_shape(input_shape)
            
            for c in model.model.memory_usage_factor_conds:
                shape = cond_shapes.get(c, None)
                if shape is not None:
                    new_shape = update_input_shape(shape)
                    cond_shapes[c] = new_shape
            
            return original_memory_required(new_input_shape, cond_shapes)
        
        if patch_memory_estimate:
            model.model.memory_required = memory_required_wrapper
        
        return (model, )


class VAEUtils_VisualizeTiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "length": ("INT", {"default": 21, "min": 1, "step": 1}),
                "tile_size": ("INT", {"default": 13, "min": 1, "step": 1}),
                "min_overlap": ("INT", {"default": 5, "min": 0, "step": 1}),
                "drop_first": ("INT", {"default": 2, "min": 0, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "VAE-Utils"
    
    def visualize(self, length, tile_size, min_overlap, drop_first):
        tile_slices = get_tiles(length, tile_size, min_overlap)
        
        image = torch.zeros(1, 3, len(tile_slices), length)
        
        for idx, (start, end) in enumerate(tile_slices):
            print(idx, start, end)
            mask = get_1d_mask(idx, tile_slices, drop_first)
            image[:, :, idx, start:end] = mask
        
        image = F.interpolate(image, scale_factor=8, mode="nearest-exact")
        image = image.movedim(1, -1)
        
        return (image, )


latent_formats = {name: obj for name, obj in vars(comfy.latent_formats).items() if isinstance(obj, type)}
del latent_formats["LatentFormat"]


class VAEUtils_ScaleLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "direction": (["scale", "unscale"], ),
                "latent_type": (list(latent_formats.keys()), ),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "scale"
    CATEGORY = "VAE-Utils"

    def scale(self, latents, direction, latent_type):
        latents = copy.deepcopy(latents)
        latent_format = latent_formats[latent_type]()
        
        if direction == "scale":
            latents["samples"] = latent_format.process_in(latents["samples"])
        else:
            latents["samples"] = latent_format.process_out(latents["samples"])
        
        return (latents, )


COMBINED_MAPPINGS = {
    "VAEUtils_CustomVAELoader": (VAEUtils_CustomVAELoader, "Load VAE (VAE Utils)"),
    "VAEUtils_DisableVAEOffload": (VAEUtils_DisableVAEOffload, "Disable VAE Offload (VAE Utils)"),
    "VAEUtils_VAEDecodeTiled": (VAEUtils_VAEDecodeTiled, "VAE Decode (VAE Utils)"),
    "VAEUtils_LatentUpscale": (VAEUtils_LatentUpscale, "Latent Upscale (VAE Utils)"),
    "VAEUtils_TileModelPatch": (VAEUtils_TileModelPatch, "Tile Model Patch (VAE Utils)"),
    "VAEUtils_VisualizeTiles": (VAEUtils_VisualizeTiles, "Visualize Tiles (VAE Utils)"),
    "VAEUtils_ScaleLatents": (VAEUtils_ScaleLatents, "Scale/Unscale Latents (VAE Utils)"),
}