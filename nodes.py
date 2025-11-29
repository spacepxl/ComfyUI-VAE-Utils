import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths
from nodes import VAELoader
from .src.sd import CustomVAE


class VAEUtils_CustomVAELoader(VAELoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(s), )}}
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "VAE-Utils"

    def load_vae(self, vae_name):
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
        return (vae,)


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
        return (images,)


COMBINED_MAPPINGS = {
    "VAEUtils_CustomVAELoader": (VAEUtils_CustomVAELoader, "Load VAE (VAE Utils)"),
    "VAEUtils_VAEDecodeTiled": (VAEUtils_VAEDecodeTiled, "VAE Decode (VAE Utils)"),
}