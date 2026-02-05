import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class CausalLatentProjector(nn.Module):
    def __init__(
        self,
        in_channels = 16,
        out_channels = 3,
        scale_t = 4,
        scale_s = 8,
        num_projections = 3,
    ):
        super().__init__()
        self.scale_t = scale_t
        self.scale_s = scale_s
        
        self.projections = nn.ModuleList([
            nn.Linear(in_channels, out_channels * scale_s ** 2)
        ])
        
        for _ in range(num_projections - 1):
            self.projections.append(
                nn.Linear(in_channels, out_channels * scale_t * scale_s ** 2)
            )

    def forward(self, latents):
        batch, channels, frames, height, width = latents.shape
        
        pixels = []
        for frame in range(frames):
            idx = min(frame, len(self.projections) - 1)
            
            feat = self.projections[idx](
                latents[:, :, frame].movedim(1, -1) # BHWC
            ).movedim(-1, 1) # BCHW
            
            feat = F.pixel_shuffle(feat, self.scale_s).unsqueeze(2) # BCFHW
            
            if frame == 0:
                pixels.append(feat)
            else:
                feat = feat.chunk(self.scale_t, dim=1)
                pixels.extend(list(feat))
        
        return torch.cat(pixels, dim=2)


def Wan21_latent_projector():
    model = CausalLatentProjector()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan21_latent_projector.safetensors")
    model.load_state_dict(load_file(model_path))
    return model