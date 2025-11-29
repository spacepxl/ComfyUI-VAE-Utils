import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class DiCoBlock3d(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, kernel_size=3):
        super().__init__()

        # self.conv1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        
        self.conv2 = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size//2, kernel_size//2),
            # groups=hidden_size,
            padding_mode="replicate",
        )
        
        self.conv3 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        
        self.cca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_size, hidden_size , kernel_size=1),
            nn.Sigmoid(),
        )

        ffn_channel = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv3d(hidden_size, ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv3d(ffn_channel, hidden_size, kernel_size=1)

        self.norm1 = LayerNorm3d(hidden_size, affine=False)
        self.norm2 = LayerNorm3d(hidden_size, affine=False)

    def forward(self, inp):
        x = self.norm1(inp)
        # x = F.gelu(self.conv2(self.conv1(x)))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x * self.cca(x))
        x = inp + x

        y = self.norm2(x)
        y = self.conv5(F.gelu(self.conv4(y)))
        return x + y


class LatentModel3d(nn.Module):
    def __init__(
        self,
        in_channels = 16,
        out_channels = 16,
        hidden_size = 128,
        kernel_size = 3,
        mlp_ratio = 4.0,
        depth = 8,
        upscale = 2,
    ):
        super().__init__()
        self.upscale = int(upscale)

        self.conv_in = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )
        
        self.blocks = nn.ModuleList([
            DiCoBlock3d(hidden_size, mlp_ratio, kernel_size)
            for _ in range(depth)
        ])
        
        out_ch = out_channels * self.upscale ** 2
        self.conv_out = nn.Conv3d(
            hidden_size,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )
        
        # self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x):
        """
        x: (B, C, F, H, W) tensor of spatial inputs (images or latent representations of images)
        """
        hidden = self.conv_in(x)
        
        for block in self.blocks:
            hidden = block(hidden)
        
        output = self.conv_out(hidden)
        
        if self.upscale > 1:
            output = F.pixel_shuffle(output.movedim(1, 2), self.upscale).movedim(2, 1)
            x = F.interpolate(x, scale_factor=(1, self.upscale, self.upscale), mode="nearest-exact")
        return output + x


def Wan21_latent_upscale_2x():
    model = LatentModel3d(
        in_channels = 16,
        out_channels = 16,
        hidden_size = 128,
        kernel_size = 3,
        mlp_ratio = 4.0,
        depth = 8,
        upscale = 2,
    )
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan21_latent_upscale_2x.safetensors")
    model.load_state_dict(load_file(model_path))
    return model


latent_upscale_models = {
    "Wan 2.1 latent upscale 2x": Wan21_latent_upscale_2x,
}


if __name__=="__main__":
    model = LatentModel3d(upscale=2).to("cuda")
    
    x = torch.randn(1, 16, 4, 64, 64).to("cuda")
    out = model(x)
    print(out.shape)
    
    total = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("trainable parameters: %.2f M" % (total / 1e6))
    
    print(model)