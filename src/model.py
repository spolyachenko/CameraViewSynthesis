import torch
import torch.nn as nn

from timm.layers.drop import DropPath
from torchinfo import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_chance=0.2, layer_scale_init_value=1e-6) -> None:
        super().__init__()

        self.norm = nn.GroupNorm(min(32, in_channels // 8), in_channels, eps=1e-6)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_chance) if drop_chance > 0. else nn.Identity()

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
                
        x = self.norm(x)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        
        if self.gamma is not None:
            x = (self.gamma * x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
        x = self.skip(residual) + self.drop_path(x)
        return x

class ReSampler(nn.Module):
    def __init__(self, num_channels, direction='down') -> None:
        super().__init__()

        self.norm = nn.GroupNorm(min(32, num_channels // 8), num_channels, eps=1e-6)
        self.resample = (
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1) if direction == 'down' 
            else nn.ConvTranspose2d(num_channels, num_channels // 2, 2, 2)
        )

    def forward(self, x) -> torch.Tensor:        
        x = self.resample(self.norm(x))
        return x

class CustomUNet(nn.Module):
    def __init__(self, depth, input_dim):
        super().__init__()

        # 6x512x512
        self.depth = depth

        self.stem = nn.Conv2d(input_dim, 32, kernel_size=7, padding=3)

        self.contracting_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()     
        last_dim = 32
        for i in range(depth):
            self.contracting_blocks.append(ResidualBlock(last_dim, last_dim*2))
            self.downsamplers.append(ReSampler(last_dim*2, direction='down'))
            last_dim *= 2

        # 512x32x32
        self.inv_bottleneck = ResidualBlock(last_dim, last_dim * 2)
        last_dim *= 2

        self.upsamplers = nn.ModuleList()
        self.expanding_blocks = nn.ModuleList()
        for i in range(depth):
            self.upsamplers.append(ReSampler(last_dim, direction='up'))
            self.expanding_blocks.append(ResidualBlock(last_dim, last_dim//2))
            last_dim //= 2
        
        self.head = nn.Conv2d(last_dim, 3, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)

        skip_connections = []
        for i in range(self.depth):
            x = self.contracting_blocks[i](x)
            skip_connections.append(x)
            x = self.downsamplers[i](x)
        
        x = self.inv_bottleneck(x)

        for i in range(self.depth):
            x = self.upsamplers[i](x)
            x = torch.cat([x, skip_connections[self.depth-i-1]], dim=1)
            x = self.expanding_blocks[i](x)

        x = self.head(x)
        return x

def load_checkpoint(path, device=None) -> CustomUNet:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", {"depth": 4, "input_dim": 6})
    model = CustomUNet(**arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = CustomUNet(depth=4, input_dim=6).to(device) 
    tens = torch.randn((1, 6, 512, 512)).to(device)

    summary(model, input_size=tens.size())

    out = model(tens)