"""Equal-byte vector memory for the same queried image task."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import IMAGE_SIZE


VECTOR_CHANNELS = 14  # 14 × 8 × 8 = 896 values


def vector_bytes(memory: Tensor) -> int:
    return memory.numel() * memory.element_size()


def quantize_vector(memory: Tensor) -> Tensor:
    return (memory.clamp(0, 1) * 255).round().to(torch.uint8)


class VectorMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, VECTOR_CHANNELS, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(VECTOR_CHANNELS + 4, 48, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 1), nn.Sigmoid(),
        )

    def write(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("image must be [B, 3, 32, 32]")
        return self.encoder(image).flatten(1)

    def read(self, memory: Tensor, query: Tensor) -> Tensor:
        if memory.ndim != 2 or memory.shape[1] != 896:
            raise ValueError("vector memory must be [B, 896]")
        batch = memory.shape[0]
        if query.shape != (batch,) or query.dtype not in (torch.int32, torch.int64):
            raise ValueError("query must be an integer tensor [B] in 0..3")
        if bool(((query < 0) | (query > 3)).any()):
            raise ValueError("query must be in 0..3")
        if query.device != memory.device:
            raise ValueError("query and memory must be on the same device")
        vector = memory.float() / 255 if memory.dtype == torch.uint8 else memory
        code = vector.reshape(batch, VECTOR_CHANNELS, 8, 8)
        ping = F.one_hot(query.long(), 4).to(code.dtype)[:, :, None, None]
        return self.decoder(torch.cat((code, ping.expand(-1, -1, 8, 8)), dim=1))

    def forward(self, image: Tensor, query: Tensor) -> Tensor:
        return self.read(self.write(image), query)
