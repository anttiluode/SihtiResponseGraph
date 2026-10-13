"""A scene writes a small graph; each image query starts from a fresh ping."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


GRID = 8
CHANNELS = 8
IMAGE_SIZE = 32


@dataclass(frozen=True)
class GraphMemory:
    """Only per-scene edge conductances, with no retained node activations."""

    horizontal: Tensor  # [batch, channels, 8, 7]
    vertical: Tensor  # [batch, channels, 7, 8]

    def index_select(self, indices: Tensor) -> "GraphMemory":
        return GraphMemory(self.horizontal.index_select(0, indices),
                           self.vertical.index_select(0, indices))


def state_bytes(memory: GraphMemory) -> int:
    """Actual bytes occupied by the two edge tensors, including a batch."""
    return sum(t.numel() * t.element_size() for t in (memory.horizontal, memory.vertical))


def quantize_memory(memory: GraphMemory) -> GraphMemory:
    """Round the only stored per-instance state to one byte per edge channel."""
    return GraphMemory(*((t.clamp(0, 1) * 255).round().to(torch.uint8)
                         for t in (memory.horizontal, memory.vertical)))


class ResponseGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.edge_writer = nn.Conv2d(48, CHANNELS, 1)
        self.ping = nn.Conv2d(5, CHANNELS, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(CHANNELS * 4, 48, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(48, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 3, 1), nn.Sigmoid(),
        )

    def write(self, image: Tensor, previous: GraphMemory | None = None,
              visibility: Tensor | None = None) -> GraphMemory:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("image must be [B, 3, 32, 32]")
        if not torch.is_floating_point(image):
            raise ValueError("image must be floating point in [0, 1]")
        features = self.encoder(image)
        horizontal = self._edge_gate(features[..., :-1], features[..., 1:])
        vertical = self._edge_gate(features[..., :-1, :], features[..., 1:, :])
        proposal = GraphMemory(horizontal, vertical)
        if previous is None and visibility is None:
            return proposal
        if previous is None:
            previous = GraphMemory(torch.full_like(horizontal, 0.5),
                                   torch.full_like(vertical, 0.5))
        self._check_memory(previous, image.shape[0])
        if visibility is None:
            visibility = torch.ones_like(image[:, :1])
        if visibility.shape != (image.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("visibility must be [B, 1, 32, 32]")
        observed = F.adaptive_avg_pool2d(visibility.float().clamp(0, 1), (GRID, GRID))
        hmask = torch.minimum(observed[..., :-1], observed[..., 1:])
        vmask = torch.minimum(observed[..., :-1, :], observed[..., 1:, :])
        old_h, old_v = self._gates(previous)
        return GraphMemory(old_h * (1 - hmask) + horizontal * hmask,
                           old_v * (1 - vmask) + vertical * vmask)

    def _edge_gate(self, a: Tensor, b: Tensor) -> Tensor:
        # A positive floor prevents byte rounding from deleting weak edges.
        logits = self.edge_writer(torch.cat(((a + b) * 0.5, (a - b).abs()), dim=1))
        return 0.05 + 0.95 * torch.sigmoid(logits)

    @staticmethod
    def _check_memory(memory: GraphMemory, batch: int) -> None:
        if (memory.horizontal.shape != (batch, CHANNELS, GRID, GRID - 1)
                or memory.vertical.shape != (batch, CHANNELS, GRID - 1, GRID)):
            raise ValueError("memory has an incompatible batch or graph shape")

    @staticmethod
    def _gates(memory: GraphMemory) -> tuple[Tensor, Tensor]:
        if memory.horizontal.dtype == torch.uint8:
            return memory.horizontal.float() / 255, memory.vertical.float() / 255
        return memory.horizontal, memory.vertical

    @staticmethod
    def _propagate(h: Tensor, horizontal: Tensor, vertical: Tensor) -> Tensor:
        right = F.pad(horizontal * h[..., 1:], (0, 1, 0, 0))
        left = F.pad(horizontal * h[..., :-1], (1, 0, 0, 0))
        down = F.pad(vertical * h[..., 1:, :], (0, 0, 0, 1))
        up = F.pad(vertical * h[..., :-1, :], (0, 0, 1, 0))
        degree = (F.pad(horizontal, (0, 1, 0, 0))
                  + F.pad(horizontal, (1, 0, 0, 0))
                  + F.pad(vertical, (0, 0, 0, 1))
                  + F.pad(vertical, (0, 0, 1, 0)))
        return 0.5 * h + 0.5 * (right + left + down + up) / degree.clamp_min(1e-6)

    def _fresh_ping(self, query: Tensor, dtype: torch.dtype) -> Tensor:
        coordinates = torch.linspace(-1, 1, GRID, device=query.device, dtype=dtype)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        angle = query.to(dtype).view(-1, 1, 1) * (torch.pi / 2)
        xr = torch.cos(angle) * x + torch.sin(angle) * y
        yr = -torch.sin(angle) * x + torch.cos(angle) * y
        inputs = torch.stack((xr, yr, xr * yr, torch.sin(torch.pi * xr),
                              torch.cos(torch.pi * yr)), dim=1)
        return torch.tanh(self.ping(inputs))

    def read(self, memory: GraphMemory, query: Tensor) -> Tensor:
        batch = memory.horizontal.shape[0]
        self._check_memory(memory, batch)
        if query.shape != (batch,) or query.dtype not in (torch.int32, torch.int64):
            raise ValueError("query must be an integer tensor [B] in 0..3")
        if bool(((query < 0) | (query > 3)).any()):
            raise ValueError("query must be in 0..3")
        if query.device != memory.horizontal.device:
            raise ValueError("query and memory must be on the same device")
        horizontal, vertical = self._gates(memory)
        h = self._fresh_ping(query, horizontal.dtype)
        responses = [h]
        for depth in range(1, 9):
            h = self._propagate(h, horizontal, vertical)
            if depth in (2, 4, 8):
                responses.append(h)
        return self.decoder(torch.cat(responses, dim=1))

    def forward(self, image: Tensor, query: Tensor) -> Tensor:
        return self.read(self.write(image), query)
