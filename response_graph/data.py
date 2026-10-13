"""Deterministic paired 2D views for the first response-graph gate."""

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


class GlyphViews(Dataset):
    """An asymmetric colored object and its requested quarter-turn.

    Instances are generated per index, so train and validation can use
    disjoint seeds without downloading or retaining a dataset archive.
    """

    def __init__(self, seed: int, count: int):
        if count <= 0:
            raise ValueError("count must be positive")
        self.seed = int(seed)
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng((self.seed * 10_000_019 + index) % (2**64))
        image = Image.new("RGB", (32, 32), (12, 16, 24))
        draw = ImageDraw.Draw(image)
        x, y = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        width, height = int(rng.integers(11, 19)), int(rng.integers(11, 19))
        body = tuple(int(z) for z in rng.integers(70, 236, size=3))
        arm = tuple(int(z) for z in rng.integers(70, 236, size=3))
        mark = tuple(int(z) for z in rng.integers(100, 256, size=3))
        draw.rectangle((x, y, x + width, y + height), fill=body)
        arm_length = int(rng.integers(3, 7))
        draw.rectangle((x + width - 3, y + height - 5,
                        min(31, x + width + arm_length), y + height - 2), fill=arm)
        mx, my = x + int(rng.integers(2, min(6, width))), y + 2
        draw.ellipse((mx, my, mx + 2, my + 2), fill=mark)
        source = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        query = torch.tensor((index + self.seed) % 4, dtype=torch.long)
        return source, query, torch.rot90(source, int(query), dims=(-2, -1))
