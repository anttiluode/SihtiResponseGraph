"""Load a trained graph and render a quarter-turn query from one image."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .model import ResponseGraph, quantize_memory


def render_query(checkpoint: Path, input_image: Path, query: int,
                 output_image: Path, device: str = "cpu") -> None:
    checkpoint, input_image, output_image = map(Path, (checkpoint, input_image, output_image))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    if not input_image.is_file():
        raise FileNotFoundError(f"input image missing: {input_image}")
    if query not in range(4):
        raise ValueError("query must be 0, 1, 2, or 3 quarter-turns")
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    if loaded.get("architecture") != "ResponseGraph":
        raise ValueError("checkpoint does not contain a ResponseGraph")
    model = ResponseGraph().to(device).eval()
    model.load_state_dict(loaded["state_dict"])
    with Image.open(input_image) as image:
        pixels = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)).copy()
    source = torch.from_numpy(pixels).permute(2, 0, 1).float()[None].to(device) / 255
    with torch.inference_mode():
        memory = quantize_memory(model.write(source))
        result = model.read(memory, torch.tensor([query], device=device))[0]
    image = (result.cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "RGB").save(output_image)
