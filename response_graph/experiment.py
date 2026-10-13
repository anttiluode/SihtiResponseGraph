"""Identical-data training and byte-matched evaluation of two image memories."""

import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baseline import VectorMemoryModel, quantize_vector, vector_bytes
from .data import GlyphViews
from .model import GraphMemory, ResponseGraph, quantize_memory, state_bytes


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cpu", "cuda") or (name == "cuda" and not torch.cuda.is_available()):
        raise ValueError("device must be cpu, an available cuda, or auto")
    return torch.device(name)


def _average_graph(model: ResponseGraph, dataset: GlyphViews,
                   device: torch.device, batch_size: int) -> GraphMemory:
    """Fixed substrate from training objects only (never validation objects)."""
    was_training = model.training
    model.eval()
    count = 0
    horizontal = vertical = None
    try:
        with torch.inference_mode():
            # A fixed, documented subset keeps the control cheap on CPU.
            for source, _, _ in DataLoader(dataset, batch_size=batch_size,
                                           sampler=range(min(64, len(dataset)))):
                memory = model.write(source.to(device))
                h, v = memory.horizontal.sum(0, keepdim=True), memory.vertical.sum(0, keepdim=True)
                horizontal = h if horizontal is None else horizontal + h
                vertical = v if vertical is None else vertical + v
                count += source.shape[0]
    finally:
        model.train(was_training)
    return GraphMemory(horizontal / count, vertical / count)


def evaluate(model: nn.Module, loader: DataLoader, device: str | torch.device,
             fixed_graph: GraphMemory | None = None) -> dict:
    """Evaluate quantized, *stored* state; preserve both weights and mode."""
    device = torch.device(device)
    if fixed_graph is not None and not isinstance(model, ResponseGraph):
        raise ValueError("fixed_graph applies only to ResponseGraph")
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    pose_sums, pose_counts = [0.0] * 4, [0] * 4
    bytes_per_scene = None
    try:
        with torch.inference_mode():
            for source, query, target in loader:
                source, query, target = source.to(device), query.to(device), target.to(device)
                if isinstance(model, ResponseGraph):
                    if fixed_graph is None:
                        memory = model.write(source)
                    else:
                        memory = GraphMemory(fixed_graph.horizontal.expand(source.shape[0], -1, -1, -1),
                                             fixed_graph.vertical.expand(source.shape[0], -1, -1, -1))
                    compact = quantize_memory(memory)
                    bytes_used = state_bytes(compact)
                else:
                    compact = quantize_vector(model.write(source))
                    bytes_used = vector_bytes(compact)
                prediction = model.read(compact, query)
                errors = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
                total += errors.sum().item()
                count += source.shape[0]
                observed_bytes = bytes_used // source.shape[0]
                if bytes_per_scene is not None and bytes_per_scene != observed_bytes:
                    raise RuntimeError("inconsistent serialized memory size")
                bytes_per_scene = observed_bytes
                for pose in range(4):
                    selected = errors[query == pose]
                    pose_sums[pose] += selected.sum().item()
                    pose_counts[pose] += selected.numel()
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("evaluation loader is empty")
    mse = total / count
    by_query = {}
    for pose in range(4):
        if pose_counts[pose]:
            value = pose_sums[pose] / pose_counts[pose]
            by_query[str(pose)] = {"mse": value,
                                   "psnr_db": -10 * math.log10(max(value, 1e-12)),
                                   "count": pose_counts[pose]}
    return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12)),
            "by_query": by_query, "memory_bytes": bytes_per_scene,
            "parameters": sum(p.numel() for p in model.parameters())}


def _pil(image: torch.Tensor) -> Image.Image:
    pixels = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _save_contact_sheet(graph: ResponseGraph, vector: VectorMemoryModel,
                        dataset: GlyphViews, device: torch.device, path: Path) -> None:
    tile, gap, rows = 96, 10, min(4, len(dataset))
    sheet = Image.new("RGB", (4 * (tile + gap) + gap, rows * (tile + 28) + gap), (25, 29, 37))
    draw = ImageDraw.Draw(sheet)
    for row in range(rows):
        source, query, target = dataset[row]
        batch, q = source[None].to(device), query[None].to(device)
        with torch.inference_mode():
            image_graph = graph.read(quantize_memory(graph.write(batch)), q)[0]
            image_vector = vector.read(quantize_vector(vector.write(batch)), q)[0]
        for col, (label, image) in enumerate(zip(
                ("source", "target", "graph", "vector"),
                (source, target, image_graph, image_vector))):
            x, y = gap + col * (tile + gap), gap + row * (tile + 28)
            sheet.paste(_pil(image).resize((tile, tile), Image.Resampling.NEAREST), (x, y + 17))
            draw.text((x, y), f"{label} q={int(query)}", fill=(238, 238, 242))
    sheet.save(path)


def run_experiment(steps: int, batch_size: int, seed: int, device: str,
                   output_dir: Path, train_count: int = 2048,
                   val_count: int = 128) -> dict:
    if steps <= 0 or batch_size <= 0 or train_count < batch_size or val_count < 4:
        raise ValueError("steps and batch_size must be positive; train_count >= batch_size; val_count >= 4")
    device_obj = _device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_seed, val_seed = seed, seed + 1_000_003
    train_data = GlyphViews(train_seed, train_count)
    val_data = GlyphViews(val_seed, val_count)
    shuffle = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=shuffle)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    graph, vector = ResponseGraph().to(device_obj), VectorMemoryModel().to(device_obj)
    models = {"graph": graph, "vector": vector}
    optimizers = {key: torch.optim.Adam(model.parameters(), lr=1e-3)
                  for key, model in models.items()}
    stream = iter(train_loader)
    last_loss = {}
    for _ in range(steps):
        try:
            source, query, target = next(stream)
        except StopIteration:
            stream = iter(train_loader)
            source, query, target = next(stream)
        source, query, target = source.to(device_obj), query.to(device_obj), target.to(device_obj)
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            loss = F.mse_loss(model(source, query), target)
            loss.backward()
            optimizers[name].step()
            last_loss[name] = loss.item()
    fixed = _average_graph(graph, train_data, device_obj, batch_size)
    result = {"seed": seed, "train_seed": train_seed, "val_seed": val_seed,
              "train_count": train_count, "val_count": val_count, "steps": steps,
              "batch_size": batch_size, "device": str(device_obj),
              "fixed_graph_mean_count": min(64, train_count),
              "last_train_mse": last_loss}
    for name, model in models.items():
        start = time.perf_counter()
        result[name] = evaluate(model, val_loader, device_obj)
        result[name]["evaluation_seconds"] = time.perf_counter() - start
    result["fixed_graph"] = evaluate(graph, val_loader, device_obj, fixed_graph=fixed)
    torch.save({"architecture": "ResponseGraph", "state_dict": graph.state_dict(),
                "seed": seed, "steps": steps}, output_dir / "graph.pt")
    torch.save({"architecture": "VectorMemoryModel", "state_dict": vector.state_dict(),
                "seed": seed, "steps": steps}, output_dir / "vector.pt")
    _save_contact_sheet(graph.eval(), vector.eval(), val_data, device_obj,
                        output_dir / "predictions.png")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
