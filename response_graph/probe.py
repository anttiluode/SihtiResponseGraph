"""Probe whether a frozen image writer stored a small identifying mark."""

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .baseline import VectorMemoryModel, quantize_vector
from .data import GlyphViews
from .model import ResponseGraph, quantize_memory


def _mark_parts(image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pixels = (image.detach().cpu().permute(1, 2, 0).reshape(-1, 3).clamp(0, 1) * 255).round().to(torch.uint8)
    colors, inverse, counts = torch.unique(pixels, dim=0, return_inverse=True, return_counts=True)
    if len(colors) != 4:
        raise ValueError("expected four synthetic colors: background, body, arm, mark")
    background = (colors == pixels[0]).all(dim=1).nonzero().item()
    objects = [index for index in range(4) if index != background]
    mark = min(objects, key=lambda index: int(counts[index]))
    if sum(int(counts[index]) == int(counts[mark]) for index in objects) != 1:
        raise ValueError("cannot distinguish the mark from another small region")
    body = max(objects, key=lambda index: int(counts[index]))
    return colors[mark].float() / 255, (inverse == mark).reshape(32, 32), colors[body].float() / 255


def mark_attributes(images: Tensor) -> Tensor:
    """Return [RGB, x/31, y/31] for the rare colored mark in each glyph."""
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("synthetic glyph images must be [B, 3, 32, 32]")
    labels = []
    for image in images:
        color, mask, _ = _mark_parts(image)
        coordinates = torch.nonzero(mask).float()
        y, x = coordinates.mean(0)
        labels.append(torch.cat((color, x[None] / 31, y[None] / 31)))
    return torch.stack(labels)


def erase_mark(images: Tensor) -> Tensor:
    """Replace only the mark with its underlying body color for a control."""
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("synthetic glyph images must be [B, 3, 32, 32]")
    erased = images.clone()
    for index, image in enumerate(images):
        _, mask, body = _mark_parts(image)
        erased[index, :, mask.to(images.device)] = body.to(images.device, images.dtype)[:, None]
    return erased


def collect_byte_memories(model: ResponseGraph | VectorMemoryModel, dataset: GlyphViews,
                          batch_size: int, device: str | torch.device,
                          erase: bool = False) -> tuple[Tensor, Tensor]:
    """Freeze the writer; expose only its 896 stored bytes and independent labels."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(model, (ResponseGraph, VectorMemoryModel)):
        raise TypeError("model must be a ResponseGraph or VectorMemoryModel")
    was_training = model.training
    features, labels = [], []
    model.eval()
    try:
        with torch.inference_mode():
            for source, _, _ in DataLoader(dataset, batch_size=batch_size):
                labels.append(mark_attributes(source))
                if erase:
                    source = erase_mark(source)
                memory = model.write(source.to(device))
                if isinstance(model, ResponseGraph):
                    compact = quantize_memory(memory)
                    stored = torch.cat((compact.horizontal.flatten(1),
                                        compact.vertical.flatten(1)), dim=1)
                else:
                    stored = quantize_vector(memory)
                features.append(stored.cpu())
    finally:
        model.train(was_training)
    return torch.cat(features), torch.cat(labels)


def _score(predicted: Tensor, actual: Tensor) -> dict[str, float]:
    return {
        "color_mse": (predicted[:, :3] - actual[:, :3]).square().mean().item(),
        "position_mae_px": ((predicted[:, 3:] - actual[:, 3:]) * 31).norm(dim=1).mean().item(),
    }


def memory_change_stats(intact: Tensor, erased: Tensor) -> dict[str, float]:
    """Quantify the mark's causal effect on the stored bytes themselves."""
    if (intact.shape != erased.shape or intact.ndim != 2 or intact.shape[1] != 896
            or intact.dtype != torch.uint8 or erased.dtype != torch.uint8):
        raise ValueError("expected paired uint8 memories [N, 896]")
    delta = (intact.int() - erased.int()).abs()
    per_scene = (delta > 0).sum(dim=1).float()
    return {"mean_changed_bytes": per_scene.mean().item(),
            "fraction_scenes_changed": (per_scene > 0).float().mean().item(),
            "mean_absolute_byte_delta": delta.float().mean().item()}


def fit_and_score(train_features: Tensor, train_labels: Tensor,
                  val_features: Tensor, val_labels: Tensor, *, kind: str,
                  steps: int, batch_size: int, seed: int,
                  device: str | torch.device) -> dict:
    """Fit one supervised probe and score intact, shuffled, and prior controls."""
    if (train_features.ndim != 2 or val_features.ndim != 2
            or train_features.shape[1] != 896 or val_features.shape[1] != 896
            or train_features.dtype != torch.uint8 or val_features.dtype != torch.uint8
            or train_labels.shape != (len(train_features), 5)
            or val_labels.shape != (len(val_features), 5)
            or len(train_features) < 2 or len(val_features) < 2):
        raise ValueError("expected paired byte memories [N, 896] and mark labels [N, 5]")
    if kind not in ("linear", "mlp") or steps <= 0 or batch_size <= 0:
        raise ValueError("kind must be linear or mlp; steps and batch_size must be positive")
    device = torch.device(device)
    x_train = train_features.float().to(device) / 255
    x_val = val_features.float().to(device) / 255
    y_train = train_labels.float().to(device)
    y_val = val_labels.float().to(device)
    x_mean = x_train.mean(0)
    x_std = x_train.std(0, unbiased=False).clamp_min(1 / 255)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0, unbiased=False).clamp_min(0.03)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_normalized = (y_train - y_mean) / y_std
    torch.manual_seed(seed)
    if kind == "linear":
        probe = nn.Linear(896, 5).to(device)
    else:
        probe = nn.Sequential(nn.Linear(896, 64), nn.GELU(), nn.Linear(64, 5)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    sample_generator = torch.Generator().manual_seed(seed + 10_000)
    for _ in range(steps):
        chosen = torch.randint(len(x_train), (batch_size,), generator=sample_generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = (probe(x_train[chosen]) - y_normalized[chosen]).square().mean()
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.inference_mode():
        training_prediction = probe(x_train) * y_std + y_mean
        actual_prediction = probe(x_val) * y_std + y_mean
        permutation = torch.randperm(len(x_val), generator=torch.Generator().manual_seed(seed + 20_000))
        shuffled_prediction = probe(x_val[permutation.to(device)]) * y_std + y_mean
        prior = y_mean.expand_as(y_val)
        return {
            "train": _score(training_prediction, y_train),
            "actual": _score(actual_prediction, y_val),
            "shuffled": _score(shuffled_prediction, y_val),
            "prior": _score(prior, y_val),
            "probe_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        }


def run_mark_probe(graph_checkpoint: Path, vector_checkpoint: Path, output: Path, *,
                   seed: int = 7, train_count: int = 2048, val_count: int = 128,
                   steps: int = 600, batch_size: int = 128, probe_seeds: int = 3,
                   device: str = "auto", probe_train_seed: int | None = None) -> dict:
    """Compare byte-matched frozen writers on a new held-out glyph split."""
    if min(train_count, val_count, steps, batch_size, probe_seeds) <= 0 or val_count < 2:
        raise ValueError("counts, steps, and batch_size must be positive; val_count >= 2")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("device must be cpu, available cuda, or auto")
    val_seed = seed + 2_000_003
    train_seed = seed if probe_train_seed is None else probe_train_seed
    if train_seed == val_seed:
        raise ValueError("probe training and validation seeds must differ")
    train_set, val_set = GlyphViews(train_seed, train_count), GlyphViews(val_seed, val_count)
    result = {"writer_seed": seed, "train_seed": train_seed, "probe_val_seed": val_seed,
              "train_count": train_count, "val_count": val_count,
              "steps": steps, "batch_size": batch_size, "probe_seeds": probe_seeds,
              "device": device, "memory_bytes": 896}
    training_labels = validation_labels = None
    for name, cls, checkpoint in (("graph", ResponseGraph, graph_checkpoint),
                                  ("vector", VectorMemoryModel, vector_checkpoint)):
        saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        if saved.get("architecture") != cls.__name__ or saved.get("seed") != seed:
            raise ValueError(f"{name} checkpoint architecture or training seed does not match")
        model = cls().to(device).eval()
        model.load_state_dict(saved["state_dict"])
        train_memory, train_labels = collect_byte_memories(model, train_set, batch_size, device)
        val_memory, val_labels = collect_byte_memories(model, val_set, batch_size, device)
        erased_train, erased_train_labels = collect_byte_memories(model, train_set, batch_size, device, erase=True)
        erased_val, erased_val_labels = collect_byte_memories(model, val_set, batch_size, device, erase=True)
        torch.testing.assert_close(erased_train_labels, train_labels, rtol=0, atol=0)
        torch.testing.assert_close(erased_val_labels, val_labels, rtol=0, atol=0)
        if training_labels is None:
            training_labels, validation_labels = train_labels, val_labels
        else:
            torch.testing.assert_close(train_labels, training_labels, rtol=0, atol=0)
            torch.testing.assert_close(val_labels, validation_labels, rtol=0, atol=0)
        result[name] = {"mark_intervention": memory_change_stats(val_memory, erased_val)}
        for kind in ("linear", "mlp"):
            runs = [fit_and_score(train_memory, train_labels, val_memory, val_labels,
                                  kind=kind, steps=steps, batch_size=batch_size,
                                  seed=seed + 100 * run, device=device)
                    for run in range(probe_seeds)]
            erased_runs = [fit_and_score(erased_train, train_labels, erased_val, val_labels,
                                         kind=kind, steps=steps, batch_size=batch_size,
                                         seed=seed + 100 * run, device=device)
                           for run in range(probe_seeds)]
            result[name][kind] = {"runs": runs, "erased_runs": erased_runs}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
