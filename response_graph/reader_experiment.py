"""Fine-tune image readers on frozen 896-byte memories with mark-aware loss."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .baseline import VectorMemoryModel, quantize_vector
from .data import GlyphViews
from .model import GraphMemory, ResponseGraph, quantize_memory
from .probe import _mark_parts, erase_mark, mark_attributes


@dataclass(frozen=True)
class ReaderBank:
    """The reader input is uint8 memory plus a query; targets are supervision only."""

    memory: Tensor  # [N, 896] uint8, stored bytes in fixed order
    query: Tensor  # [N] quarter-turns
    target: Tensor  # [N, 3, 32, 32]
    mark_mask: Tensor  # [N, 1, 32, 32] bool
    body_ring_mask: Tensor  # [N, 1, 32, 32] bool
    mark_label: Tensor  # [N, 5] source-view RGB and center, never reader input


def writer_fingerprint(model: ResponseGraph | VectorMemoryModel) -> str:
    """Bind a mark head to the exact encoder weights that produced its bytes."""
    modules = (("encoder", model.encoder),)
    if isinstance(model, ResponseGraph):
        modules += (("edge_writer", model.edge_writer),)
    digest = hashlib.sha256()
    for prefix, module in modules:
        for name, value in module.state_dict().items():
            digest.update(f"{prefix}.{name}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_bank(model: ResponseGraph | VectorMemoryModel, dataset: GlyphViews,
              batch_size: int, device: str | torch.device,
              erase: bool = False) -> ReaderBank:
    """Write once, quantize, and keep only stored bytes for reader input."""
    if not isinstance(model, (ResponseGraph, VectorMemoryModel)) or batch_size <= 0:
        raise ValueError("expected an image memory model and positive batch size")
    was_training = model.training
    model.eval()
    memories, queries, targets, marks, rings, labels = [], [], [], [], [], []
    try:
        with torch.inference_mode():
            for source, query, target in DataLoader(dataset, batch_size=batch_size):
                labels.append(mark_attributes(source))
                batch_marks, batch_rings = [], []
                for image in target:
                    _, mark, body = _mark_parts(image)
                    neighborhood = F.max_pool2d(mark[None, None].float(), 7, stride=1, padding=3)[0, 0].bool()
                    body_pixels = (image == body[:, None, None]).all(dim=0)
                    ring = neighborhood & ~mark & body_pixels
                    if not bool(ring.any()):
                        raise ValueError("mark has no neighboring body pixels")
                    batch_marks.append(mark[None])
                    batch_rings.append(ring[None])
                if erase:
                    source = erase_mark(source)
                raw = model.write(source.to(device))
                if isinstance(model, ResponseGraph):
                    compact = quantize_memory(raw)
                    memory = torch.cat((compact.horizontal.flatten(1),
                                        compact.vertical.flatten(1)), dim=1)
                else:
                    memory = quantize_vector(raw)
                if memory.dtype != torch.uint8 or memory.shape[1] != 896:
                    raise RuntimeError("stored memory must be exactly 896 uint8 values")
                memories.append(memory.cpu())
                queries.append(query)
                targets.append(target)
                marks.append(torch.stack(batch_marks))
                rings.append(torch.stack(batch_rings))
    finally:
        model.train(was_training)
    return ReaderBank(torch.cat(memories), torch.cat(queries), torch.cat(targets),
                      torch.cat(marks), torch.cat(rings), torch.cat(labels))


def _read_bytes(model: nn.Module, memory: Tensor, query: Tensor) -> Tensor:
    if memory.dtype != torch.uint8 or memory.ndim != 2 or memory.shape[1] != 896:
        raise ValueError("reader input must be [N, 896] stored bytes")
    if isinstance(model, ResponseGraph):
        horizontal = memory[:, :448].reshape(-1, 8, 8, 7)
        vertical = memory[:, 448:].reshape(-1, 8, 7, 8)
        return model.read(GraphMemory(horizontal, vertical), query)
    if isinstance(model, VectorMemoryModel):
        return model.read(memory, query)
    raise TypeError("reader must be a graph or equal-byte vector model")


class MarkHead(nn.Module):
    """A small image-detail reader that predicts mark RGB and source position."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(896, 64), nn.GELU(), nn.Linear(64, 5))
        self.register_buffer("x_mean", torch.zeros(896))
        self.register_buffer("x_std", torch.ones(896))
        self.register_buffer("y_mean", torch.zeros(5))
        self.register_buffer("y_std", torch.ones(5))

    def forward(self, memory: Tensor) -> Tensor:
        if memory.dtype != torch.uint8 or memory.ndim != 2 or memory.shape[1] != 896:
            raise ValueError("mark head accepts only [N, 896] uint8 memory")
        normalized = (memory.float() / 255 - self.x_mean) / self.x_std
        return self.net(normalized) * self.y_std + self.y_mean


def fit_mark_head(bank: ReaderBank, *, steps: int, batch_size: int,
                  seed: int, device: str | torch.device) -> tuple[MarkHead, dict]:
    """Train on mark labels; normalization is derived from training bytes only."""
    if steps <= 0 or batch_size <= 0 or len(bank.memory) < batch_size:
        raise ValueError("invalid mark-head training settings")
    device = torch.device(device)
    memory = bank.memory.to(device)
    labels = bank.mark_label.to(device)
    torch.manual_seed(seed)
    head = MarkHead().to(device)
    with torch.no_grad():
        values = memory.float() / 255
        head.x_mean.copy_(values.mean(0))
        head.x_std.copy_(values.std(0, unbiased=False).clamp_min(1 / 255))
        head.y_mean.copy_(labels.mean(0))
        head.y_std.copy_(labels.std(0, unbiased=False).clamp_min(0.03))
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 10_000)
    for _ in range(steps):
        indices = torch.randint(len(memory), (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        normalized_error = (head(memory[indices]) - labels[indices]) / head.y_std
        loss = normalized_error.square().mean()
        loss.backward()
        optimizer.step()
    head.eval()
    return head, {"last_train_loss": loss.item(),
                  "trainable_parameters": sum(p.numel() for p in head.parameters())}


def composite_mark(base: Tensor, attributes: Tensor, query: Tensor) -> Tensor:
    """Render a predicted five-pixel cross into the queried image frame."""
    if (base.ndim != 4 or base.shape[1:] != (3, 32, 32)
            or attributes.shape != (len(base), 5)
            or query.shape != (len(base),)
            or bool(((query < 0) | (query > 3)).any())):
        raise ValueError("expected images [N,3,32,32], mark [N,5], and quarter-turn queries")
    x, y = attributes[:, 3].clamp(0, 1) * 31, attributes[:, 4].clamp(0, 1) * 31
    xq = torch.where(query == 1, y, torch.where(query == 2, 31 - x,
                     torch.where(query == 3, 31 - y, x)))
    yq = torch.where(query == 1, 31 - x, torch.where(query == 2, 31 - y,
                     torch.where(query == 3, x, y)))
    yy, xx = torch.meshgrid(torch.arange(32, device=base.device, dtype=base.dtype),
                            torch.arange(32, device=base.device, dtype=base.dtype), indexing="ij")
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    strokes = [torch.exp(-((xx[None] - xq[:, None, None] - dx).square()
                           + (yy[None] - yq[:, None, None] - dy).square())
                         / (2 * 0.45**2)) for dx, dy in offsets]
    alpha = torch.stack(strokes).amax(dim=0)[:, None]
    color = attributes[:, :3].clamp(0, 1)[:, :, None, None]
    return base * (1 - alpha) + color * alpha


def reader_loss(prediction: Tensor, target: Tensor, mark_mask: Tensor,
                mark_weight: float) -> Tensor:
    """Maintain image fidelity while paying for the otherwise tiny mark."""
    if (prediction.shape != target.shape or target.ndim != 4
            or mark_mask.shape != (len(target), 1, 32, 32)
            or mark_weight < 0):
        raise ValueError("incompatible image, mark mask, or negative mark weight")
    error = (prediction - target).square()
    selected = mark_mask.to(error.dtype)
    mark_mse = (error * selected).sum() / (selected.sum() * 3).clamp_min(1)
    return error.mean() + mark_weight * mark_mse


def score_render(prediction: Tensor, target: Tensor, mark_mask: Tensor,
                 body_ring_mask: Tensor) -> dict[str, float]:
    """Score actual RGB pixels, including mark contrast against adjacent body."""
    if (prediction.shape != target.shape or target.ndim != 4
            or mark_mask.shape != (len(target), 1, 32, 32)
            or body_ring_mask.shape != mark_mask.shape):
        raise ValueError("incompatible render and masks")
    mark, ring = mark_mask.float(), body_ring_mask.float()
    if not bool((mark.sum((1, 2, 3)) > 0).all() and (ring.sum((1, 2, 3)) > 0).all()):
        raise ValueError("each image needs mark and neighboring body pixels")
    error = (prediction - target).square()
    global_mse = error.mean().item()
    mark_mse = (error * mark).sum().item() / (3 * mark.sum().item())
    ring_mse = (error * ring).sum().item() / (3 * ring.sum().item())

    def mean_color(image: Tensor, mask: Tensor) -> Tensor:
        return (image * mask).sum((2, 3)) / mask.sum((2, 3))

    target_delta = mean_color(target, mark) - mean_color(target, ring)
    predicted_delta = mean_color(prediction, mark) - mean_color(prediction, ring)
    alignment = (predicted_delta * target_delta).sum() / target_delta.square().sum().clamp_min(1e-12)
    return {"image_mse": global_mse,
            "psnr_db": -10 * math.log10(max(global_mse, 1e-12)),
            "mark_rgb_mse": mark_mse,
            "neighbor_body_mse": ring_mse,
            "mark_contrast_alignment": alignment.item()}


def train_reader(model: ResponseGraph | VectorMemoryModel, bank: ReaderBank, *,
                 steps: int, batch_size: int, mark_weight: float,
                 seed: int, device: str | torch.device) -> dict:
    """Update query/readout weights, never the pretrained writer."""
    if steps <= 0 or batch_size <= 0 or mark_weight < 0 or len(bank.memory) < batch_size:
        raise ValueError("invalid reader training settings")
    device = torch.device(device)
    if isinstance(model, ResponseGraph):
        writer = (model.encoder, model.edge_writer)
        reader = (model.ping, model.decoder)
    elif isinstance(model, VectorMemoryModel):
        writer = (model.encoder,)
        reader = (model.decoder,)
    else:
        raise TypeError("expected graph or vector model")
    model.to(device).train()
    for module in writer:
        module.requires_grad_(False)
    for module in reader:
        module.requires_grad_(True)
    parameters = [parameter for module in reader for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    memory = bank.memory.to(device)
    query = bank.query.to(device)
    target = bank.target.to(device)
    mark = bank.mark_mask.to(device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(len(memory), (batch_size,), generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = _read_bytes(model, memory[indices], query[indices])
        loss = reader_loss(output, target[indices], mark[indices], mark_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    return {"last_train_loss": loss.item(),
            "trainable_parameters": sum(p.numel() for p in parameters)}


def _evaluate_reader(model: nn.Module, bank: ReaderBank,
                     batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(bank.memory), batch_size):
            indices = slice(offset, offset + batch_size)
            predictions.append(_read_bytes(model, bank.memory[indices].to(device),
                                           bank.query[indices].to(device)).cpu())
    return score_render(torch.cat(predictions), bank.target, bank.mark_mask,
                        bank.body_ring_mask)


def _evaluate_composed(model: nn.Module, head: MarkHead, bank: ReaderBank,
                       batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    head.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(bank.memory), batch_size):
            indices = slice(offset, offset + batch_size)
            memory = bank.memory[indices].to(device)
            query = bank.query[indices].to(device)
            base = _read_bytes(model, memory, query)
            predictions.append(composite_mark(base, head(memory), query).cpu())
    return score_render(torch.cat(predictions), bank.target, bank.mark_mask,
                        bank.body_ring_mask)


def _evaluate_head(head: MarkHead, bank: ReaderBank,
                   batch_size: int, device: torch.device) -> dict[str, float]:
    head.eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(bank.memory), batch_size):
            outputs.append(head(bank.memory[offset:offset + batch_size].to(device)).cpu())
    predicted = torch.cat(outputs)
    actual = bank.mark_label
    return {"color_mse": (predicted[:, :3] - actual[:, :3]).square().mean().item(),
            "position_error_px": ((predicted[:, 3:] - actual[:, 3:]) * 31).norm(dim=1).mean().item()}


def _label_prior(train: ReaderBank, test: ReaderBank) -> dict[str, float]:
    predicted = train.mark_label.mean(0)[None].expand_as(test.mark_label)
    return {"color_mse": (predicted[:, :3] - test.mark_label[:, :3]).square().mean().item(),
            "position_error_px": ((predicted[:, 3:] - test.mark_label[:, 3:]) * 31).norm(dim=1).mean().item()}


def _pil(image: Tensor) -> Image.Image:
    pixels = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def _save_panel(base: dict, tuned: dict, heads: dict, erased_tuned: dict,
                erased_heads: dict, banks: dict, data: GlyphViews,
                device: torch.device, path: Path, zoom: bool = False) -> None:
    labels = ("source", "target", "graph old", "graph tuned", "graph mark",
              "graph erased", "graph erased fit", "vector old", "vector tuned",
              "vector mark", "vector erased", "vector erased fit")
    tile, gap, rows = 96, 10, min(4, len(data))
    panel = Image.new("RGB", (len(labels) * (tile + gap) + gap,
                              rows * (tile + 30) + gap), (25, 29, 37))
    draw = ImageDraw.Draw(panel)
    for row in range(rows):
        source, query, target = data[row]
        predictions = []
        for name in ("graph", "vector"):
            bank, erased = banks[name]
            q = query[None].to(device)
            with torch.inference_mode():
                memory = bank.memory[row:row + 1].to(device)
                erased_memory = erased.memory[row:row + 1].to(device)
                tuned_image = _read_bytes(tuned[name], memory, q)
                erased_image = _read_bytes(tuned[name], erased_memory, q)
                predictions.extend((
                    _read_bytes(base[name], memory, q)[0].cpu(),
                    tuned_image[0].cpu(),
                    composite_mark(tuned_image, heads[name](memory), q)[0].cpu(),
                    composite_mark(erased_image, heads[name](erased_memory), q)[0].cpu(),
                    composite_mark(_read_bytes(erased_tuned[name], erased_memory, q),
                                   erased_heads[name](erased_memory), q)[0].cpu()))
        images = (source, target, *predictions)
        for col, (label, image) in enumerate(zip(labels, images)):
            picture = _pil(image)
            if zoom:
                if col == 0:
                    _, source_mark, _ = _mark_parts(source)
                    coordinates = torch.nonzero(source_mark).float().mean(0)
                else:
                    coordinates = torch.nonzero(banks["graph"][0].mark_mask[row, 0]).float().mean(0)
                y, x = [int(round(value.item())) for value in coordinates]
                picture = picture.crop((x - 4, y - 4, x + 5, y + 5))
            x0, y0 = gap + col * (tile + gap), gap + row * (tile + 30)
            panel.paste(picture.resize((tile, tile), Image.Resampling.NEAREST), (x0, y0 + 18))
            draw.text((x0, y0), f"{label} q={int(query)}", fill=(238, 238, 242))
    panel.save(path)


def run_reader_experiment(graph_checkpoint: Path, vector_checkpoint: Path,
                          output_dir: Path, *, writer_seed: int = 7,
                          train_seed: int = 5_000_010, dev_seed: int = 5_000_011,
                          test_seed: int = 6_000_010, train_count: int = 8192,
                          dev_count: int = 128, test_count: int = 256,
                          steps: int = 1200, batch_size: int = 128,
                          mark_weight: float = 0.1, device: str = "auto") -> dict:
    """Train identical objectives on frozen graph/vector writers, then test once."""
    if (len({writer_seed, train_seed, dev_seed, test_seed}) != 4
            or min(train_count, dev_count, test_count, steps, batch_size) <= 0
            or train_count < batch_size or mark_weight < 0):
        raise ValueError("seeds must differ; counts and steps positive; train_count >= batch_size")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("device must be cpu, available cuda, or auto")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    where = torch.device(device)
    data = {"train": GlyphViews(train_seed, train_count),
            "dev": GlyphViews(dev_seed, dev_count),
            "test": GlyphViews(test_seed, test_count)}
    result = {"writer_seed": writer_seed,
              "seeds": {"train": train_seed, "dev": dev_seed, "test": test_seed},
              "counts": {"train": train_count, "dev": dev_count, "test": test_count},
              "steps": steps, "batch_size": batch_size,
              "mark_weight": mark_weight, "device": device, "memory_bytes": 896}
    original, trained, heads, erased_models, erased_heads, panels = {}, {}, {}, {}, {}, {}
    for name, cls, checkpoint in (("graph", ResponseGraph, graph_checkpoint),
                                  ("vector", VectorMemoryModel, vector_checkpoint)):
        saved = torch.load(Path(checkpoint), map_location=where, weights_only=True)
        if saved.get("architecture") != cls.__name__ or saved.get("seed") != writer_seed:
            raise ValueError(f"{name} checkpoint architecture or writer seed mismatch")
        model = cls().to(where).eval()
        model.load_state_dict(saved["state_dict"])
        fingerprint = writer_fingerprint(model)
        reader = deepcopy(model)
        erased_reader = deepcopy(model)
        train = make_bank(model, data["train"], batch_size, where)
        erased_train = make_bank(model, data["train"], batch_size, where, erase=True)
        dev = make_bank(model, data["dev"], batch_size, where)
        test = make_bank(model, data["test"], batch_size, where)
        erased = make_bank(model, data["test"], batch_size, where, erase=True)
        if (not torch.equal(test.target, erased.target)
                or not torch.equal(train.mark_label, erased_train.mark_label)):
            raise RuntimeError("erasure must not change targets or mark labels")
        training = train_reader(reader, train, steps=steps, batch_size=batch_size,
                                mark_weight=mark_weight, seed=writer_seed, device=where)
        head, head_training = fit_mark_head(train, steps=steps, batch_size=batch_size,
                                           seed=writer_seed, device=where)
        erased_training = train_reader(erased_reader, erased_train, steps=steps,
                                       batch_size=batch_size, mark_weight=mark_weight,
                                       seed=writer_seed, device=where)
        erased_head, erased_head_training = fit_mark_head(erased_train, steps=steps,
                                                         batch_size=batch_size,
                                                         seed=writer_seed, device=where)
        if (writer_fingerprint(reader) != fingerprint
                or writer_fingerprint(erased_reader) != fingerprint):
            raise RuntimeError("reader training changed frozen writer weights")
        result[name] = {"reader_training": training,
                        "mark_head_training": head_training,
                        "erased_reader_training": erased_training,
                        "erased_head_training": erased_head_training,
                        "dev": {"trained": _evaluate_reader(reader, dev, batch_size, where),
                                "composed": _evaluate_composed(reader, head, dev, batch_size, where),
                                "head": _evaluate_head(head, dev, batch_size, where)},
                        "test": {"original": _evaluate_reader(model, test, batch_size, where),
                                 "trained": _evaluate_reader(reader, test, batch_size, where),
                                 "erased_memory": _evaluate_reader(reader, erased, batch_size, where),
                                 "composed": _evaluate_composed(reader, head, test, batch_size, where),
                                 "erased_composed": _evaluate_composed(reader, head, erased, batch_size, where),
                                 "head": _evaluate_head(head, test, batch_size, where),
                                 "erased_head": _evaluate_head(head, erased, batch_size, where),
                                 "erased_trained_composed": _evaluate_composed(erased_reader, erased_head,
                                                                                erased, batch_size, where),
                                 "erased_trained_head": _evaluate_head(erased_head, erased,
                                                                       batch_size, where),
                                 "prior": _label_prior(train, test)}}
        pair_id, erased_pair_id = uuid4().hex, uuid4().hex
        torch.save({"architecture": cls.__name__, "state_dict": reader.state_dict(),
                    "seed": writer_seed, "writer_fingerprint": fingerprint, "pair_id": pair_id,
                    "reader_training": {"steps": steps, "train_seed": train_seed,
                                        "mark_weight": mark_weight, "memory_bytes": 896}},
                   destination / f"{name}_reader.pt")
        torch.save({"architecture": "MarkHead", "base_architecture": cls.__name__,
                    "writer_seed": writer_seed, "writer_fingerprint": fingerprint,
                    "pair_id": pair_id, "state_dict": head.state_dict(),
                    "reader_training": {"steps": steps, "train_seed": train_seed,
                                        "memory_bytes": 896}},
                   destination / f"{name}_mark_head.pt")
        torch.save({"architecture": cls.__name__, "state_dict": erased_reader.state_dict(),
                    "seed": writer_seed, "writer_fingerprint": fingerprint,
                    "pair_id": erased_pair_id,
                    "reader_training": {"steps": steps, "train_seed": train_seed,
                                        "mark_weight": mark_weight, "memory_bytes": 896,
                                        "mark_erased": True}},
                   destination / f"{name}_erased_reader.pt")
        torch.save({"architecture": "MarkHead", "base_architecture": cls.__name__,
                    "writer_seed": writer_seed, "writer_fingerprint": fingerprint,
                    "pair_id": erased_pair_id, "state_dict": erased_head.state_dict(),
                    "reader_training": {"steps": steps, "train_seed": train_seed,
                                        "memory_bytes": 896, "mark_erased": True}},
                   destination / f"{name}_erased_mark_head.pt")
        original[name], trained[name], heads[name] = model, reader, head
        erased_models[name], erased_heads[name], panels[name] = erased_reader, erased_head, (test, erased)
    _save_panel(original, trained, heads, erased_models, erased_heads,
                panels, data["test"], where,
                destination / "predictions.png")
    _save_panel(original, trained, heads, erased_models, erased_heads,
                panels, data["test"], where,
                destination / "mark_crops.png", zoom=True)
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render_mark_aware_query(reader_checkpoint: Path, mark_checkpoint: Path,
                            input_image: Path, query: int, output_image: Path,
                            device: str = "cpu") -> None:
    """Write an observed image into 896 bytes and render a marked query view."""
    paths = [Path(reader_checkpoint), Path(mark_checkpoint), Path(input_image)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing reader input: {path}")
    if query not in range(4):
        raise ValueError("query must be 0, 1, 2, or 3")
    saved = torch.load(paths[0], map_location=device, weights_only=True)
    mark_saved = torch.load(paths[1], map_location=device, weights_only=True)
    architecture = saved.get("architecture")
    if (architecture not in ("ResponseGraph", "VectorMemoryModel")
            or mark_saved.get("architecture") != "MarkHead"
            or mark_saved.get("base_architecture") != architecture
            or mark_saved.get("writer_seed") != saved.get("seed")
            or not saved.get("writer_fingerprint")
            or mark_saved.get("writer_fingerprint") != saved.get("writer_fingerprint")
            or not saved.get("pair_id")
            or mark_saved.get("pair_id") != saved.get("pair_id")):
        raise ValueError("reader and mark-head checkpoints do not match")
    cls = ResponseGraph if architecture == "ResponseGraph" else VectorMemoryModel
    model = cls().to(device).eval()
    model.load_state_dict(saved["state_dict"])
    if writer_fingerprint(model) != saved["writer_fingerprint"]:
        raise ValueError("reader checkpoint's writer fingerprint is invalid")
    head = MarkHead().to(device).eval()
    head.load_state_dict(mark_saved["state_dict"])
    with Image.open(paths[2]) as source:
        pixels = np.asarray(source.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)).copy()
    image = torch.from_numpy(pixels).permute(2, 0, 1).float()[None].to(device) / 255
    with torch.inference_mode():
        raw = model.write(image)
        if isinstance(model, ResponseGraph):
            compact = quantize_memory(raw)
            memory = torch.cat((compact.horizontal.flatten(1), compact.vertical.flatten(1)), dim=1)
        else:
            memory = quantize_vector(raw)
        request = torch.tensor([query], device=device)
        base = _read_bytes(model, memory, request)
        prediction = composite_mark(base, head(memory), request)[0]
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    _pil(prediction).save(output_image)
