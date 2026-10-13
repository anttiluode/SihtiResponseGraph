# Sihti Response Graph v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a graph-written, question-read image model with an equal-byte vector comparison.

**Architecture:** A convolutional writer emits undirected edge gates on an 8×8 lattice. A query seeds fresh activity, a lazy graph operator propagates it, and a decoder renders a 32×32 image. Synthetic quarter-turn views and a vector baseline supply the first falsifiable experiment.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, Pillow, unittest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-13-response-graph-design.md`

## Global Constraints

- Only conductances belong to per-instance graph memory: 896 uint8 scalars.
- Graph read cannot receive the source image or carry activity between calls.
- Dataset and evaluation splits are deterministic and disjoint.
- Training and comparison use the same targets, queries, optimizer, and step budget.
- The first task is planar quarter-turn prediction; do not describe it as unobserved 3D geometry.

## Review Focus

- Two consecutive reads and reversed read order must leave the memory unchanged: test in Task 1.
- Empty visibility must retain a previous graph exactly: test in Task 1.
- Query outside `0..3` and wrong image shape must fail clearly: tests in Tasks 1 and 2.
- Quantization must count bytes actually serialized, not a nominal latent size: test in Task 2.
- Evaluation must leave training weights unchanged and use separate sample seeds: test in Task 3.

---

### Task 1: Graph writer and reader

**Files:** `response_graph/model.py`, `response_graph/__init__.py`, `tests/test_model.py`.

**Interfaces:** `GraphMemory(horizontal: Tensor, vertical: Tensor)`; `ResponseGraph.write(image, previous=None, visibility=None) -> GraphMemory`; `ResponseGraph.read(memory, query: Tensor) -> Tensor`; `ResponseGraph.forward(image, query) -> Tensor`; `state_bytes(memory) -> int`; `quantize_memory(memory) -> GraphMemory`.

- [ ] Write tests for shapes, source-independent read, zero-visibility update, swapped conductances, equal repeated reads, query validation, byte count, writer gradient.
- [ ] Run tests to see missing-model failure.
- [ ] Implement writer, symmetric lazy graph propagation, fresh ping, and RGB decoder.
- [ ] Run Task 1 tests and suite; check backward gradient and read isolation.

### Task 2: Synthetic task and vector baseline

**Files:** `response_graph/data.py`, `response_graph/baseline.py`, `tests/test_data_baseline.py`.

**Interfaces:** `GlyphViews(seed:int, count:int) -> Dataset` yielding `(source, query, target)`; `VectorMemoryModel.write(image) -> Tensor`; `VectorMemoryModel.read(memory, query) -> Tensor`; `VectorMemoryModel.forward(image, query) -> Tensor`; `quantize_vector(memory) -> Tensor`.

- [ ] Write deterministic sample, exact quarter-turn, disjoint seed, baseline shape, invalid query, and measured byte tests.
- [ ] Run tests to see missing-data/baseline failure.
- [ ] Implement asymmetric glyph generator and exactly 896-element vector memory.
- [ ] Run Task 2 tests and suite.

### Task 3: Training, metrics, images, CI

**Files:** `train.py`, `response_graph/experiment.py`, `tests/test_experiment.py`, `README.md`, `pyproject.toml`, `.gitignore`, `.github/workflows/tests.yml`.

**Interfaces:** `run_experiment(steps:int, batch_size:int, seed:int, device:str, output_dir:Path) -> dict`; `evaluate(model, loader, device) -> dict`.

- [ ] Write tests for a two-step run that saves metrics/checkpoints/image grid, seeded split, no evaluation weight changes, and fixed-graph control.
- [ ] Run tests to see missing-training failure.
- [ ] Implement shared-batch training, evaluation, quantized controls, CLI and reproducibility outputs.
- [ ] Run complete tests and a CPU smoke command; inspect metrics and PNG.
- [ ] Document what is measured, what remains untested, exact commands, and expected resource use. Run CI on the feature branch.
