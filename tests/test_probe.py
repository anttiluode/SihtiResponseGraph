import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import torch

from response_graph.baseline import VectorMemoryModel
from response_graph.data import GlyphViews
from response_graph.model import ResponseGraph, quantize_memory
from response_graph.probe import (collect_byte_memories, erase_mark, fit_and_score,
                                  mark_attributes, memory_change_stats, run_mark_probe)


class MarkProbeTests(unittest.TestCase):
    def test_mark_attributes_find_color_and_center_from_synthetic_pixels(self):
        image = torch.tensor([12, 16, 24], dtype=torch.float32)[None, :, None, None].expand(1, 3, 32, 32).clone() / 255
        image[:, :, 5:20, 6:22] = torch.tensor([90, 160, 220])[:, None, None] / 255
        image[:, :, 15:18, 22:27] = torch.tensor([180, 100, 70])[:, None, None] / 255
        for y, x in ((8, 10), (9, 9), (9, 10), (9, 11), (10, 10)):
            image[0, :, y, x] = torch.tensor([240, 220, 100]) / 255
        result = mark_attributes(image)
        expected = torch.tensor([[240 / 255, 220 / 255, 100 / 255, 10 / 31, 9 / 31]])
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-6)

    def test_mark_attributes_reject_images_without_a_distinct_mark(self):
        with self.assertRaises(ValueError):
            mark_attributes(torch.zeros(1, 3, 32, 32))

    def test_erasing_mark_replaces_only_that_patch_with_body_color(self):
        background = torch.tensor([12, 16, 24], dtype=torch.float32) / 255
        body = torch.tensor([90, 160, 220], dtype=torch.float32) / 255
        arm = torch.tensor([180, 100, 70], dtype=torch.float32) / 255
        mark = torch.tensor([240, 220, 100], dtype=torch.float32) / 255
        source = background[None, :, None, None].expand(1, 3, 32, 32).clone()
        source[:, :, 5:20, 6:22] = body[:, None, None]
        source[:, :, 15:18, 22:27] = arm[:, None, None]
        mark_pixels = ((8, 10), (9, 9), (9, 10), (9, 11), (10, 10))
        for y, x in mark_pixels:
            source[0, :, y, x] = mark
        expected = source.clone()
        for y, x in mark_pixels:
            expected[0, :, y, x] = body
        erased = erase_mark(source)
        torch.testing.assert_close(erased, expected, rtol=0, atol=0)

    def test_collected_graph_features_are_quantized_edges_only(self):
        torch.manual_seed(9)
        model = ResponseGraph().train()
        data = GlyphViews(17, 8)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        features, labels = collect_byte_memories(model, data, batch_size=4, device="cpu")
        self.assertEqual(features.shape, (8, 896))
        self.assertEqual(features.dtype, torch.uint8)
        self.assertEqual(labels.shape, (8, 5))
        self.assertTrue(model.training)
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        with torch.inference_mode():
            source = torch.stack([data[index][0] for index in range(4)])
            memory = quantize_memory(model.write(source))
            expected = torch.cat((memory.horizontal.flatten(1), memory.vertical.flatten(1)), dim=1)
        torch.testing.assert_close(features[:4], expected, rtol=0, atol=0)
        erased_features, erased_labels = collect_byte_memories(model, data, 4, "cpu", erase=True)
        torch.testing.assert_close(erased_labels, labels, rtol=0, atol=0)
        with torch.inference_mode():
            erased_memory = quantize_memory(model.write(erase_mark(source)))
            expected_erased = torch.cat((erased_memory.horizontal.flatten(1),
                                         erased_memory.vertical.flatten(1)), dim=1)
        torch.testing.assert_close(erased_features[:4], expected_erased, rtol=0, atol=0)

    def test_collected_vector_features_share_labels_and_byte_budget(self):
        data = GlyphViews(17, 8)
        graph_features, graph_labels = collect_byte_memories(ResponseGraph(), data, 4, "cpu")
        vector_features, vector_labels = collect_byte_memories(VectorMemoryModel(), data, 4, "cpu")
        self.assertEqual(vector_features.shape, graph_features.shape)
        self.assertEqual(vector_features.dtype, torch.uint8)
        torch.testing.assert_close(vector_labels, graph_labels, rtol=0, atol=0)

    def test_linear_probe_recovers_known_signal_and_shuffling_breaks_it(self):
        generator = torch.Generator().manual_seed(3)
        train = torch.zeros(256, 896, dtype=torch.uint8)
        val = torch.zeros(64, 896, dtype=torch.uint8)
        train[:, :5] = torch.randint(0, 256, (256, 5), generator=generator, dtype=torch.uint8)
        val[:, :5] = torch.randint(0, 256, (64, 5), generator=generator, dtype=torch.uint8)
        result = fit_and_score(train, train[:, :5].float() / 255,
                               val, val[:, :5].float() / 255,
                               kind="linear", steps=500, batch_size=64, seed=5, device="cpu")
        self.assertLess(result["actual"]["color_mse"], result["prior"]["color_mse"] * 0.2)
        self.assertLess(result["actual"]["position_mae_px"], result["prior"]["position_mae_px"] * 0.2)
        self.assertGreater(result["shuffled"]["color_mse"], result["actual"]["color_mse"] * 3)

    def test_mlp_probe_produces_finite_metrics_on_small_input(self):
        torch.manual_seed(4)
        train = torch.randint(0, 256, (16, 896), dtype=torch.uint8)
        val = torch.randint(0, 256, (8, 896), dtype=torch.uint8)
        label_train = torch.rand(16, 5)
        label_val = torch.rand(8, 5)
        result = fit_and_score(train, label_train, val, label_val,
                               kind="mlp", steps=2, batch_size=8, seed=7, device="cpu")
        self.assertGreaterEqual(result["train"]["color_mse"], 0)
        for control in ("actual", "shuffled", "prior"):
            self.assertGreaterEqual(result[control]["color_mse"], 0)
            self.assertGreaterEqual(result[control]["position_mae_px"], 0)

    def test_memory_change_counts_changed_stored_bytes(self):
        intact = torch.zeros(2, 896, dtype=torch.uint8)
        erased = intact.clone()
        erased[0, 0] = 3
        erased[0, 1] = 1
        result = memory_change_stats(intact, erased)
        self.assertEqual(result["mean_changed_bytes"], 1)
        self.assertEqual(result["fraction_scenes_changed"], 0.5)
        self.assertAlmostEqual(result["mean_absolute_byte_delta"], 4 / (2 * 896))

    def test_smoke_run_uses_distinct_probe_validation_and_saves_metrics(self):
        checkpoints = Path(__file__).resolve().parents[1] / "results"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            result = run_mark_probe(checkpoints / "pilot_graph.pt",
                                    checkpoints / "pilot_vector.pt",
                                    output, seed=7, train_count=16, val_count=8,
                                    steps=2, batch_size=8, probe_seeds=1,
                                    probe_train_seed=3_000_010, device="cpu")
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(result["writer_seed"], 7)
            self.assertEqual(result["train_seed"], 3_000_010)
            self.assertEqual(result["probe_val_seed"], 2_000_010)
            self.assertEqual(result["memory_bytes"], 896)
            self.assertEqual(set(result["graph"]), {"linear", "mlp", "mark_intervention"})
            self.assertEqual(set(result["vector"]), {"linear", "mlp", "mark_intervention"})
            self.assertEqual(len(result["graph"]["linear"]["erased_runs"]), 1)


if __name__ == "__main__":
    unittest.main()
