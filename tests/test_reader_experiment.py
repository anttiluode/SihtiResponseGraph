"""The image reader sees stored bytes and must make the small mark visible."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image

from response_graph.baseline import VectorMemoryModel
from response_graph.data import GlyphViews
from response_graph.model import ResponseGraph
from response_graph.reader_experiment import (ReaderBank, composite_mark,
                                              fit_mark_head, make_bank, reader_loss,
                                              render_mark_aware_query, run_reader_experiment,
                                              score_render, train_reader)


class ReaderExperimentTests(unittest.TestCase):
    def test_bank_contains_quantized_memory_and_rotated_mark_mask(self):
        model = ResponseGraph().eval()
        data = GlyphViews(71, 8)
        intact = make_bank(model, data, batch_size=4, device="cpu")
        erased = make_bank(model, data, batch_size=4, device="cpu", erase=True)
        self.assertEqual(intact.memory.shape, (8, 896))
        self.assertEqual(intact.memory.dtype, torch.uint8)
        self.assertEqual(intact.target.shape, (8, 3, 32, 32))
        self.assertEqual(intact.mark_mask.shape, (8, 1, 32, 32))
        self.assertEqual(intact.mark_label.shape, (8, 5))
        self.assertTrue(torch.equal(intact.target, erased.target))
        self.assertTrue(torch.equal(intact.mark_mask, erased.mark_mask))
        self.assertTrue((intact.memory != erased.memory).any().item())
        self.assertTrue((intact.mark_mask.sum((1, 2, 3)) > 0).all().item())
        self.assertTrue((intact.body_ring_mask.sum((1, 2, 3)) > 0).all().item())
        self.assertFalse((intact.mark_mask & intact.body_ring_mask).any().item())
        for index in range(8):
            target = data[index][2]
            self.assertTrue(torch.equal(intact.target[index], target))
            mark_color = target[:, intact.mark_mask[index, 0]].T.unique(dim=0)
            self.assertEqual(len(mark_color), 1)

    def test_composite_mark_rotates_location_and_preserves_distant_pixels(self):
        base = torch.zeros(4, 3, 32, 32)
        mark = torch.tensor([[0.8, 0.4, 0.2, 10 / 31, 8 / 31]]).expand(4, -1)
        image = composite_mark(base, mark, torch.tensor([0, 1, 2, 3]))
        self.assertGreater(image[0, 0, 8, 10].item(), 0.75)
        self.assertGreater(image[1, 0, 21, 8].item(), 0.75)
        self.assertGreater(image[2, 0, 23, 21].item(), 0.75)
        self.assertGreater(image[3, 0, 10, 23].item(), 0.75)
        self.assertLess(image[:, :, 25, 25].abs().sum().item(), 1e-10)

    def test_mark_head_learns_byte_signal_and_erasure_removes_it(self):
        generator = torch.Generator().manual_seed(32)
        memory = torch.zeros(256, 896, dtype=torch.uint8)
        memory[:, :5] = torch.randint(0, 256, (256, 5), generator=generator,
                                      dtype=torch.uint8)
        labels = memory[:, :5].float() / 255
        empty = torch.zeros(256, 3, 32, 32)
        mask = torch.zeros(256, 1, 32, 32, dtype=torch.bool)
        mask[:, :, 10, 10] = True
        ring = torch.zeros_like(mask)
        ring[:, :, 10, 11] = True
        bank = ReaderBank(memory, torch.zeros(256, dtype=torch.long), empty,
                          mask, ring, labels)
        head, stats = fit_mark_head(bank, steps=400, batch_size=64, seed=9, device="cpu")
        self.assertGreater(stats["trainable_parameters"], 0)
        with torch.inference_mode():
            recovered = head(memory)
            erased = head(torch.zeros_like(memory))
        self.assertLess((recovered - labels).square().mean().item(), 0.005)
        self.assertGreater((erased - labels).square().mean().item(), 0.03)

    def test_mark_loss_and_contrast_score_react_to_missing_mark(self):
        target = torch.zeros(2, 3, 32, 32)
        target[:, :, 10, 10] = torch.tensor([0.8, 0.4, 0.2])
        mark = torch.zeros(2, 1, 32, 32, dtype=torch.bool)
        mark[:, :, 10, 10] = True
        ring = torch.zeros_like(mark)
        ring[:, :, 10, 11] = True
        blank = torch.zeros_like(target)
        self.assertEqual(reader_loss(target, target, mark, mark_weight=0.1).item(), 0)
        self.assertGreater(reader_loss(blank, target, mark, mark_weight=0.1).item(),
                           reader_loss(blank, target, mark, mark_weight=0).item())
        self.assertAlmostEqual(score_render(target, target, mark, ring)["mark_contrast_alignment"], 1)
        self.assertAlmostEqual(score_render(blank, target, mark, ring)["mark_contrast_alignment"], 0)
        self.assertEqual(score_render(target, target, mark, ring)["mark_rgb_mse"], 0)
        self.assertGreater(score_render(blank, target, mark, ring)["mark_rgb_mse"], 0)

    def test_training_changes_reader_but_never_writer(self):
        for model in (ResponseGraph(), VectorMemoryModel()):
            data = make_bank(model, GlyphViews(91, 16), batch_size=8, device="cpu")
            writer_before = {name: value.clone() for name, value in model.encoder.state_dict().items()}
            edges_before = ({name: value.clone() for name, value in model.edge_writer.state_dict().items()}
                            if isinstance(model, ResponseGraph) else {})
            decoder_before = {name: value.clone() for name, value in model.decoder.state_dict().items()}
            stats = train_reader(model, data, steps=2, batch_size=8, mark_weight=0.1,
                                 seed=2, device="cpu")
            self.assertGreater(stats["trainable_parameters"], 0)
            self.assertTrue(any(not torch.equal(value, model.decoder.state_dict()[name])
                                for name, value in decoder_before.items()))
            for name, value in writer_before.items():
                torch.testing.assert_close(value, model.encoder.state_dict()[name], rtol=0, atol=0)
            for name, value in edges_before.items():
                torch.testing.assert_close(value, model.edge_writer.state_dict()[name], rtol=0, atol=0)

    def test_smoke_run_writes_results_checkpoints_and_image(self):
        results = Path(__file__).resolve().parents[1] / "results"
        with TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = run_reader_experiment(
                results / "pilot_graph.pt", results / "pilot_vector.pt", output,
                writer_seed=7, train_seed=3_000_010, dev_seed=4_000_010,
                test_seed=5_000_010, train_count=16, dev_count=8, test_count=8,
                steps=2, batch_size=8, mark_weight=0.1, device="cpu")
            self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)
            self.assertEqual(metrics["memory_bytes"], 896)
            self.assertEqual(metrics["seeds"]["test"], 5_000_010)
            for name in ("graph", "vector"):
                self.assertEqual(set(metrics[name]),
                                 {"reader_training", "mark_head_training",
                                  "erased_reader_training", "erased_head_training",
                                  "dev", "test"})
                self.assertEqual(set(metrics[name]["test"]),
                                 {"original", "trained", "erased_memory",
                                  "composed", "erased_composed", "head", "erased_head",
                                  "erased_trained_composed", "erased_trained_head", "prior"})
                self.assertGreaterEqual(metrics[name]["test"]["head"]["color_mse"], 0)
                self.assertGreaterEqual(metrics[name]["test"]["head"]["position_error_px"], 0)
                saved = torch.load(output / f"{name}_reader.pt", weights_only=True)
                self.assertEqual(saved["reader_training"]["mark_weight"], 0.1)
                mark_head = torch.load(output / f"{name}_mark_head.pt", weights_only=True)
                self.assertEqual(mark_head["architecture"], "MarkHead")
                self.assertEqual(saved["writer_fingerprint"], mark_head["writer_fingerprint"])
                self.assertEqual(saved["writer_fingerprint"],
                                 torch.load(output / f"{name}_erased_mark_head.pt",
                                            weights_only=True)["writer_fingerprint"])
            with Image.open(output / "predictions.png") as panel:
                self.assertGreater(panel.width, 32)
                self.assertGreater(panel.height, 32)
            rendered = output / "query.png"
            render_mark_aware_query(output / "graph_reader.pt",
                                    output / "graph_mark_head.pt",
                                    results / "pilot_input.png", 2, rendered)
            with Image.open(rendered) as image:
                self.assertEqual(image.size, (32, 32))
            wrong = torch.load(output / "graph_mark_head.pt", weights_only=True)
            wrong["writer_fingerprint"] = "wrong-writer"
            torch.save(wrong, output / "wrong_head.pt")
            with self.assertRaises(ValueError):
                render_mark_aware_query(output / "graph_reader.pt",
                                        output / "wrong_head.pt",
                                        results / "pilot_input.png", 2, rendered)
            with self.assertRaises(ValueError):
                render_mark_aware_query(output / "graph_reader.pt",
                                        output / "graph_erased_mark_head.pt",
                                        results / "pilot_input.png", 2, rendered)


if __name__ == "__main__":
    unittest.main()
