import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader

from response_graph.data import GlyphViews
from response_graph.experiment import evaluate, run_experiment
from response_graph.model import ResponseGraph
from response_graph.render import render_query


class ExperimentTests(unittest.TestCase):
    def test_two_step_run_saves_models_metrics_and_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_experiment(steps=2, batch_size=4, seed=11, device="cpu",
                                    output_dir=root, train_count=16, val_count=8)
            self.assertEqual(result["train_seed"], 11)
            self.assertNotEqual(result["train_seed"], result["val_seed"])
            for name in ("graph", "vector", "fixed_graph"):
                self.assertGreaterEqual(result[name]["mse"], 0)
                self.assertEqual(set(result[name]["by_query"]), {"0", "1", "2", "3"})
            self.assertEqual(result["graph"]["memory_bytes"], 896)
            self.assertEqual(result["vector"]["memory_bytes"], 896)
            self.assertEqual(result["fixed_graph"]["memory_bytes"], 896)
            self.assertEqual(json.loads((root / "metrics.json").read_text()), result)
            for name in ("graph.pt", "vector.pt", "predictions.png"):
                self.assertGreater((root / name).stat().st_size, 0)
            with Image.open(root / "predictions.png") as grid:
                self.assertGreater(grid.width, 32)
                self.assertGreater(grid.height, 32)
            output = root / "query.png"
            render_query(root / "graph.pt", root / "predictions.png", 2, output)
            with Image.open(output) as prediction:
                self.assertEqual(prediction.size, (32, 32))

    def test_evaluation_changes_no_parameters_or_training_mode(self):
        model = ResponseGraph().train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        metrics = evaluate(model, DataLoader(GlyphViews(123, 8), batch_size=4), "cpu")
        self.assertTrue(model.training)
        self.assertEqual(set(metrics["by_query"]), {"0", "1", "2", "3"})
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_bad_run_and_missing_checkpoint_fail_clearly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                run_experiment(steps=0, batch_size=4, seed=1, device="cpu",
                               output_dir=root, train_count=16, val_count=8)
            with self.assertRaises(FileNotFoundError):
                render_query(root / "missing.pt", root / "image.png", 0, root / "out.png")

    def test_bundled_pilot_checkpoint_renders_sample(self):
        results = Path(__file__).resolve().parents[1] / "results"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "query.png"
            render_query(results / "pilot_graph.pt", results / "pilot_input.png", 2, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (32, 32))
                self.assertGreater(len(image.getcolors(32 * 32)), 1)


if __name__ == "__main__":
    unittest.main()
