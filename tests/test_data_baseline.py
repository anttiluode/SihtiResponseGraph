import unittest

import torch

from response_graph.baseline import VectorMemoryModel, quantize_vector, vector_bytes
from response_graph.data import GlyphViews


class GlyphViewsTests(unittest.TestCase):
    def test_same_seed_repeats_image_and_query(self):
        a, b = GlyphViews(31, 8), GlyphViews(31, 8)
        for index in range(len(a)):
            one, other = a[index], b[index]
            for x, y in zip(one, other):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_query_is_exact_quarter_turn(self):
        dataset = GlyphViews(8, 12)
        for source, query, target in dataset:
            torch.testing.assert_close(target, torch.rot90(source, int(query), (-2, -1)),
                                       rtol=0, atol=0)

    def test_seed_and_index_change_source_object(self):
        self.assertFalse(torch.equal(GlyphViews(3, 2)[0][0], GlyphViews(4, 2)[0][0]))
        self.assertFalse(torch.equal(GlyphViews(3, 2)[0][0], GlyphViews(3, 2)[1][0]))
        with self.assertRaises(ValueError):
            GlyphViews(3, 0)


class VectorBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = VectorMemoryModel()
        self.source = torch.rand(2, 3, 32, 32)

    def test_state_is_896_real_serialized_bytes(self):
        memory = self.model.write(self.source)
        self.assertEqual(tuple(memory.shape), (2, 896))
        compact = quantize_vector(memory)
        self.assertEqual(compact.dtype, torch.uint8)
        self.assertEqual(vector_bytes(compact), 2 * 896)
        prediction = self.model.read(compact, torch.tensor([0, 3]))
        self.assertEqual(tuple(prediction.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(prediction).all().item())

    def test_invalid_query_or_image_fails(self):
        memory = self.model.write(self.source)
        with self.assertRaises(ValueError):
            self.model.read(memory, torch.tensor([-1, 0]))
        with self.assertRaises(ValueError):
            self.model.write(torch.rand(2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
