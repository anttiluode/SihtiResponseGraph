import unittest

import torch

from response_graph.model import ResponseGraph, quantize_memory, state_bytes


class ResponseGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ResponseGraph()
        self.a = torch.rand(2, 3, 32, 32)

    def test_write_read_shape_and_finite(self):
        memory = self.model.write(self.a)
        self.assertEqual(tuple(memory.horizontal.shape), (2, 8, 8, 7))
        self.assertEqual(tuple(memory.vertical.shape), (2, 8, 7, 8))
        image = self.model.read(memory, torch.tensor([0, 3]))
        self.assertEqual(tuple(image.shape), (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(image).all().item())
        self.assertTrue(((image >= 0) & (image <= 1)).all().item())

    def test_read_restarts_activity_and_does_not_mutate_graph(self):
        memory = self.model.write(self.a)
        before = (memory.horizontal.clone(), memory.vertical.clone())
        first = self.model.read(memory, torch.tensor([1, 1]))
        self.model.read(memory, torch.tensor([3, 2]))
        second = self.model.read(memory, torch.tensor([1, 1]))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(before[0], memory.horizontal, rtol=0, atol=0)
        torch.testing.assert_close(before[1], memory.vertical, rtol=0, atol=0)

    def test_zero_visibility_preserves_previous_memory(self):
        memory = self.model.write(self.a)
        changed = self.model.write(torch.zeros_like(self.a), memory, torch.zeros(2, 1, 32, 32))
        torch.testing.assert_close(changed.horizontal, memory.horizontal, rtol=0, atol=0)
        torch.testing.assert_close(changed.vertical, memory.vertical, rtol=0, atol=0)

    def test_graph_controls_same_ping_response(self):
        memory = self.model.write(self.a)
        query = torch.tensor([0, 0])
        normal = self.model.read(memory, query)
        swapped = self.model.read(memory.index_select(torch.tensor([1, 0])), query)
        self.assertGreater((normal - swapped).abs().max().item(), 1e-7)

    def test_writer_receives_image_loss_gradient(self):
        loss = self.model(self.a, torch.tensor([0, 1])).square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.edge_writer.weight.grad)
        self.assertGreater(self.model.edge_writer.weight.grad.abs().sum().item(), 1e-9)

    def test_quantized_graph_occupies_896_bytes_per_image(self):
        memory = quantize_memory(self.model.write(self.a))
        self.assertEqual(memory.horizontal.dtype, torch.uint8)
        self.assertEqual(state_bytes(memory), 2 * 896)
        self.assertTrue(torch.isfinite(self.model.read(memory, torch.tensor([0, 1]))).all().item())

    def test_weak_conductances_survive_byte_round_trip(self):
        with torch.no_grad():
            self.model.edge_writer.weight.zero_()
            self.model.edge_writer.bias.fill_(-20)
        memory = self.model.write(self.a)
        compact = quantize_memory(memory)
        self.assertGreater(compact.horizontal.min().item(), 0)
        query = torch.tensor([0, 2])
        self.assertLess((self.model.read(memory, query) - self.model.read(compact, query))
                        .square().mean().item(), 1e-4)

    def test_bad_shapes_and_queries_fail(self):
        with self.assertRaises(ValueError):
            self.model.write(torch.rand(1, 3, 31, 32))
        memory = self.model.write(self.a)
        with self.assertRaises(ValueError):
            self.model.read(memory, torch.tensor([0, 4]))
        with self.assertRaises(ValueError):
            self.model.read(memory, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
