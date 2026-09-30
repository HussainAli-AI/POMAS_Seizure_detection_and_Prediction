"""Focused tests for the patient-independent dual-graph GNN."""

import unittest

import torch

from config import SELECTED_CHANNELS
from features.gnn_patient_independent import (
    DualGraphSeizurePredictor,
    TorchBandPower,
    functional_adjacency,
    physical_adjacency,
)


class DualGraphGNNTests(unittest.TestCase):
    def test_physical_graph_is_symmetric_and_normalized(self):
        graph = physical_adjacency(SELECTED_CHANNELS)
        self.assertEqual(graph.shape, (19, 19))
        self.assertTrue(torch.allclose(graph, graph.T, atol=1e-6))
        self.assertTrue(torch.all(graph.diagonal() > 0))

    def test_reversed_bipolar_derivations_are_connected(self):
        graph = physical_adjacency(["T7-P7", "P7-T7", "FP1-F7"])
        self.assertGreater(float(graph[0, 1]), 0.0)

    def test_functional_graph_is_symmetric_and_finite(self):
        windows = torch.randn(2, 19, 512)
        graph = functional_adjacency(windows, top_k=4)
        self.assertEqual(graph.shape, (2, 19, 19))
        self.assertTrue(torch.allclose(graph, graph.transpose(1, 2), atol=1e-6))
        self.assertTrue(torch.isfinite(graph).all())

    def test_ten_hertz_signal_has_dominant_alpha_power(self):
        extractor = TorchBandPower()
        time = torch.arange(2560, dtype=torch.float32) / 256.0
        signal = torch.sin(2 * torch.pi * 10.0 * time)
        windows = signal.reshape(1, 1, -1).repeat(1, 19, 1)
        features = extractor(windows)
        self.assertEqual(features.shape, (1, 19, 5))
        self.assertEqual(int(features[0, 0].argmax()), 2)

    def test_model_forward_shape(self):
        model = DualGraphSeizurePredictor()
        logits = model(torch.randn(2, 19, 2560))
        self.assertEqual(logits.shape, (2,))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()

