"""Focused tests for the global-v2 MFCC-Siamese branch."""

import unittest

import numpy as np
import torch

from features.mfcc_siamese import MFCCSiamesePredictor, TorchMFCC, patient_contrastive_loss
from train_mfcc_siamese_global import (
    alarm_event_metrics,
    select_validation_threshold,
    window_metrics,
)


class MFCCSiameseTests(unittest.TestCase):
    def test_mfcc_retains_time_dimension(self):
        extractor = TorchMFCC()
        windows = torch.randn(1, 19, 2560)
        features = extractor(windows)
        self.assertEqual(features.shape[:3], (1, 19, 13))
        self.assertGreater(features.shape[-1], 1)
        self.assertTrue(torch.isfinite(features).all())

    def test_predictor_output_shapes(self):
        model = MFCCSiamesePredictor(n_channels=19, n_train_subjects=2)
        seizure, patient, embedding = model(torch.randn(2, 19, 2560))
        self.assertEqual(seizure.shape, (2,))
        self.assertEqual(patient.shape, (2, 2))
        self.assertEqual(embedding.shape, (2, 100))

    def test_contrastive_relation_is_patient_based(self):
        embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        same_loss = patient_contrastive_loss(embedding, embedding, torch.ones(2))
        different_loss = patient_contrastive_loss(embedding, embedding, torch.zeros(2))
        self.assertAlmostEqual(float(same_loss), 0.0, places=5)
        self.assertGreater(float(different_loss), 0.0)

    def test_validation_threshold_separates_simple_classes(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        threshold = select_validation_threshold(labels, scores)
        metrics = window_metrics(labels, scores, threshold)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)

    def test_refractory_alarm_events_are_not_window_counts(self):
        labels = np.asarray([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
        scores = np.asarray([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9])
        timestamps = np.asarray([0, 5, 10, 15, 20, 25, 2000], dtype=np.float64)
        metrics = alarm_event_metrics(
            labels, scores, timestamps, threshold=0.5, refractory_seconds=1800
        )
        self.assertEqual(metrics["n_alarm_events"], 2)
        self.assertEqual(metrics["n_true_alarm_events"], 1)
        self.assertEqual(metrics["n_false_alarm_events"], 1)
        self.assertEqual(metrics["preictal_episode_sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main()

