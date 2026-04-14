"""
Tests for retrieval_manifold.py — verifies architecture shapes, state
compatibility with Celestia's MambaBlock, and loss computation.
"""

import math
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, '/workspace/helix')
from retrieval_manifold import (
    RetrievalManifold, MambaBlock, RetrievalCWoLaDataset,
    pack_input, compute_k_target,
    INPUT_DIM, N_SCALING, D_MODEL, D_STATE, N_MAMBA_LAYERS,
)


class TestForwardShapes(unittest.TestCase):
    """Model forward works on synthetic (batch, input_dim) tensors with correct output shapes."""

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RetrievalManifold().to(self.device)

    def test_single_tick(self):
        x = torch.randn(1, INPUT_DIM, device=self.device)
        outputs, states = self.model(x)

        self.assertEqual(outputs['scaling'].shape, (1, N_SCALING))
        self.assertEqual(outputs['K'].shape, (1, 1))
        self.assertEqual(outputs['K_internal'].shape, (1, 1))
        self.assertEqual(len(states), N_MAMBA_LAYERS)

    def test_batch(self):
        x = torch.randn(32, INPUT_DIM, device=self.device)
        outputs, states = self.model(x)

        self.assertEqual(outputs['scaling'].shape, (32, N_SCALING))
        self.assertEqual(outputs['K'].shape, (32, 1))
        self.assertEqual(outputs['K_internal'].shape, (32, 1))

    def test_scaling_positive(self):
        """Softplus output must be non-negative."""
        x = torch.randn(16, INPUT_DIM, device=self.device)
        outputs, _ = self.model(x)
        self.assertTrue((outputs['scaling'] >= 0).all())

    def test_k_range(self):
        """K and K_internal must be in [0, 1] (sigmoid)."""
        x = torch.randn(16, INPUT_DIM, device=self.device)
        outputs, _ = self.model(x)
        for key in ('K', 'K_internal'):
            self.assertTrue((outputs[key] >= 0).all(), f"{key} < 0")
            self.assertTrue((outputs[key] <= 1).all(), f"{key} > 1")

    def test_state_carry(self):
        """Forward with state carry produces different outputs than without."""
        x = torch.randn(1, INPUT_DIM, device=self.device)
        out1, states1 = self.model(x)
        out2, states2 = self.model(x, states=states1)
        # Second pass with state should differ (state modifies hidden)
        self.assertFalse(torch.allclose(out1['K'], out2['K']),
                         "State carry should produce different K values")


class TestMambaStateShape(unittest.TestCase):
    """Mamba state shape matches what train_reactor_v7.py expects.

    ReactorV7 uses MambaBlock(d_model=192, d_state=32) with state shape
    (1, d_inner=384, d_state=32). Our MambaBlock(d_model=128, d_state=32)
    should produce state shape (1, d_inner=256, d_state=32).

    The key invariant: state shape is (batch, d_model*2, d_state) — this
    must hold for any MambaBlock instance, matching Celestia's implementation.
    """

    def test_state_shape_formula(self):
        """State shape follows (batch, d_model*2, d_state) for any d_model."""
        for d_model, d_state in [(128, 32), (192, 32), (256, 64)]:
            block = MambaBlock(d_model, d_state)
            x = torch.randn(1, d_model)
            _, state = block(x)
            expected = (1, d_model * 2, d_state)
            self.assertEqual(state.shape, expected,
                             f"d_model={d_model}, d_state={d_state}")

    def test_retrieval_manifold_states(self):
        """RetrievalManifold's Mamba states match expected shapes."""
        model = RetrievalManifold()
        expected_shapes = model.get_mamba_state_shapes()
        x = torch.randn(1, INPUT_DIM)
        _, states = model(x)

        for i, (state, expected) in enumerate(zip(states, expected_shapes)):
            self.assertEqual(state.shape, torch.Size(expected),
                             f"Layer {i} state shape mismatch")

    def test_celestia_reactor_compatible(self):
        """Our MambaBlock produces states compatible with Celestia's reactor pattern.

        ReactorV7 uses: MambaBlock(d_model=192, d_state=32)
        State shape: (1, 384, 32)

        Our block uses: MambaBlock(d_model=128, d_state=32)
        State shape: (1, 256, 32)

        The shapes differ (different d_model) but the STRUCTURE is identical:
        (batch, d_inner, d_state) where d_inner = d_model * 2.
        """
        # Celestia reactor dimensions
        reactor_block = MambaBlock(d_model=192, d_state=32)
        x_r = torch.randn(1, 192)
        _, state_r = reactor_block(x_r)
        self.assertEqual(state_r.shape, (1, 384, 32))

        # Our dimensions
        helix_block = MambaBlock(d_model=128, d_state=32)
        x_h = torch.randn(1, 128)
        _, state_h = helix_block(x_h)
        self.assertEqual(state_h.shape, (1, 256, 32))

        # Same structural pattern: (1, d_model*2, d_state)
        self.assertEqual(len(state_r.shape), len(state_h.shape))
        self.assertEqual(state_r.shape[0], state_h.shape[0])  # batch dim
        self.assertEqual(state_r.shape[2], state_h.shape[2])  # d_state matches


class TestLossComputation(unittest.TestCase):
    """Loss computation runs without error on a dummy batch."""

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RetrievalManifold().to(self.device)

    def test_bce_loss(self):
        """Primary BCE loss on K vs bucket label computes cleanly."""
        x = torch.randn(16, INPUT_DIM, device=self.device)
        labels = torch.randint(0, 2, (16,), device=self.device).float()

        outputs, _ = self.model(x)
        k_pred = outputs['K'].squeeze(-1)
        loss = nn.BCELoss()(k_pred, labels)

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        loss.backward()

    def test_multitask_loss(self):
        """Combined BCE (K) + MSE (K calibration) loss runs without error."""
        x = torch.randn(16, INPUT_DIM, device=self.device)
        labels = torch.randint(0, 2, (16,), device=self.device).float()
        k_targets = torch.rand(16, device=self.device)  # simulated rolling rate

        outputs, _ = self.model(x)
        k_pred = outputs['K'].squeeze(-1)

        bce_loss = nn.BCELoss()(k_pred, labels)
        k_mse = nn.MSELoss()(k_pred, k_targets)
        total_loss = bce_loss + 0.5 * k_mse  # weighted combination

        self.assertFalse(torch.isnan(total_loss))
        total_loss.backward()

    def test_scaling_loss(self):
        """Scaling head can be supervised with target weights."""
        x = torch.randn(16, INPUT_DIM, device=self.device)
        target_scaling = torch.rand(16, N_SCALING, device=self.device)

        outputs, _ = self.model(x)
        loss = nn.MSELoss()(outputs['scaling'], target_scaling)

        self.assertFalse(torch.isnan(loss))
        loss.backward()


class TestPackInput(unittest.TestCase):
    """pack_input helper produces correct tensor shapes."""

    def test_full_input(self):
        tier = torch.randn(9)
        query = torch.randn(20)
        cand = torch.randn(20)
        party = torch.zeros(8)
        party[3] = 1.0

        packed = pack_input(tier, query, cand, requery_delta_s=5.0,
                            party_id_onehot=party)
        self.assertEqual(packed.shape, (1, INPUT_DIM))

    def test_minimal_input(self):
        tier = torch.randn(9)
        packed = pack_input(tier)
        self.assertEqual(packed.shape, (1, INPUT_DIM))

    def test_log_dt_value(self):
        tier = torch.zeros(9)
        packed = pack_input(tier, requery_delta_s=10.0)
        # log1p(10) ≈ 2.397
        log_dt_idx = 9 + 20 + 20  # after tier + query + candidate
        self.assertAlmostEqual(packed[0, log_dt_idx].item(),
                               math.log1p(10.0), places=4)


class TestComputeKTarget(unittest.TestCase):
    """Rolling K target computation."""

    def test_all_ones(self):
        labels = torch.ones(10)
        k = compute_k_target(labels, window_size=5)
        self.assertTrue(torch.allclose(k, torch.ones(10)))

    def test_window(self):
        labels = torch.tensor([1., 0., 1., 0., 1.])
        k = compute_k_target(labels, window_size=3)
        # t=0: mean([1]) = 1.0
        # t=1: mean([1,0]) = 0.5
        # t=2: mean([0,1,0]) wait — window_size=3, so t=2: mean([1,0,1]) = 0.667
        # Actually: start = max(0, t-window_size+1)
        # t=0: start=0, mean([1]) = 1.0
        # t=1: start=0, mean([1,0]) = 0.5
        # t=2: start=0, mean([1,0,1]) = 0.667
        # t=3: start=1, mean([0,1,0]) = 0.333
        # t=4: start=2, mean([1,0,1]) = 0.667
        self.assertAlmostEqual(k[0].item(), 1.0, places=3)
        self.assertAlmostEqual(k[1].item(), 0.5, places=3)
        self.assertAlmostEqual(k[3].item(), 1/3, places=3)


class TestParamBudget(unittest.TestCase):
    """Parameter count stays under 2M budget."""

    def test_under_budget(self):
        model = RetrievalManifold()
        total = sum(p.numel() for p in model.parameters())
        self.assertLess(total, 2_000_000,
                        f"Parameter count {total:,} exceeds 2M budget")


if __name__ == '__main__':
    unittest.main(verbosity=2)
