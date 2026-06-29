import unittest
import torch
import numpy as np
from branches_lm_v0_2_1 import (
    TrainableAIEngine,
    SimplicialComplex,
    SimplicialNN,
    DiagnosticBus,
    DCSR,
    DCSC,
    BUS
)

class TestProductionAudit(unittest.TestCase):
    def setUp(self):
        self.engine = TrainableAIEngine(d_model=32, n_layers=2)

    def test_simplicial_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex((0, 1, 2))

        # Verify boundary operators
        B1 = sc.boundary_operator(1)
        B2 = sc.boundary_operator(2)

        # Fundamental property: B1 @ B2 = 0
        composition = torch.matmul(B1, B2)
        self.assertTrue(torch.allclose(composition, torch.zeros_like(composition), atol=1e-5))

        # Verify Hodge Laplacians
        L0 = sc.hodge_laplacian(0)
        self.assertEqual(L0.shape, (3, 3))
        # L0 = B1 @ B1.T
        self.assertTrue(torch.allclose(L0, torch.matmul(B1, B1.T), atol=1e-5))

    def test_simplicial_nn(self):
        sc = SimplicialComplex()
        sc.add_simplex((0, 1, 2))
        snn = SimplicialNN(sc, d_features=8, d_hidden=16, target_dim=0)
        h_nodes = torch.randn(2, 3, 8) # (batch, seq, d_model)
        out = snn(h_nodes)
        self.assertEqual(out.shape, (2, 3, 16))
        self.assertTrue(torch.all(out >= 0)) # ReLU

    def test_sparse_roundtrip(self):
        dense = torch.randn(20, 20) * (torch.rand(20, 20) > 0.9)
        dcsr = DCSR(20, 20, dense)
        self.assertTrue(torch.allclose(dense, dcsr.to_dense(), atol=1e-5))

        dcsc = DCSC(20, 20, dense)
        self.assertTrue(torch.allclose(dense, dcsc.to_dense(), atol=1e-5))

    def test_diagnostic_bus_singleton(self):
        bus1 = DiagnosticBus()
        bus2 = DiagnosticBus()
        self.assertIs(bus1, bus2)

if __name__ == "__main__":
    unittest.main()
