from __future__ import annotations

import unittest

from sparse_circuit_repro.audit import audit


class ReleaseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit()

    def test_integrity(self) -> None:
        self.assertTrue(self.result["integrity"]["valid"])

    def test_full_candidate_universes(self) -> None:
        self.assertEqual(self.result["candidate_universes"]["quote"]["count"], 64)
        self.assertEqual(self.result["candidate_universes"]["bracket"]["count"], 133)

    def test_necessity_results(self) -> None:
        self.assertLess(self.result["necessity"]["quote_U"]["accuracy_after_ablation"], 0.60)
        self.assertLess(self.result["necessity"]["bracket_R"]["accuracy_after_ablation"], 0.65)

    def test_no_redundancy_claim(self) -> None:
        self.assertEqual(self.result["redundancy"], {"quote": "not_recovered", "bracket": "not_recovered"})

    def test_progressive_depth_model(self) -> None:
        progressive = self.result["progressive_depth"]
        self.assertEqual(progressive["accepted_model"], "X -> D -> R -> Y")
        self.assertEqual(
            progressive["upstream_depth_handle"],
            ["2.attn.resid_delta:1249", "3.attn.resid_delta:1249"],
        )
        self.assertGreaterEqual(progressive["Dte_pearson"], 0.99)
        self.assertTrue(progressive["all_heldout_gates_pass"])


if __name__ == "__main__":
    unittest.main()
