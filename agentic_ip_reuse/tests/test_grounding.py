import unittest

from agentic_ip_reuse.grounding import (
    completeness_gaps,
    completeness_score,
    ground_reuse_decisions,
)

CATALOG = ["e203_exu", "e203_ifu", "e203_lsu", "sirv_gnrl_ram"]


class GroundReuseDecisionsTests(unittest.TestCase):
    def test_exact_match_is_kept(self):
        plan = {"reuse_decisions": [{"module_name": "m", "selected_ip": "e203_exu"}]}
        plan, report = ground_reuse_decisions(plan, CATALOG)
        self.assertEqual(plan["reuse_decisions"][0]["selected_ip"], "e203_exu")
        self.assertEqual(report["exact"], 1)

    def test_suffixed_name_is_remapped(self):
        plan = {"reuse_decisions": [{"module": "m", "ip": "e203_exu_core"}]}
        plan, report = ground_reuse_decisions(plan, CATALOG)
        entry = plan["reuse_decisions"][0]
        self.assertEqual(entry["selected_ip"], "e203_exu")
        self.assertEqual(entry["module_name"], "m")
        self.assertEqual(report["remapped"], 1)

    def test_unknown_name_is_dropped_and_marked_new_rtl(self):
        plan = {"reuse_decisions": [{"module_name": "m", "ip_id": "nonexistent_block"}]}
        plan, report = ground_reuse_decisions(plan, CATALOG)
        entry = plan["reuse_decisions"][0]
        self.assertIsNone(entry["selected_ip"])
        self.assertTrue(entry["new_rtl_required"])
        self.assertEqual(report["dropped"], 1)

    def test_empty_catalog_leaves_names_untouched(self):
        plan = {"reuse_decisions": [{"module_name": "m", "selected_ip": "whatever"}]}
        plan, report = ground_reuse_decisions(plan, [])
        self.assertEqual(plan["reuse_decisions"][0]["selected_ip"], "whatever")
        self.assertEqual(report["unmatched_no_catalog"], 1)

    def test_case_insensitive_exact_match(self):
        plan = {"reuse_decisions": [{"module_name": "m", "selected_ip": "E203_EXU"}]}
        plan, report = ground_reuse_decisions(plan, CATALOG)
        self.assertEqual(plan["reuse_decisions"][0]["selected_ip"], "e203_exu")


class CompletenessTests(unittest.TestCase):
    def test_gaps_flagged_when_catalog_present(self):
        gaps = completeness_gaps({"requirements": {"functionality": ["x"]}}, has_catalog=True)
        self.assertIn("reuse_decisions", gaps)
        self.assertIn("integration_plan", gaps)

    def test_no_reuse_gaps_for_leaf_module_without_catalog(self):
        gaps = completeness_gaps({"requirements": {"functionality": ["x"]}}, has_catalog=False)
        self.assertNotIn("reuse_decisions", gaps)
        self.assertNotIn("integration_plan", gaps)

    def test_score_prefers_more_complete_plan(self):
        thin = {"reuse_decisions": [], "integration_plan": [], "modules": [{"name": "m"}]}
        rich = {
            "reuse_decisions": [{"module_name": "m", "selected_ip": "e203_exu"}],
            "integration_plan": ["wire it"],
            "modules": [{"name": "m"}],
        }
        self.assertGreater(completeness_score(rich), completeness_score(thin))


if __name__ == "__main__":
    unittest.main()
