import unittest

from rag_rtl.siliconmind_utils import (
    SILICONMIND_WORKFLOW_GUIDE,
    get_attempt_prompts,
    get_debug_prompts,
    get_test_prompts,
    parse_code,
    wrap_code,
    wrap_text,
)


class SiliconMindUtilsTests(unittest.TestCase):
    def test_parse_code_uses_last_verilog_fence(self):
        text = """```verilog
module old;
endmodule
```

```systemverilog
module chosen;
endmodule
```"""

        self.assertEqual(parse_code(text), "module chosen;\nendmodule")

    def test_wrap_helpers_match_chat_prompt_shape(self):
        self.assertEqual(wrap_text("Design an inverter."), '"""\nDesign an inverter.\n"""')
        self.assertEqual(wrap_code("module dut; endmodule"), "```verilog\nmodule dut; endmodule\n```")
        prompts = get_attempt_prompts(["Design an inverter."], internal_workflow=True)
        self.assertEqual(prompts[0][0]["role"], "user")
        self.assertIn("First draft a solution", prompts[0][0]["content"])

    def test_test_and_debug_prompt_builders_pair_inputs(self):
        test_prompt = get_test_prompts(["Problem"], ["module dut; endmodule"])[0][0]["content"]
        debug_prompt = get_debug_prompts(
            ["Problem"],
            ["module broken;"],
            ["Missing endmodule."],
        )[0][0]["content"]

        self.assertIn("[DESIGN NEEDS FIXING]", test_prompt)
        self.assertIn("```verilog\nmodule dut; endmodule\n```", test_prompt)
        self.assertIn("Missing endmodule.", debug_prompt)
        self.assertIn("corrected Verilog code", debug_prompt)

    def test_workflow_guide_is_visible_for_prompting_module(self):
        self.assertIn("SiliconMind-style internal workflow", SILICONMIND_WORKFLOW_GUIDE)


if __name__ == "__main__":
    unittest.main()
