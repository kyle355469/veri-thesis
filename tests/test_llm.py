import unittest

from rag_rtl.llm import extract_code


class LlmTests(unittest.TestCase):
    def test_extract_code_prefers_final_rtl_tag(self):
        response = """The tool result passed.
<final_rtl>
```verilog
module dut;
endmodule
```
</final_rtl>
Extra text that must not be treated as RTL."""

        self.assertEqual(extract_code(response), "module dut;\nendmodule")

    def test_extract_code_accepts_unfenced_final_rtl_tag(self):
        response = """<final_rtl>
module dut;
endmodule
</final_rtl>"""

        self.assertEqual(extract_code(response), "module dut;\nendmodule")

    def test_extract_code_uses_last_final_rtl_tag_after_prompt_echo(self):
        response = """We need to produce final HDL code inside exactly one <final_rtl> block.
The user request says: "Return the final HDL code inside exactly one <final_rtl>...</final_rtl> block."

<final_rtl>
```verilog
module real_answer;
endmodule
```
</final_rtl>"""

        self.assertEqual(extract_code(response), "module real_answer;\nendmodule")

    def test_extract_code_uses_last_fenced_block_in_selected_source(self):
        response = """Here is an old-style response without final tags.
```verilog
module prompt_example;
endmodule
```

```verilog
module real_answer;
endmodule
```"""

        self.assertEqual(extract_code(response), "module real_answer;\nendmodule")

    def test_extract_code_ignores_placeholder_final_rtl_example(self):
        response = """Return format:
<final_rtl>
```verilog
...code...
```
</final_rtl>

```verilog
module real_answer;
endmodule
```"""

        self.assertEqual(extract_code(response), "module real_answer;\nendmodule")

    def test_extract_code_keeps_fenced_fallback_for_old_responses(self):
        response = """```verilog
module old_style;
endmodule
```"""

        self.assertEqual(extract_code(response), "module old_style;\nendmodule")


if __name__ == "__main__":
    unittest.main()
