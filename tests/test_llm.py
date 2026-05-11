import unittest

from rag_rtl.llm import extract_code


class LlmTests(unittest.TestCase):
    def test_extract_code_accepts_siliconmind_style_fence(self):
        response = """```systemverilog
module dut;
endmodule
```"""

        self.assertEqual(extract_code(response), "module dut;\nendmodule")

    def test_extract_code_uses_last_fenced_block(self):
        response = """```verilog
module prompt_example;
endmodule
```

```verilog
module real_answer;
endmodule
```"""

        self.assertEqual(extract_code(response), "module real_answer;\nendmodule")

    def test_extract_code_accepts_plain_hdl(self):
        response = """module dut;
endmodule"""

        self.assertEqual(extract_code(response), "module dut;\nendmodule")

    def test_extract_code_returns_empty_when_no_hdl_is_present(self):
        response = "I cannot provide the code, but the design should be an inverter."

        self.assertEqual(extract_code(response), "")


if __name__ == "__main__":
    unittest.main()
