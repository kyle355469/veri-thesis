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

    def test_extract_code_prefers_answer_block_after_malformed_fence(self):
        response = """```verilog. Provide final answer.

Thus final answer: (the code block).
</think>

<answer>
```verilog
module A_Array (
    input               b_i,
    input  signed [255:0] a,
    output reg signed [255:0] sum
);
    integer i;

    always @* begin
        sum = '0;
        for (i = 0; i < 256; i = i + 1) begin
            sum[i] = a[i] & b_i;
        end
    end
endmodule
```
</answer>"""

        extracted = extract_code(response)

        self.assertTrue(extracted.startswith("module A_Array"))
        self.assertIn("sum[i] = a[i] & b_i;", extracted)
        self.assertNotIn("Provide final answer", extracted)
        self.assertNotIn("<answer>", extracted)

    def test_extract_code_accepts_plain_hdl(self):
        response = """module dut;
endmodule"""

        self.assertEqual(extract_code(response), "module dut;\nendmodule")

    def test_extract_code_returns_empty_when_no_hdl_is_present(self):
        response = "I cannot provide the code, but the design should be an inverter."

        self.assertEqual(extract_code(response), "")


if __name__ == "__main__":
    unittest.main()
