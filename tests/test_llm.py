import unittest

from rag_rtl.llm import extract_code, split_reasoning


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

    # --- reasoning-model formats: CodeV-R1 (<think>...</think>) and
    # --- SiliconMind-V1 (Qwen3-Thinking-2507: bare closing </think> only) ---

    def test_codev_r1_final_answer_beats_draft_inside_think(self):
        response = """<think>
Let me try a draft first:
```verilog
module draft_version;
endmodule
```
Hmm, that is wrong. The final version follows.
</think>

```verilog
module final_version;
endmodule
```"""

        self.assertEqual(extract_code(response), "module final_version;\nendmodule")

    def test_siliconmind_bare_closing_think_tag(self):
        response = """First I consider the interface.
module counter needs a clock, so I will write an always block.
</think>

```verilog
module counter(input clk, output reg [3:0] q);
  always @(posedge clk) q <= q + 1;
endmodule
```"""

        extracted = extract_code(response)

        self.assertTrue(extracted.startswith("module counter"))
        self.assertNotIn("I consider the interface", extracted)

    def test_bare_hdl_after_think_close(self):
        response = """thinking about ports...
</think>
module top(input a, output b);
assign b = a;
endmodule"""

        self.assertTrue(extract_code(response).startswith("module top"))

    def test_reasoning_prose_is_not_returned_as_code(self):
        # The whole reply is reasoning (truncated before </think>); prose that
        # happens to start a line with "module" must not be returned as HDL.
        response = """<think>
module counter should have four ports, let me think about the reset polarity
and whether the spec wants synchronous or asynchronous behavior."""

        self.assertEqual(extract_code(response), "")

    def test_truncated_reply_salvages_draft_from_reasoning(self):
        response = """<think>
Here is my draft:
```verilog
module salvage_me(input a, output b);
assign b = a;
endmodule
```
Now let me double-check the timing beha"""

        self.assertEqual(
            extract_code(response),
            "module salvage_me(input a, output b);\nassign b = a;\nendmodule",
        )

    def test_unterminated_final_fence_is_salvaged(self):
        response = """<think>reasoning</think>

```verilog
module cut_off(input a, output b);
assign b = a;
endmodule"""

        self.assertTrue(extract_code(response).startswith("module cut_off"))

    def test_split_reasoning_pairs_and_bare_close(self):
        self.assertEqual(split_reasoning("<think>notes</think>answer"), ("notes", "answer"))
        self.assertEqual(split_reasoning("notes</think>answer"), ("notes", "answer"))
        self.assertEqual(split_reasoning("plain answer"), ("", "plain answer"))
        reasoning, answer = split_reasoning("<think>cut off mid thought")
        self.assertEqual(reasoning, "cut off mid thought")
        self.assertEqual(answer, "")


if __name__ == "__main__":
    unittest.main()
