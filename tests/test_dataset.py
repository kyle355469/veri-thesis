import unittest

from rag_rtl.dataset import extract_rtl_code, normalize_record


class DatasetTests(unittest.TestCase):
    def test_extract_rtl_code_removes_private_reasoning(self):
        text = """<think>hidden reasoning</think>
<answer>
```verilog
module invert(input wire i, output wire o);
assign o = ~i;
endmodule
```
</answer>"""
        self.assertTrue(extract_rtl_code(text).startswith("module invert"))
        self.assertNotIn("hidden reasoning", extract_rtl_code(text))

    def test_normalize_record_from_chat_shape(self):
        record = {
            "prompt": [{"role": "user", "content": "Design an inverter."}],
            "completion": [{"role": "assistant", "content": "```verilog\nmodule invert(); endmodule\n```"}],
        }
        document = normalize_record(record, 0)
        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.problem, "Design an inverter.")
        self.assertIn("module invert", document.solution)


if __name__ == "__main__":
    unittest.main()
