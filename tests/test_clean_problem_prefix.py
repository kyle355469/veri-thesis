import io
import json
import unittest

from scripts.clean_problem_prefix import (
    DEFAULT_PREFIX,
    clean_jsonl,
    clean_problem_text,
    clean_problem_value,
)


class CleanProblemPrefixTests(unittest.TestCase):
    def test_clean_problem_text_removes_default_prefix_only(self):
        problem = DEFAULT_PREFIX + '"""\nDesign an inverter.\n"""'

        cleaned = clean_problem_text(problem)

        self.assertEqual(cleaned, '"""\nDesign an inverter.\n"""')

    def test_clean_problem_text_can_strip_wrapper_quotes(self):
        problem = DEFAULT_PREFIX + '"""\nDesign an inverter.\n"""'

        cleaned = clean_problem_text(problem, strip_wrapper_quotes=True)

        self.assertEqual(cleaned, "Design an inverter.")

    def test_clean_jsonl_preserves_other_fields(self):
        input_record = {
            "doc_id": "merged-1",
            "problem": DEFAULT_PREFIX + '"""\nDesign an inverter.\n"""',
            "solution": "module inv; endmodule",
            "tags": ["combinational"],
        }
        input_handle = io.StringIO(json.dumps(input_record) + "\n")
        output_handle = io.StringIO()

        stats = clean_jsonl(
            input_handle,
            output_handle,
            field="problem",
            prefix=DEFAULT_PREFIX,
            strip_wrapper_quotes=True,
        )

        self.assertEqual(stats, {"records": 1, "changed": 1})
        output_record = json.loads(output_handle.getvalue())
        self.assertEqual(output_record["problem"], "Design an inverter.")
        self.assertEqual(output_record["solution"], input_record["solution"])
        self.assertEqual(output_record["tags"], input_record["tags"])

    def test_clean_problem_value_handles_chat_prompt_messages(self):
        prompt = [
            {
                "role": "user",
                "content": DEFAULT_PREFIX + '"""\nDesign an inverter.\n"""',
            }
        ]

        cleaned, changed = clean_problem_value(prompt, strip_wrapper_quotes=True)

        self.assertTrue(changed)
        self.assertEqual(cleaned, [{"role": "user", "content": "Design an inverter."}])

    def test_clean_jsonl_auto_cleans_raw_prompt_and_preserves_completion(self):
        input_record = {
            "prompt": [
                {
                    "content": DEFAULT_PREFIX + '"""\nDesign an inverter.\n"""',
                    "role": "user",
                }
            ],
            "completion": [
                {
                    "content": "<answer>\n```verilog\nmodule inv; endmodule\n```\n</answer>",
                    "role": "assistant",
                }
            ],
            "length": 123,
        }
        input_handle = io.StringIO(json.dumps(input_record) + "\n")
        output_handle = io.StringIO()

        stats = clean_jsonl(
            input_handle,
            output_handle,
            field="auto",
            prefix=DEFAULT_PREFIX,
            strip_wrapper_quotes=True,
        )

        self.assertEqual(stats, {"records": 1, "changed": 1})
        output_record = json.loads(output_handle.getvalue())
        self.assertEqual(output_record["prompt"][0]["content"], "Design an inverter.")
        self.assertEqual(output_record["prompt"][0]["role"], "user")
        self.assertEqual(output_record["completion"], input_record["completion"])
        self.assertEqual(output_record["length"], input_record["length"])


if __name__ == "__main__":
    unittest.main()
