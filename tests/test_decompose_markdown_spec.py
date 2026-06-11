import tempfile
import unittest
from pathlib import Path

from scripts.decompose_markdown_spec import decompose_markdown, write_section_files


class DecomposeMarkdownSpecTests(unittest.TestCase):
    def test_decomposes_sections_by_heading(self):
        text = "# Top\nintro\n## Child\nbody\n# Next #\nmore\n"

        sections = decompose_markdown(text)

        self.assertEqual([section.title for section in sections], ["Top", "Child", "Next"])
        self.assertEqual([section.level for section in sections], [1, 2, 1])
        self.assertEqual(sections[0].content, "# Top\nintro\n")

    def test_ignores_hash_headings_inside_fenced_code(self):
        text = "# Spec\n```markdown\n# Not a heading\n```\n## Real\ncontent\n"

        sections = decompose_markdown(text)

        self.assertEqual([section.title for section in sections], ["Spec", "Real"])
        self.assertIn("# Not a heading", sections[0].content)

    def test_can_split_only_top_level_headings(self):
        text = "# A\n## B\nb\n# C\nc\n"

        sections = decompose_markdown(text, max_level=1)

        self.assertEqual([section.title for section in sections], ["A", "C"])
        self.assertIn("## B\nb\n", sections[0].content)

    def test_writes_sanitized_section_files(self):
        sections = decompose_markdown("# Bus / Interface\nbody\n# Bus / Interface\nagain\n")

        with tempfile.TemporaryDirectory() as tmp:
            written = write_section_files(sections, Path(tmp))

            self.assertEqual(
                [path.name for path in written],
                ["000-bus-interface.md", "001-bus-interface-2.md"],
            )
            self.assertEqual(written[0].read_text(encoding="utf-8"), "# Bus / Interface\nbody\n")


if __name__ == "__main__":
    unittest.main()
