"""Check the published directory and its PDF build entry point."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_baseline_path0_tutorial as builder
from baseline_path0_layout import DATA, DOCS, FIGURES, RESULT


class LayoutTests(unittest.TestCase):
    def test_only_final_pdfs_at_top_level(self):
        self.assertEqual(
            {path.name for path in RESULT.iterdir() if path.is_file()},
            {f"{builder.STEM}.pdf", "path0_deadline_and_staggering_notes_v3.pdf"},
        )

    def test_supporting_files_are_in_their_directories(self):
        self.assertTrue((DATA / "result.json").is_file())
        self.assertTrue((DATA / "physical_block_trace.csv").is_file())
        self.assertTrue((FIGURES / "05a_per_npu_ssu_events.pdf").is_file())
        self.assertTrue((DOCS / "report.md").is_file())
        self.assertTrue((DOCS / "path0_deadline_and_staggering_notes_v3.md").is_file())

    def test_tutorial_image_links_resolve_from_source(self):
        source = DOCS / f"{builder.STEM}.md"
        references = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source.read_text())
        self.assertEqual(len(references), 11)
        for reference in references:
            with self.subTest(reference=reference):
                self.assertTrue((source.parent / reference).is_file())

    def test_builder_uses_moved_source_data_and_resources(self):
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[0] == "pandoc":
                Path(command[command.index("--output") + 1]).write_bytes(b"test PDF")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tutorial.pdf"
            with patch("sys.argv", ["build", "--output", str(output)]), \
                 patch.object(builder.shutil, "which", return_value="available"), \
                 patch.object(builder.subprocess, "run", side_effect=run), \
                 patch("builtins.print"):
                builder.main()
            self.assertEqual(output.read_bytes(), b"test PDF")
            self.assertEqual(commands[0][-1], str(DATA / "tutorial_numeric_audit.json"))
            pandoc = commands[-1]
            self.assertEqual(pandoc[1], str(DOCS / f"{builder.STEM}.md"))
            self.assertEqual(pandoc[pandoc.index("--resource-path") + 1], str(DOCS))


if __name__ == "__main__":
    unittest.main()
