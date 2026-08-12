"""Markdown parsing, especially the code-fence hazard.

The corpus has 555 fenced code blocks, many holding Python list literals like
``[[1,2],[3,4]]``. A parser that does not mask code invents links from them.
"""

from __future__ import annotations

from forge.parsing.markdown import mask_code, parse_markdown, split_frontmatter


class TestCodeFenceMasking:
    def test_python_matrix_literal_is_not_a_wikilink(self):
        text = """# Title

Real link: [[Binary Search]]

```python
grid = [[0, 1], [1, 0]]
adj = [[1], [0], [2]]
```

Another real link: [[DFS]]
"""
        parsed = parse_markdown(text)
        targets = [w.target for w in parsed.wikilinks]
        assert targets == ["Binary Search", "DFS"]

    def test_inline_code_is_masked(self):
        parsed = parse_markdown("Use `[[not a link]]` but [[this is]].")
        assert [w.target for w in parsed.wikilinks] == ["this is"]

    def test_tilde_fences(self):
        parsed = parse_markdown("~~~\n[[x]]\n~~~\n[[y]]")
        assert [w.target for w in parsed.wikilinks] == ["y"]

    def test_line_numbers_survive_masking(self):
        """Masking must preserve line structure or every span location breaks."""
        text = "# T\n\n```py\na=[[1]]\nb=2\n```\n\n[[Real]]\n"
        parsed = parse_markdown(text)
        assert parsed.wikilinks[0].line == 8

    def test_mask_preserves_line_count(self):
        text = "a\n```\nb\nc\n```\nd"
        masked, count, _ = mask_code(text)
        assert masked.count("\n") == text.count("\n")
        assert count == 1

    def test_code_languages_captured(self):
        parsed = parse_markdown("```python\nx=1\n```\n\n```bash\nls\n```")
        assert parsed.code_block_count == 2
        assert parsed.code_languages == ["python", "bash"]

    def test_unclosed_fence_masks_to_end(self):
        parsed = parse_markdown("# T\n\n```python\n[[not a link]]\n")
        assert parsed.wikilinks == []


class TestFrontmatterSplit:
    def test_split_returns_body_and_end_line(self):
        raw, end = split_frontmatter("---\na: 1\nb: 2\n---\n# Title\n")
        assert raw == "a: 1\nb: 2"
        assert end == 4

    def test_absent(self):
        assert split_frontmatter("# Title\n") == (None, 0)

    def test_unterminated_is_not_frontmatter(self):
        assert split_frontmatter("---\na: 1\n# Title\n") == (None, 0)

    def test_frontmatter_excluded_from_headings(self):
        parsed = parse_markdown("---\ntitle: x\n---\n# Real Heading\n")
        assert [h.text for h in parsed.headings] == ["Real Heading"]

    def test_frontmatter_wikilinks_are_still_captured(self):
        """The related: field holds real links even though its YAML is broken."""
        parsed = parse_markdown("---\nrelated: [[A]], [[B]]\n---\n# T\n\n[[C]]\n")
        assert [w.target for w in parsed.wikilinks] == ["C", "A", "B"]
        assert [w.in_frontmatter for w in parsed.wikilinks] == [False, True, True]


class TestWikilinkForms:
    def test_alias_and_anchor(self):
        parsed = parse_markdown("[[Target|Alias]] [[Other#Section]] [[Both#S|A]]")
        links = parsed.wikilinks
        assert (links[0].target, links[0].alias) == ("Target", "Alias")
        assert (links[1].target, links[1].anchor) == ("Other", "Section")
        assert (links[2].target, links[2].anchor, links[2].alias) == ("Both", "S", "A")

    def test_headings_and_title(self):
        parsed = parse_markdown("# Title\n## Sub\n### Deep\n")
        assert [(h.level, h.text) for h in parsed.headings] == [
            (1, "Title"),
            (2, "Sub"),
            (3, "Deep"),
        ]
        assert parsed.title == "Title"

    def test_tags_extracted_but_not_headings(self):
        parsed = parse_markdown("# Title\n\n#status/active and #stack/python\n")
        assert parsed.tags == ["status/active", "stack/python"]

    def test_markdown_links_captured_images_ignored(self):
        parsed = parse_markdown("[text](a.md) and ![img](pic.png)")
        assert [m.target for m in parsed.markdown_links] == ["a.md"]

    def test_crlf_normalized(self):
        parsed = parse_markdown("# T\r\n\r\n[[X]]\r\n")
        assert [w.target for w in parsed.wikilinks] == ["X"]
