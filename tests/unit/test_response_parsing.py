#!/usr/bin/env python3
"""
Unit tests for Fast Apply response parsing functionality.
Tests edge cases including multiple blocks, empty responses, and malformed output.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_fast_apply_response(raw_response: str) -> str:
    """Parse Fast Apply response with stricter validation."""
    if not raw_response or not raw_response.strip():
        raise ValueError("Fast Apply API response is empty")

    max_char_response = 240_000  # explicit cap (~40k tokens @6 chars/token)
    if len(raw_response) > max_char_response:
        # Truncate for safety
        raw_response = raw_response[:max_char_response]

    # Try XML tag extraction first
    if "<updated-code>" in raw_response and "</updated-code>" in raw_response:
        import re

        segments = []
        # Use non-greedy matching to capture content between tags
        pattern = r"<updated-code[^>]*>(.*?)</updated-code>"
        matches = re.findall(pattern, raw_response, re.DOTALL)

        for match in matches:
            content = match.strip()
            if content:
                segments.append(content)

        if len(segments) > 1:
            raise ValueError("Multiple <updated-code> blocks detected.")

        if segments:
            merged_content = "\n".join(segments)
            return merged_content

    # Fallback to original parsing logic
    lines = raw_response.split("\n")
    content_lines = []
    in_code_block = False

    for line in lines:
        if "<updated-code" in line and ">" in line:
            in_code_block = True
            # Handle case where tag is mixed with content
            # Find the opening tag and extract content after it
            tag_start = line.find("<updated-code")
            tag_end = line.find(">", tag_start) + 1
            if tag_end < len(line):
                remaining_content = line[tag_end:]
                if remaining_content.strip():
                    content_lines.append(remaining_content.strip())
        elif "</updated-code>" in line:
            in_code_block = False
            # Handle case where end tag is mixed with content
            parts = line.split("</updated-code>")
            if parts[0].strip():
                content_lines.append(parts[0].strip())
            if len(parts) > 1 and parts[1].strip():
                content_lines.append(parts[1].strip())
        elif in_code_block:
            content_lines.append(line)

    if not content_lines:
        raise ValueError("No updated code found in response")

    return "\n".join(content_lines)


class TestResponseParsing(unittest.TestCase):
    """Test Fast Apply response parsing with various edge cases."""

    def test_empty_response(self):
        """Test handling of empty responses."""
        with self.assertRaises(ValueError) as cm:
            _parse_fast_apply_response("")
        self.assertIn("response is empty", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _parse_fast_apply_response("   ")
        self.assertIn("response is empty", str(cm.exception))

    def test_single_xml_block(self):
        """Test parsing of single XML block."""
        response = """Some introductory text
<updated-code>
def hello():
    return "world"
</updated-code>
Some closing text"""

        result = _parse_fast_apply_response(response)
        expected = 'def hello():\n    return "world"'
        self.assertEqual(result, expected)

    def test_multiple_xml_blocks_rejection(self):
        """Test rejection of multiple XML blocks."""
        response = """<updated-code>
def first():
    pass
</updated-code>
<updated-code>
def second():
    pass
</updated-code>"""

        with self.assertRaises(ValueError) as cm:
            _parse_fast_apply_response(response)
        self.assertIn("Multiple <updated-code> blocks", str(cm.exception))

    def test_fallback_code_block_parsing(self):
        """Test fallback parsing for code blocks."""
        response = """Here's the updated code:
<updated-code>def hello():</updated-code>
    return "world"
<updated-code></updated-code>
"""

        result = _parse_fast_apply_response(response)
        # Should only capture content within tags
        expected = "def hello():"
        self.assertEqual(result, expected)

    def test_no_code_block_error(self):
        """Test error when no code blocks found."""
        response = "Here is some text without any code blocks"

        with self.assertRaises(ValueError) as cm:
            _parse_fast_apply_response(response)
        self.assertIn("No updated code found", str(cm.exception))

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in code blocks."""
        response = """<updated-code>

def hello():


    return "world"


</updated-code>"""

        result = _parse_fast_apply_response(response)
        expected = 'def hello():\n\n\n    return "world"'
        self.assertEqual(result, expected)

    def test_mixed_content_extraction(self):
        """Test extraction from mixed content with multiple formats."""
        response = """The response contains:
<updated-code>def main():
    print("Hello")</updated-code>
Additional explanation here."""

        result = _parse_fast_apply_response(response)
        expected = 'def main():\n    print("Hello")'
        self.assertEqual(result, expected)

    def test_large_response_truncation(self):
        """Test handling of oversized responses."""
        # Create a large response (> 240,000 chars)
        large_content = "x" * 250_000
        response = f"<updated-code>{large_content}</updated-code>"

        result = _parse_fast_apply_response(response)
        # Should be truncated to MAX_CHAR_RESPONSE minus the XML tags
        self.assertLess(len(result), 240_000)

    def test_xml_with_attributes(self):
        """Test XML blocks with attributes are handled correctly."""
        response = """<updated-code attr="value">
def hello():
    return "world"
</updated-code>"""

        result = _parse_fast_apply_response(response)
        expected = '''def hello():
    return "world"'''
        self.assertEqual(result, expected)

    def test_nested_xml_like_content(self):
        """Test handling of content that looks like XML but isn't a code block."""
        response = """<updated-code>
def parse_xml(xml_string):
    # This contains XML-like content but should be preserved
    if "<tag>" in xml_string:
        return True
</updated-code>"""

        result = _parse_fast_apply_response(response)
        expected = """def parse_xml(xml_string):
    # This contains XML-like content but should be preserved
    if "<tag>" in xml_string:
        return True"""
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
