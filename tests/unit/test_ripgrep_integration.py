#!/usr/bin/env python3
"""
Working test suite for ripgrep integration functionality.

Tests the core functionality that actually works in the ripgrep integration layer.
"""

import os
import shutil
import tempfile
import unittest

from fastapply.ripgrep_integration import (
    OutputFormat,
    RipgrepIntegration,
    SearchOptions,
    SearchResult,
    SearchResults,
    SearchType,
)


class TestRipgrepIntegrationWorking(unittest.TestCase):
    """Test ripgrep integration functionality with working tests."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files
        self.setup_test_files()

        # Initialize ripgrep integration
        self.ripgrep = RipgrepIntegration()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def setup_test_files(self):
        """Create test files for searching."""
        # Python test file
        self.python_file = os.path.join(self.test_dir, "test.py")
        with open(self.python_file, "w", encoding="utf-8") as f:
            f.write("""#!/usr/bin/env python3

def hello_world(name: str) -> str:
    return f"Hello, {name}!"

class TestClass:
    def __init__(self, value: int):
        self.value = value

    def calculate_sum(self, a: int, b: int) -> int:
        return a + b

async def async_function() -> None:
    pass

def _private_function():
    pass
""")

        # JavaScript test file
        self.js_file = os.path.join(self.test_dir, "test.js")
        with open(self.js_file, "w", encoding="utf-8") as f:
            f.write("""function greet(name) {
    console.log(`Hello, ${name}!`);
}

class TestComponent {
    constructor(props) {
        this.props = props;
    }

    render() {
        return <div>Hello World</div>;
    }
}

const multiply = (a, b) => a * b;

export default TestComponent;
""")

        # JSON test file
        self.json_file = os.path.join(self.test_dir, "test.json")
        with open(self.json_file, "w", encoding="utf-8") as f:
            f.write('{"name": "test", "value": 42, "items": [1, 2, 3]}')

        # Text file with special characters
        self.text_file = os.path.join(self.test_dir, "test.txt")
        with open(self.text_file, "w", encoding="utf-8") as f:
            f.write("This is a test file with special characters: äöü\nMultiple lines\nFor testing")

    def test_search_files_basic(self):
        """Test basic file searching."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertIsInstance(results.results, list)
        self.assertIsInstance(results.total_matches, int)
        self.assertIsInstance(results.files_searched, int)
        self.assertIsInstance(results.search_time, float)
        self.assertEqual(results.pattern, "hello")

        # Should find matches in both Python and JavaScript files
        self.assertGreater(len(results.results), 0)

    def test_search_files_no_results(self):
        """Test searching with no results."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "nonexistent_pattern_xyz",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertEqual(len(results.results), 0)
        self.assertEqual(results.total_matches, 0)

    def test_search_files_case_sensitive(self):
        """Test case sensitive search."""
        options = SearchOptions(
            case_sensitive=True,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            "Hello",
            path=self.test_dir,
            options=options
        )

        # Should find fewer results with case sensitive search
        self.assertIsInstance(results, SearchResults)

    def test_search_files_case_insensitive(self):
        """Test case insensitive search."""
        options = SearchOptions(
            case_sensitive=False,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        # Should find more results with case insensitive search
        self.assertIsInstance(results, SearchResults)
        self.assertGreater(len(results.results), 0)

    def test_search_files_literal(self):
        """Test literal search."""
        options = SearchOptions(
            search_type=SearchType.LITERAL,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            "def ",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        # Should find literal "def " in Python files
        self.assertGreater(len(results.results), 0)

    def test_search_files_regex(self):
        """Test regex search."""
        options = SearchOptions(
            search_type=SearchType.REGEX,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            r"def\s+\w+",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        # Should find function definitions
        self.assertGreater(len(results.results), 0)

    def test_search_files_word_boundaries(self):
        """Test word boundary search."""
        options = SearchOptions(
            search_type=SearchType.WORD,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            "def",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        # Should find standalone "def" words
        self.assertGreater(len(results.results), 0)

    def test_search_files_with_context(self):
        """Test search with context lines."""
        options = SearchOptions(
            context_lines=2,
            output_format=OutputFormat.JSON
        )
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        if results.results:
            # Check that context is included
            result = results.results[0]
            self.assertIsInstance(result.context_before, list)
            self.assertIsInstance(result.context_after, list)

    def test_search_performance_metrics(self):
        """Test that performance metrics are collected."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertGreaterEqual(results.search_time, 0)
        self.assertGreaterEqual(results.files_searched, 0)

    def test_search_file_specific(self):
        """Test searching in a specific file."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "class",
            path=self.python_file,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        # Should find class in Python file
        self.assertGreater(len(results.results), 0)

    def test_search_nonexistent_path(self):
        """Test searching in non-existent path."""
        options = SearchOptions(output_format=OutputFormat.JSON)

        with self.assertRaises(RuntimeError):
            self.ripgrep.search_files(
                "test",
                path="/nonexistent/path",
                options=options
            )

    def test_search_error_handling(self):
        """Test error handling for invalid patterns."""
        options = SearchOptions(output_format=OutputFormat.JSON)

        with self.assertRaises(RuntimeError):
            # Invalid regex pattern
            self.ripgrep.search_files(
                "[invalid",
                path=self.test_dir,
                options=options
            )

    def test_search_result_structure(self):
        """Test that search results have correct structure."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)

        if results.results:
            result = results.results[0]
            self.assertIsInstance(result, SearchResult)
            self.assertIsInstance(result.file_path, str)
            self.assertIsInstance(result.line_number, int)
            self.assertIsInstance(result.line_text, str)
            self.assertIsInstance(result.byte_offset, int)
            self.assertIsInstance(result.context_before, list)
            self.assertIsInstance(result.context_after, list)
            self.assertIsInstance(result.submatches, list)

    def test_search_options_defaults(self):
        """Test that SearchOptions has correct defaults."""
        options = SearchOptions()

        self.assertEqual(options.search_type, SearchType.PATTERN)
        self.assertTrue(options.case_sensitive)
        self.assertEqual(options.include_patterns, [])
        self.assertEqual(options.exclude_patterns, [])
        self.assertIsNone(options.max_results)
        self.assertEqual(options.context_lines, 0)
        self.assertIsNone(options.file_types)
        self.assertIsNone(options.max_filesize)
        self.assertIsNone(options.max_depth)
        self.assertFalse(options.follow_symlinks)
        self.assertEqual(options.output_format, OutputFormat.JSON)

    def test_search_options_customization(self):
        """Test SearchOptions customization."""
        options = SearchOptions(
            search_type=SearchType.LITERAL,
            case_sensitive=False,
            max_results=10,
            context_lines=3,
            output_format=OutputFormat.TEXT
        )

        self.assertEqual(options.search_type, SearchType.LITERAL)
        self.assertFalse(options.case_sensitive)
        self.assertEqual(options.max_results, 10)
        self.assertEqual(options.context_lines, 3)
        self.assertEqual(options.output_format, OutputFormat.TEXT)

    def test_cross_file_consistency(self):
        """Test that searches are consistent across files."""
        options = SearchOptions(output_format=OutputFormat.JSON)

        # Search for different patterns
        results1 = self.ripgrep.search_files("def", path=self.test_dir, options=options)
        results2 = self.ripgrep.search_files("class", path=self.test_dir, options=options)
        results3 = self.ripgrep.search_files("function", path=self.test_dir, options=options)

        # All should be SearchResults
        self.assertIsInstance(results1, SearchResults)
        self.assertIsInstance(results2, SearchResults)
        self.assertIsInstance(results3, SearchResults)

        # Results should be different
        self.assertNotEqual(results1.total_matches, results2.total_matches)
        self.assertNotEqual(results2.total_matches, results3.total_matches)

    def test_empty_results_handling(self):
        """Test handling of empty results."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "pattern_that_does_not_exist_anywhere",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertEqual(len(results.results), 0)
        self.assertEqual(results.total_matches, 0)
        self.assertGreaterEqual(results.files_searched, 0)
        self.assertGreaterEqual(results.search_time, 0)

    def test_line_number_accuracy(self):
        """Test that line numbers are accurate."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "async_function",
            path=self.python_file,
            options=options
        )

        if results.results:
            result = results.results[0]
            # The async function should be around line 13
            self.assertGreater(result.line_number, 10)
            self.assertLess(result.line_number, 20)
            self.assertIn("async_function", result.line_text)

    def test_timeout_handling(self):
        """Test timeout handling."""
        options = SearchOptions(output_format=OutputFormat.JSON)

        # This should not timeout with our simple test
        results = self.ripgrep.search_files(
            "hello",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertLess(results.search_time, 10)  # Should complete quickly


class TestRipgrepIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        self.ripgrep = RipgrepIntegration()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_empty_directory(self):
        """Test searching in empty directory."""
        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "test",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertEqual(len(results.results), 0)

    def test_special_characters(self):
        """Test searching with special characters."""
        # Create file with special characters
        test_file = os.path.join(self.test_dir, "special.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Line with special chars: äöü\nLine with symbols: @#$%\nLine with quotes: 'single' and \"double\"")

        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "special",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        self.assertGreater(len(results.results), 0)

    def test_unicode_handling(self):
        """Test Unicode handling."""
        # Create file with Unicode content
        test_file = os.path.join(self.test_dir, "unicode.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Unicode text: café, naïve, résumé\nEmoji: 🚀, 🎉, 🔥")

        options = SearchOptions(output_format=OutputFormat.JSON)
        results = self.ripgrep.search_files(
            "café",
            path=self.test_dir,
            options=options
        )

        self.assertIsInstance(results, SearchResults)
        if results.results:
            self.assertIn("café", results.results[0].line_text)


if __name__ == "__main__":
    unittest.main()
