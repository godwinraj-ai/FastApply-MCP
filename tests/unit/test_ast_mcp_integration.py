"""
Unit tests for AST search MCP tool integration.

Tests the integration of semantic search tools with the MCP server.
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Import the main module and tools
from fastapply import main
from fastapply.ast_search import (
    AstSearchError,
    PatternSearchResult,
    StructureInfo,
)


class TestAstSearchMCPIntegration(unittest.TestCase):
    """Test AST search tools integration with MCP server."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files
        self.python_file = os.path.join(self.test_dir, "sample.py")
        with open(self.python_file, "w", encoding="utf-8") as f:
            f.write("""
def calculate_sum(a, b):
    return a + b

class MathUtils:
    def __init__(self):
        self.pi = 3.14159

    def area_circle(self, radius):
        return self.pi * radius * radius

import math
from typing import List
""")

        self.js_file = os.path.join(self.test_dir, "sample.js")
        with open(self.js_file, "w", encoding="utf-8") as f:
            f.write("""
function multiply(x, y) {
    return x * y;
}

const divide = (a, b) => {
    if (b === 0) throw new Error('Division by zero');
    return a / b;
};

class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }
}

export { multiply, divide };
export default Counter;
""")

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_ast_search_not_available(self):
        """Test tool behavior when AST search is not available."""
        async def test_async():
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "def $name($args)",
                    "language": "python",
                    "path": "."
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    def test_search_code_patterns_missing_parameters(self):
        """Test search_code_patterns with missing required parameters."""
        async def test_async():
            # Missing pattern
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "language": "python",
                    "path": "."
                })
            self.assertIn("pattern parameter is required", str(cm.exception))

            # Missing language
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "def $name($args)",
                    "path": "."
                })
            self.assertIn("language parameter is required", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.search_code_patterns")
    def test_search_code_patterns_error_handling(self, mock_search):
        """Test search_code_patterns error handling when functionality is available."""
        async def test_async():
            # Mock error response
            mock_search.side_effect = AstSearchError("Pattern search failed")

            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "invalid pattern",
                    "language": "python",
                    "path": self.test_dir
                })
            self.assertIn("Pattern search failed", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_search_code_patterns_unavailable(self):
        """Test search_code_patterns when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "def $name($args)",
                    "language": "python",
                    "path": self.test_dir
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.search_code_patterns")
    def test_search_code_patterns_success(self, mock_search):
        """Test search_code_patterns when functionality is available."""
        async def test_async():
            # Mock successful response
            mock_search.return_value = [
                PatternSearchResult(
                    file_path="sample.py",
                    line=37,
                    column=0,
                    text="def calculate_sum(a, b):",
                    matches={}
                )
            ]

            result = await main.call_tool("search_code_patterns", {
                "pattern": "def $NAME($ARGS)",
                "language": "python",
                "path": self.test_dir
            })

            # Parse the JSON response from the text content
            response_text = result[0]["text"]
            response_data = json.loads(response_text)

            self.assertIn("matches", response_data)
            self.assertEqual(len(response_data["matches"]), 1)
            self.assertEqual(response_data["matches"][0]["file_path"], "sample.py")

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.search_code_patterns")
    def test_search_code_patterns_no_results(self, mock_search):
        """Test search_code_patterns when no results are found."""
        async def test_async():
            # Mock empty response
            mock_search.return_value = []

            result = await main.call_tool("search_code_patterns", {
                "pattern": "class NonExistent",
                "language": "python",
                "path": self.test_dir
            })

            response_text = result[0]["text"]
            # Handle both JSON responses and plain text responses
            try:
                response_data = json.loads(response_text)
                self.assertIn("matches", response_data)
                self.assertEqual(len(response_data["matches"]), 0)
            except json.JSONDecodeError:
                # Plain text response
                self.assertIn("No code patterns found", response_text)

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_search_code_patterns_no_results_unavailable(self):
        """Test search_code_patterns when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "class NonExistent",
                    "language": "python",
                    "path": self.test_dir
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_search_code_patterns_error_handling_unavailable(self):
        """Test search_code_patterns when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("search_code_patterns", {
                    "pattern": "invalid pattern",
                    "language": "python",
                    "path": self.test_dir
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    def test_analyze_code_structure_missing_parameters(self):
        """Test analyze_code_structure with missing parameters."""
        async def test_async():
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("analyze_code_structure", {})

            self.assertIn("file_path parameter is required", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    def test_analyze_code_structure_file_not_found(self):
        """Test analyze_code_structure with non-existent file."""
        async def test_async():
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("analyze_code_structure", {
                    "file_path": "nonexistent.py"
                })

            self.assertIn("File not found", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_analyze_code_structure_unavailable(self):
        """Test analyze_code_structure when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("analyze_code_structure", {
                    "file_path": os.path.relpath(self.python_file, self.test_dir)
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.analyze_code_structure")
    def test_analyze_code_structure_success(self, mock_analyze):
        """Test analyze_code_structure when functionality is available."""
        async def test_async():
            # Mock successful response
            mock_result = StructureInfo(
                file_path="sample.py",
                language="python"
            )
            mock_result.functions = ["calculate_sum"]
            mock_result.classes = ["MathUtils"]
            mock_result.imports = ["math", "typing"]
            mock_analyze.return_value = mock_result

            result = await main.call_tool("analyze_code_structure", {
                "file_path": os.path.relpath(self.python_file, self.test_dir)
            })

            response_text = result[0]["text"]
            response_data = json.loads(response_text)

            self.assertIn("structure", response_data)
            self.assertIn("functions", response_data["structure"])
            self.assertIn("classes", response_data["structure"])
            self.assertEqual(len(response_data["structure"]["functions"]), 1)

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.analyze_code_structure")
    def test_analyze_code_structure_error_handling(self, mock_analyze):
        """Test analyze_code_structure error handling when functionality is available."""
        async def test_async():
            # Mock error response
            mock_analyze.side_effect = AstSearchError("Analysis failed")

            with self.assertRaises(ValueError) as cm:
                await main.call_tool("analyze_code_structure", {
                    "file_path": os.path.relpath(self.python_file, self.test_dir)
                })
            self.assertIn("Analysis failed", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_analyze_code_structure_error_handling_unavailable(self):
        """Test analyze_code_structure when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("analyze_code_structure", {
                    "file_path": os.path.relpath(self.python_file, self.test_dir)
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    def test_find_references_missing_parameters(self):
        """Test find_references with missing parameters."""
        async def test_async():
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "path": "."
                })

            self.assertIn("symbol parameter is required", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_find_references_unavailable(self):
        """Test find_references when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "symbol": "calculate_sum",
                    "path": self.test_dir,
                    "symbol_type": "function"
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_find_references_no_results_unavailable(self):
        """Test find_references when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "symbol": "nonexistent_function",
                    "path": self.test_dir
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.find_references")
    def test_find_references_success(self, mock_find):
        """Test find_references when functionality is available."""
        async def test_async():
            # Mock successful response
            mock_find.return_value = [
                PatternSearchResult(
                    file_path="sample.py",
                    line=45,
                    column=0,
                    text="self.pi * radius * radius",
                    matches={}
                )
            ]

            result = await main.call_tool("find_references", {
                "symbol": "pi",
                "path": self.test_dir,
                "symbol_type": "variable"
            })

            response_text = result[0]["text"]
            response_data = json.loads(response_text)

            self.assertIn("references", response_data)
            self.assertEqual(len(response_data["references"]), 1)
            self.assertEqual(response_data["references"][0]["file_path"], "sample.py")

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.find_references")
    def test_find_references_error_handling(self, mock_find):
        """Test find_references error handling when functionality is available."""
        async def test_async():
            # Mock error response
            mock_find.side_effect = AstSearchError("Reference search failed")

            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "symbol": "nonexistent_symbol",
                    "path": self.test_dir
                })
            self.assertIn("Reference search failed", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.find_references")
    def test_find_references_no_results(self, mock_find):
        """Test find_references when no results are found."""
        async def test_async():
            # Mock empty response
            mock_find.return_value = []

            result = await main.call_tool("find_references", {
                "symbol": "nonexistent_function",
                "path": self.test_dir
            })

            response_text = result[0]["text"]
            # Handle both JSON responses and plain text responses
            try:
                response_data = json.loads(response_text)
                self.assertIn("references", response_data)
                self.assertEqual(len(response_data["references"]), 0)
            except json.JSONDecodeError:
                # Plain text response
                self.assertIn("No references found", response_text)

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", True)
    @patch("fastapply.ast_search.find_references")
    def test_find_references_default_parameters(self, mock_find):
        """Test find_references with default parameters."""
        async def test_async():
            # This test would verify default parameter handling
            # Since AST functionality is mocked as available, it should check parameters
            mock_find.return_value = []

            result = await main.call_tool("find_references", {
                "symbol": "test_symbol"
                # path should default to "."
                # symbol_type should default to "any"
            })

            response_text = result[0]["text"]
            # Handle both JSON responses and plain text responses
            try:
                response_data = json.loads(response_text)
                self.assertIn("references", response_data)
                self.assertEqual(len(response_data["references"]), 0)
            except json.JSONDecodeError:
                # Plain text response
                self.assertIn("No references found", response_text)

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_find_references_default_parameters_unavailable(self):
        """Test find_references when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "symbol": "test_symbol"
                    # path should default to "."
                    # symbol_type should default to "any"
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())

    @patch("fastapply.main.AST_SEARCH_AVAILABLE", False)
    def test_find_references_error_handling_unavailable(self):
        """Test find_references when AST functionality is not available."""
        async def test_async():
            # Call the tool when AST functionality is not available
            with self.assertRaises(ValueError) as cm:
                await main.call_tool("find_references", {
                    "symbol": "test_symbol",
                    "path": self.test_dir
                })
            self.assertIn("AST search functionality not available", str(cm.exception))

        asyncio.run(test_async())


class TestAstSearchToolList(unittest.TestCase):
    """Test that AST search tools are properly registered."""

    def test_tools_in_list(self):
        """Test that AST search tools are included in the tool list."""
        from fastapply.main import list_tools

        tools = list_tools()
        tool_names = [tool["name"] for tool in tools]

        # Verify all new tools are registered
        self.assertIn("search_code_patterns", tool_names)
        self.assertIn("analyze_code_structure", tool_names)
        self.assertIn("find_references", tool_names)

    def test_tool_schemas(self):
        """Test that AST search tools have correct schemas."""
        from fastapply.main import list_tools

        tools = list_tools()
        tool_dict = {tool["name"]: tool for tool in tools}

        # Test search_code_patterns schema
        search_tool = tool_dict["search_code_patterns"]
        self.assertIn("pattern", search_tool["inputSchema"]["required"])
        self.assertIn("language", search_tool["inputSchema"]["required"])
        self.assertIn("path", search_tool["inputSchema"]["required"])

        # Test analyze_code_structure schema
        analyze_tool = tool_dict["analyze_code_structure"]
        self.assertIn("file_path", analyze_tool["inputSchema"]["required"])

        # Test find_references schema
        find_tool = tool_dict["find_references"]
        self.assertIn("symbol", find_tool["inputSchema"]["required"])
        self.assertIn("path", find_tool["inputSchema"]["required"])


if __name__ == "__main__":
    unittest.main()
