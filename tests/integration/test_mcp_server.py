#!/usr/bin/env python3
"""
Test script for Fast Apply MCP Server

This script tests the MCP server functionality by directly calling the tools
and verifying the Fast Apply code editing capabilities.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to sys.path to import fastapply
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import FastApplyConnector, call_tool, list_tools


class TestFastApplyMCP:
    """Test suite for Fast Apply MCP server functionality."""

    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = {}

        # Create test files
        self._create_test_files()

    def teardown_method(self):
        """Cleanup after each test."""
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_files(self):
        """Create test files in temp directory."""
        # Python test file
        python_content = """def hello_world():
    print("Hello, World!")
    return "hello"

def add_numbers(a, b):
    return a + b
"""
        python_file = os.path.join(self.temp_dir, "test.py")
        with open(python_file, "w") as f:
            f.write(python_content)
        self.test_files["python"] = python_file

        # JavaScript test file
        js_content = """function greetUser(name) {
    console.log("Hello, " + name);
    return name;
}

function calculateSum(x, y) {
    return x + y;
}
"""
        js_file = os.path.join(self.temp_dir, "test.js")
        with open(js_file, "w") as f:
            f.write(js_content)
        self.test_files["javascript"] = js_file

    def test_fast_apply_connector_initialization(self):
        """Test FastApplyConnector initialization."""
        connector = FastApplyConnector()

        assert connector.url == os.getenv("FAST_APPLY_URL", "http://localhost:1234/v1")
        assert connector.model == os.getenv("FAST_APPLY_MODEL", "fastapply-1.5b")
        assert connector.timeout == float(os.getenv("FAST_APPLY_TIMEOUT", "30.0"))
        assert connector.max_tokens == int(os.getenv("FAST_APPLY_MAX_TOKENS", "8000"))
        assert connector.temperature == float(os.getenv("FAST_APPLY_TEMPERATURE", "0.05"))

    def test_markdown_stripping(self):
        """Test markdown code block stripping functionality."""
        connector = FastApplyConnector()

        # Test various markdown patterns
        test_cases = [
            ("```python\nprint('hello')\n```", "print('hello')"),
            ("```\nsome code\n```", "some code"),
            ("```javascript\nconsole.log('test');\n```", "console.log('test');"),
            ("plain text", "plain text"),
            ("```inline code```", "inline code"),
        ]

        for input_text, expected in test_cases:
            result = connector._strip_markdown_blocks(input_text)
            assert result == expected, f"Failed for input: {input_text}"

    @patch("fastapply.main.openai.OpenAI")
    def test_apply_edit_success(self, mock_openai):
        """Test successful code edit application."""
        # Mock the OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "def hello_world():\n    print('Hello, Modified World!')\n    return 'modified'"
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        connector = FastApplyConnector()

        original_code = "def hello_world():\n    print('Hello, World!')\n    return 'hello'"
        instructions = "Change the greeting message"
        code_edit = "print('Hello, Modified World!')"

        result = connector.apply_edit(instructions, original_code, code_edit)

        assert "Hello, Modified World!" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch("fastapply.main.openai.OpenAI")
    def test_apply_edit_with_markdown_response(self, mock_openai):
        """Test code edit with markdown-wrapped response."""
        # Mock response with markdown code blocks
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "```python\ndef hello_world():\n    print('Hello, Clean!')\n    return 'clean'\n```"
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        connector = FastApplyConnector()

        result = connector.apply_edit("test", "original", "edit")

        # Should strip markdown blocks
        assert result == "def hello_world():\n    print('Hello, Clean!')\n    return 'clean'"

    def test_mcp_list_tools(self):
        """Test MCP server tool listing."""
        # Already imported at top

        tools = list_tools()

        assert isinstance(tools, list)
        names = {t["name"] for t in tools}
        # Expect all unified tools
        assert {"edit_file", "dry_run_edit_file", "search_files", "read_multiple_files"}.issubset(names)
        # Basic schema presence
        for t in tools:
            assert "description" in t and "inputSchema" in t

    @patch("fastapply.main.fast_apply_connector")
    def test_mcp_call_tool_success(self, mock_connector):
        """Test successful MCP tool call."""
        # Already imported at top

        # Mock the connector
        mock_connector.apply_edit.return_value = "modified code content"

        # Create a test file
        test_file = os.path.join(self.temp_dir, "edit_test.py")
        with open(test_file, "w") as f:
            f.write("original content")

        # Change working directory to temp dir for relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            result = asyncio.run(call_tool("edit_file", {
                "target_file": "edit_test.py",
                "instructions": "Test edit",
                "code_edit": "new content"
            }))

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["type"] == "text"
            assert "Successfully applied edit" in result[0]["text"]

            # Verify file was modified
            with open(test_file, "r") as f:
                content = f.read()
            assert content == "modified code content"

            # Verify connector was called correctly with new unified signature
            assert mock_connector.apply_edit.call_count == 1
            _args, _kwargs = mock_connector.apply_edit.call_args
            assert _kwargs["original_code"] == "original content"
            assert _kwargs["code_edit"] == "new content"
            assert _kwargs.get("instruction") == "Test edit"
            assert "file_path" in _kwargs
        finally:
            os.chdir(original_cwd)

    def test_mcp_call_tool_invalid_parameters(self):
        """Test MCP tool call with invalid parameters."""
        # Already imported at top

        # Test missing parameters
        with pytest.raises(ValueError, match="target_file parameter is required"):
            asyncio.run(call_tool("edit_file", {"instructions": "test", "code_edit": "test"}))

        with pytest.raises(ValueError, match="instructions parameter is required"):
            asyncio.run(call_tool("edit_file", {"target_file": "test.py", "code_edit": "test"}))

        with pytest.raises(ValueError, match="code_edit parameter is required"):
            asyncio.run(call_tool("edit_file", {"target_file": "test.py", "instructions": "test"}))

    def test_mcp_call_tool_security_path_validation(self):
        """Test path validation security measures."""
        # Already imported at top

        # Test directory traversal attempts
        with pytest.raises(ValueError, match="Invalid file path"):
            asyncio.run(call_tool("edit_file", {"target_file": "../../../etc/passwd", "instructions": "hack", "code_edit": "malicious"}))

        with pytest.raises(ValueError, match="Invalid file path"):
            asyncio.run(call_tool("edit_file", {"target_file": "/absolute/path/file.py", "instructions": "hack", "code_edit": "malicious"}))

    def test_mcp_call_tool_file_not_found(self):
        """Test MCP tool call with non-existent file."""
        # Already imported at top

        with pytest.raises(ValueError, match="File not found"):
            asyncio.run(call_tool("edit_file", {"target_file": "non_existent_file.py", "instructions": "test", "code_edit": "test"}))

    def test_mcp_call_tool_unknown_tool(self):
        """Test MCP tool call with unknown tool name."""
        # Already imported at top

        with pytest.raises(ValueError, match=r"Unknown tool: unknown_tool"):
            asyncio.run(call_tool("unknown_tool", {"param": "value"}))

    @patch("fastapply.main.openai.OpenAI")
    def test_connector_api_error_handling(self, mock_openai):
        """Test API error handling in FastApplyConnector."""
        import openai

        # Mock API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APIError("API Error")
        mock_openai.return_value = mock_client

        connector = FastApplyConnector()

        with pytest.raises(RuntimeError, match="Fast Apply API error"):
            connector.apply_edit("test", "original", "edit")

    @patch("fastapply.main.openai.OpenAI")
    def test_connector_empty_response(self, mock_openai):
        """Test handling of empty API response."""
        # Mock empty response
        mock_response = Mock()
        mock_response.choices = []

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        connector = FastApplyConnector()

        with pytest.raises(ValueError, match="Invalid Fast Apply API response"):
            connector.apply_edit("test", "original", "edit")

    def test_connector_config_update(self):
        """Test configuration update functionality."""
        connector = FastApplyConnector()

        new_config = connector.update_config(
            url="http://new-url:8080/v1", model="new-model", timeout=60.0, max_tokens=4000, temperature=0.1
        )

        assert new_config["url"] == "http://new-url:8080/v1"
        assert new_config["model"] == "new-model"
        assert new_config["timeout"] == 60.0
        assert new_config["max_tokens"] == 4000
        assert new_config["temperature"] == 0.1

        # Verify the connector was updated
        assert connector.url == "http://new-url:8080/v1"
        assert connector.model == "new-model"
        assert connector.timeout == 60.0
        assert connector.max_tokens == 4000
        assert connector.temperature == 0.1


def run_integration_tests():
    """Run integration tests that require a running Fast Apply server."""
    print("🚀 Running Fast Apply MCP Integration Tests")

    # Test file paths
    test_files = [
        "tests/sample_files/calculator.py",
        "tests/sample_files/user-manager.js",
        "tests/sample_files/TaskManager.java",
        "tests/sample_files/inventory.go",
    ]

    test_scenarios = [
        {
            "file": "tests/sample_files/calculator.py",
            "instructions": "Add a power method to the Calculator class",
            "code_edit": """
    def power(self, a, b):
        result = a ** b
        self.history.append(f"{a} ** {b} = {result}")
        return result
""",
            "expected_content": "def power(self, a, b):",
        },
        {
            "file": "tests/sample_files/user-manager.js",
            "instructions": "Add a method to count active users",
            "code_edit": """
    countActiveUsers() {
        return this.getActiveUsers().length;
    }
""",
            "expected_content": "countActiveUsers()",
        },
        {
            "file": "tests/sample_files/inventory.go",
            "instructions": "Add a method to get products by category",
            "code_edit": """
// GetProductsByCategory returns all products in a specific category
func (im *InventoryManager) GetProductsByCategory(category string) []*Product {
	var products []*Product
	for _, product := range im.products {
		if product.Category == category {
			products = append(products, product)
		}
	}
	return products
}
""",
            "expected_content": "GetProductsByCategory",
        },
    ]

    print(f"📁 Found {len(test_files)} test files")

    # Import the call_tool function
    # Already imported at top

    results = []

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Test {i}: {scenario['file']}")
        print(f"📝 Instructions: {scenario['instructions']}")

        try:
            # Read original content
            with open(scenario["file"], "r") as f:
                original_content = f.read()

            # Apply edit using MCP tool
            result = asyncio.run(call_tool(
                "edit_file", {"target_file": scenario["file"], "instructions": scenario["instructions"], "code_edit": scenario["code_edit"]}
            ))

            # Read modified content
            with open(scenario["file"], "r") as f:
                modified_content = f.read()

            # Check if expected content is present
            success = scenario["expected_content"] in modified_content

            results.append(
                {
                    "test": i,
                    "file": scenario["file"],
                    "success": success,
                    "result": result[0]["text"] if result else "No result",
                    "original_length": len(original_content),
                    "modified_length": len(modified_content),
                }
            )

            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} - Expected content {'found' if success else 'not found'}")
            print(f"📏 Length: {len(original_content)} → {len(modified_content)}")

            # Restore original content for next test
            with open(scenario["file"], "w") as f:
                f.write(original_content)

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({"test": i, "file": scenario["file"], "success": False, "error": str(e)})

    # Print summary
    print("\n📊 Test Summary")
    print(f"{'=' * 50}")
    passed = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    for result in results:
        status = "✅" if result.get("success", False) else "❌"
        print(f"{status} Test {result['test']}: {Path(result['file']).name}")
        if "error" in result:
            print(f"   Error: {result['error']}")

    return results


if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running unit tests...")
    import subprocess

    result = subprocess.run(["python", "-m", "pytest", __file__, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print("\n" + "=" * 60)

    # Run integration tests
    print("🔧 Running integration tests...")
    print("⚠️  Note: These require a running Fast Apply server")
    integration_results = run_integration_tests()
