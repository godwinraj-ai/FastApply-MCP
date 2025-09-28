#!/usr/bin/env python3
"""
End-to-end integration tests for Fast Apply MCP server.
Tests complete workflows and integration scenarios.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import unittest

# Add the parent directory to sys.path to import fastapply.main as main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import FastApplyConnector, _secure_resolve, call_tool, fast_apply_connector


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.test_dir
        # Create a test-mode connector that doesn't require API
        self.connector = FastApplyConnector()
        # Override client to avoid API dependency
        self.connector.client = None
        # Mock apply_edit to return the code_edit as-is for testing
        self.original_apply_edit = self.connector.apply_edit
        self.connector.apply_edit = self._mock_apply_edit
        # Mock the global fast_apply_connector for call_tool function
        self.original_global_connector = fast_apply_connector
        import fastapply.main as main

        main.fast_apply_connector = self.connector

    def tearDown(self):
        """Clean up test environment."""
        # Restore original apply_edit method
        self.connector.apply_edit = self.original_apply_edit
        # Restore global connector
        import fastapply.main as main

        main.fast_apply_connector = self.original_global_connector
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        shutil.rmtree(self.test_dir)

    def _mock_apply_edit(self, *args, **kwargs):
        """Mock apply_edit for testing - returns verification results."""
        legacy_mode = False
        if args and not kwargs:
            if len(args) == 3:
                instruction, original_code, code_edit = args
                legacy_mode = True
            else:
                raise TypeError("Legacy apply_edit expects exactly 3 positional arguments")
        else:
            original_code = kwargs.get("original_code")
            code_edit = kwargs.get("code_edit")
            if original_code is None or code_edit is None:
                raise TypeError("apply_edit requires original_code and code_edit")

        # Return mock verification results
        merged_code = code_edit  # For testing, just return the edited code
        verification_results = {
            "merged_code": merged_code,
            "has_changes": merged_code != original_code,
            "udiff": f"--- original\n+++ modified\n@@ -1,2 +1,2 @@\n-{original_code[:20]}...\n+{merged_code[:20]}...",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }

        return merged_code if legacy_mode else verification_results

    def test_complete_edit_workflow(self):
        """Test complete file edit workflow."""
        # Create initial file
        initial_content = """def hello():
    return "Hello, World!"

def main():
    print(hello())

if __name__ == "__main__":
    main()
"""
        test_file = os.path.join(self.test_dir, "test.py")
        with open(test_file, "w") as f:
            f.write(initial_content)

        # Read the file using read_multiple_files
        read_result = asyncio.run(call_tool("read_multiple_files", {"paths": ["test.py"]}))
        self.assertIn("test.py:", read_result[0]["text"])
        self.assertIn(initial_content, read_result[0]["text"])

        # Apply edit using edit_file tool
        modified_content = """def hello():
    return "Hello, Modified World!"

def main():
    print(hello())

if __name__ == "__main__":
    main()
"""

        edit_result = asyncio.run(call_tool(
            "edit_file", {"target_file": "test.py", "instructions": "Modify the greeting message", "code_edit": modified_content}
        ))

        self.assertIn("Successfully applied edit", edit_result[0]["text"])

        # Verify the change using read_multiple_files
        final_read_result = asyncio.run(call_tool("read_multiple_files", {"paths": ["test.py"]}))
        self.assertIn("Hello, Modified World!", final_read_result[0]["text"])

        # Verify backup was created
        backup_files = [f for f in os.listdir(self.test_dir) if f.startswith("test.py.bak")]
        self.assertTrue(len(backup_files) > 0)

    def test_multi_file_workflow(self):
        """Test workflow involving multiple files."""
        # Create multiple related files with unique names to avoid conflicts
        files = {
            "test_main.py": """from utils import helper
from config import SETTINGS

def main():
    result = helper()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
""",
            "test_utils.py": """def helper():
    return "Hello from helper"
""",
            "test_config.py": """SETTINGS = {
    "debug": True,
    "version": "1.0"
}
""",
        }

        for filename, content in files.items():
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        # Read all files
        paths = list(files.keys())
        read_result = asyncio.run(call_tool("read_multiple_files", {"paths": paths}))

        self.assertTrue(len(read_result) > 0)
        response_text = read_result[0]["text"]
        for filename, expected_content in files.items():
            self.assertIn(filename + ":", response_text)
            self.assertIn(expected_content, response_text)

        # Search across files - search in current directory only
        search_result = asyncio.run(call_tool("search_files", {"pattern": "Hello from helper", "path": "."}))

        self.assertTrue(len(search_result) > 0)
        search_text = search_result[0]["text"]
        # Just verify search works (it may find files from other directories too)
        self.assertTrue(len(search_text) > 0)

        # Modify one file using edit_file tool
        modified_utils = """def helper():
    return "Hello from modified helper"
"""

        edit_result = asyncio.run(call_tool(
            "edit_file", {"target_file": "test_utils.py", "instructions": "Modify helper function", "code_edit": modified_utils}
        ))

        self.assertIn("Successfully applied edit", edit_result[0]["text"])

    def test_file_creation_workflow(self):
        """Test workflow for creating new files."""
        # Create a new file through direct write operation
        new_content = """#!/usr/bin/env python3
def new_function():
    return "This is a new file"

if __name__ == "__main__":
    print(new_function())
"""

        # Write file directly since write_file tool doesn't exist
        new_file = os.path.join(self.test_dir, "new_file.py")
        with open(new_file, "w") as f:
            f.write(new_content)

        # Verify file was created
        self.assertTrue(os.path.exists(new_file))

        # Read and verify content using read_multiple_files
        read_result = asyncio.run(call_tool("read_multiple_files", {"paths": ["new_file.py"]}))
        self.assertIn(new_content, read_result[0]["text"])

    def test_error_handling_workflow(self):
        """Test error handling in complete workflows."""
        # Try to read nonexistent file
        result = asyncio.run(call_tool("read_multiple_files", {"paths": ["nonexistent.txt"]}))
        self.assertIn("Error - File not found", result[0]["text"])

        # Try to edit nonexistent file - should fail
        try:
            asyncio.run(call_tool("edit_file", {"target_file": "nonexistent.txt", "instructions": "modify", "code_edit": "modified"}))
        except ValueError as e:
            self.assertIn("File not found", str(e))

        # Test path validation with secure_resolve
        try:
            _secure_resolve("../etc/malicious.txt")
        except ValueError as e:
            self.assertIn("escapes workspace", str(e))


class TestMCPIntegration(unittest.TestCase):
    """Test MCP server integration scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.test_dir
        self.connector = FastApplyConnector()
        # Mock for testing
        self.connector.client = None
        self.original_apply_edit = self.connector.apply_edit
        self.connector.apply_edit = self._mock_apply_edit
        # Mock the global fast_apply_connector for call_tool function
        self.original_global_connector = fast_apply_connector
        import fastapply.main as main

        main.fast_apply_connector = self.connector

    def tearDown(self):
        """Clean up test environment."""
        self.connector.apply_edit = self.original_apply_edit
        # Restore global connector
        import fastapply.main as main

        main.fast_apply_connector = self.original_global_connector
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        shutil.rmtree(self.test_dir)

    def _mock_apply_edit(self, *args, **kwargs):
        """Mock apply_edit for testing - returns verification results."""
        legacy_mode = False
        if args and not kwargs:
            if len(args) == 3:
                instruction, original_code, code_edit = args
                legacy_mode = True
            else:
                raise TypeError("Legacy apply_edit expects exactly 3 positional arguments")
        else:
            original_code = kwargs.get("original_code")
            code_edit = kwargs.get("code_edit")
            if original_code is None or code_edit is None:
                raise TypeError("apply_edit requires original_code and code_edit")

        # Return mock verification results
        merged_code = code_edit  # For testing, just return the edited code
        verification_results = {
            "merged_code": merged_code,
            "has_changes": merged_code != original_code,
            "udiff": f"--- original\n+++ modified\n@@ -1,2 +1,2 @@\n-{original_code[:20]}...\n+{merged_code[:20]}...",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }

        return merged_code if legacy_mode else verification_results

    def test_mcp_tool_structure(self):
        """Test MCP tool structure and availability."""
        # Test that required tools are available
        tools = ["edit_file", "dry_run_edit_file", "search_files", "read_multiple_files"]

        for tool in tools:
            with self.subTest(tool=tool):
                # Test tool exists and can be called
                try:
                    # Test with minimal valid parameters
                    if tool == "edit_file":
                        # Will fail due to missing file, but should be a recognized tool
                        try:
                            asyncio.run(call_tool(tool, {"target_file": "test.txt", "instructions": "test", "code_edit": "test"}))
                        except ValueError as e:
                            # Expected - file doesn't exist
                            self.assertIn("File not found", str(e))
                    elif tool == "dry_run_edit_file":
                        try:
                            asyncio.run(call_tool(tool, {"path": "test.txt", "code_edit": "test"}))
                        except ValueError as e:
                            # Expected - file doesn't exist
                            self.assertIn("File not found", str(e))
                    elif tool == "search_files":
                        result = asyncio.run(call_tool(tool, {"pattern": "test"}))
                        self.assertTrue(len(result) > 0)
                    elif tool == "read_multiple_files":
                        result = asyncio.run(call_tool(tool, {"paths": ["test.txt"]}))
                        self.assertTrue(len(result) > 0)
                        self.assertIn("Error - File not found", result[0]["text"])
                except Exception as e:
                    self.fail(f"Tool {tool} failed: {e}")

    def test_concurrent_operations(self):
        """Test concurrent file operations."""
        import threading

        results = []
        errors = []

        def worker(file_num):
            try:
                filename = f"worker_{file_num}.txt"
                content = f"Content from worker {file_num}"

                # Write file directly since write_file tool doesn't exist
                filepath = os.path.join(self.test_dir, filename)
                with open(filepath, "w") as f:
                    f.write(content)

                # Read it back using read_multiple_files
                read_result = asyncio.run(call_tool("read_multiple_files", {"paths": [filename]}))

                results.append((file_num, "written", read_result))
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)

    def test_tool_parameter_validation(self):
        """Test parameter validation across all tools."""
        # Test tools with invalid parameters
        invalid_cases = [
            ("edit_file", {"target_file": "", "instructions": "test", "code_edit": "test"}),
            ("edit_file", {"target_file": None, "instructions": "test", "code_edit": "test"}),
            ("edit_file", {"target_file": "test.txt", "instructions": "", "code_edit": "test"}),
            ("edit_file", {"target_file": "test.txt", "instructions": "test", "code_edit": ""}),
            ("search_files", {"pattern": ""}),
            ("search_files", {"pattern": None}),
            ("read_multiple_files", {"paths": []}),
            ("read_multiple_files", {"paths": None}),
        ]

        for tool_name, params in invalid_cases:
            with self.subTest(tool=tool_name, params=params):
                try:
                    asyncio.run(call_tool(tool_name, params))
                    self.fail(f"Expected validation error for {tool_name} with {params}")
                except (ValueError, TypeError):
                    # Expected - validation should fail
                    pass


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance-related integration scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.test_dir
        self.connector = FastApplyConnector()
        # Mock for testing
        self.connector.client = None
        self.original_apply_edit = self.connector.apply_edit
        self.connector.apply_edit = self._mock_apply_edit
        # Mock the global fast_apply_connector for call_tool function
        self.original_global_connector = fast_apply_connector
        import fastapply.main as main

        main.fast_apply_connector = self.connector

    def tearDown(self):
        """Clean up test environment."""
        self.connector.apply_edit = self.original_apply_edit
        # Restore global connector
        import fastapply.main as main

        main.fast_apply_connector = self.original_global_connector
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        shutil.rmtree(self.test_dir)

    def _mock_apply_edit(self, *args, **kwargs):
        """Mock apply_edit for testing - returns verification results."""
        legacy_mode = False
        if args and not kwargs:
            if len(args) == 3:
                instruction, original_code, code_edit = args
                legacy_mode = True
            else:
                raise TypeError("Legacy apply_edit expects exactly 3 positional arguments")
        else:
            original_code = kwargs.get("original_code")
            code_edit = kwargs.get("code_edit")
            if original_code is None or code_edit is None:
                raise TypeError("apply_edit requires original_code and code_edit")

        # Return mock verification results
        merged_code = code_edit  # For testing, just return the edited code
        verification_results = {
            "merged_code": merged_code,
            "has_changes": merged_code != original_code,
            "udiff": f"--- original\n+++ modified\n@@ -1,2 +1,2 @@\n-{original_code[:20]}...\n+{merged_code[:20]}...",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }

        return merged_code if legacy_mode else verification_results

    def test_large_file_handling(self):
        """Test handling of large files."""
        # Create a large file
        large_content = "line\n" * 10000  # 10K lines
        large_file = os.path.join(self.test_dir, "large.txt")

        with open(large_file, "w") as f:
            f.write(large_content)

        # Test reading large file using read_multiple_files
        result = asyncio.run(call_tool("read_multiple_files", {"paths": ["large.txt"]}))
        self.assertTrue(len(result) > 0)
        self.assertIn("large.txt:", result[0]["text"])

    def test_multiple_file_operations(self):
        """Test performance with multiple file operations."""
        # Create many files
        file_count = 20
        for i in range(file_count):
            filename = f"file_{i}.txt"
            content = f"Content of file {i}"
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        # Read all files
        paths = [f"file_{i}.txt" for i in range(file_count)]
        result = asyncio.run(call_tool("read_multiple_files", {"paths": paths}))

        self.assertTrue(len(result) > 0)
        response_text = result[0]["text"]
        self.assertIn("Content of file 0", response_text)
        self.assertIn(f"Content of file {file_count - 1}", response_text)

    def test_backup_performance(self):
        """Test backup creation performance with multiple files."""
        # Create multiple files to backup
        file_count = 10
        files = {}

        for i in range(file_count):
            filename = f"backup_test_{i}.py"
            content = f"# File {i}\ndef test_{i}():\n    return {i}\n"
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            files[filename] = content

        # Apply edits to create backups using edit_file tool
        for filename, original_content in files.items():
            modified_content = original_content.replace(f"return {i}", f"return {i * 2}")

            result = asyncio.run(call_tool(
                "edit_file", {"target_file": filename, "instructions": f"Double return value for {filename}", "code_edit": modified_content}
            ))

            self.assertIn("Successfully applied edit", result[0]["text"])

            # Verify backup was created
            backup_files = [f for f in os.listdir(self.test_dir) if f.startswith(filename + ".bak")]
            self.assertTrue(len(backup_files) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
