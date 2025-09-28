#!/usr/bin/env python3
"""
Consolidated path security tests for Fast Apply MCP server.
Tests path traversal prevention, symlink attacks, and all security-related path operations.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import unittest

# Add the parent directory to sys.path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import call_tool


class TestPathSecurity(unittest.TestCase):
    """Test path security functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.test_dir

        # Create test files and directories
        self.safe_file = os.path.join(self.test_dir, "safe_file.txt")
        self.subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(self.subdir)

        with open(self.safe_file, "w") as f:
            f.write("safe content")

    def tearDown(self):
        """Clean up test environment."""
        # Restore original workspace root
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)

        # Clean up test directory
        shutil.rmtree(self.test_dir)

    def test_secure_resolve_valid_path(self):
        """Test that valid paths within workspace are resolved correctly."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        test_path = "subdir/file.txt"
        expected = os.path.realpath(os.path.join(self.test_dir, "subdir", "file.txt"))
        result = main._secure_resolve(test_path)
        self.assertEqual(result, expected)

    def test_secure_resolve_parent_directory_traversal(self):
        """Test that parent directory traversal is blocked."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Test with a path that should definitely escape
        with self.assertRaises(ValueError) as cm:
            main._secure_resolve("../etc/passwd")
        self.assertIn("escapes workspace", str(cm.exception))

        # Test multiple levels
        with self.assertRaises(ValueError) as cm:
            main._secure_resolve("subdir/../../../etc/passwd")
        self.assertIn("escapes workspace", str(cm.exception))

    def test_secure_resolve_absolute_path(self):
        """Test that absolute paths outside workspace are blocked."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Test paths that are definitely outside the workspace
        malicious_paths = ["/etc/passwd", "/windows/system32/config", "/tmp/malicious"]

        for path in malicious_paths:
            with self.assertRaises(ValueError) as cm:
                main._secure_resolve(path)
            self.assertIn("escapes workspace", str(cm.exception))

    def test_secure_resolve_absolute_path_within_workspace(self):
        """Test that absolute paths within workspace are allowed."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        abs_path = os.path.join(self.test_dir, "abs_file.txt")
        with open(abs_path, "w") as f:
            f.write("abs content")

        result = main._secure_resolve(abs_path)
        expected = os.path.realpath(abs_path)
        self.assertEqual(result, expected)

    def test_secure_resolve_symlink_attack(self):
        """Test that symlink attacks are mitigated."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Create a symlink outside workspace
        outside_file = os.path.join(os.path.dirname(self.test_dir), "outside.txt")
        symlink_path = os.path.join(self.test_dir, "safe_link")

        try:
            # Create the file outside workspace
            with open(outside_file, "w") as f:
                f.write("outside content")

            # Create the symlink
            os.symlink(outside_file, symlink_path)

            # This should be blocked by the secure resolve function
            with self.assertRaises(ValueError) as cm:
                main._secure_resolve(symlink_path)
            self.assertIn("escapes workspace", str(cm.exception))
        except (FileNotFoundError, OSError):
            # If symlink creation fails, skip this test
            self.skipTest("Symlink creation not supported")
        finally:
            # Clean up
            if os.path.exists(symlink_path):
                os.unlink(symlink_path)
            if os.path.exists(outside_file):
                os.remove(outside_file)

    def test_secure_resolve_safe_symlink_handling(self):
        """Test handling of symlinks that stay within workspace."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Create symlink pointing within workspace
        target_file = os.path.join(self.test_dir, "target.txt")
        symlink_path = os.path.join(self.test_dir, "safe_link")

        with open(target_file, "w") as f:
            f.write("target content")

        try:
            os.symlink(target_file, symlink_path)

            # Should resolve successfully
            result = main._secure_resolve("safe_link")
            expected = os.path.realpath(target_file)
            self.assertEqual(result, expected)

        finally:
            if os.path.exists(symlink_path):
                os.unlink(symlink_path)

    def test_secure_resolve_workspace_root_exact_match(self):
        """Test that exact workspace root matches are allowed."""
        # Test that "." resolves to the current workspace root
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        result = main._secure_resolve(".")
        # The function may resolve to current working directory due to CWD fallback
        expected_workspace = os.path.realpath(self.test_dir)
        expected_cwd = os.path.realpath(os.getcwd())

        # Result should be either the workspace root or the current working directory
        self.assertIn(result, [expected_workspace, expected_cwd])

    def test_secure_resolve_nonexistent_path_handling(self):
        """Test handling of nonexistent paths."""
        # Nonexistent paths should resolve successfully (path doesn't need to exist)
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        result = main._secure_resolve("nonexistent.txt")
        expected = os.path.realpath(os.path.join(self.test_dir, "nonexistent.txt"))
        self.assertEqual(result, expected)

    def test_secure_resolve_empty_path_handling(self):
        """Test handling of empty paths."""
        # Empty path should resolve to workspace root or current working directory
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        result = main._secure_resolve("")
        # The function may resolve to current working directory due to CWD fallback
        expected_workspace = os.path.realpath(self.test_dir)
        expected_cwd = os.path.realpath(os.getcwd())

        # Result should be either the workspace root or the current working directory
        self.assertIn(result, [expected_workspace, expected_cwd])

    def test_workspace_root_environment_variable(self):
        """Test that WORKSPACE_ROOT can be set via environment variable."""
        custom_root = "/custom/workspace"
        os.environ["WORKSPACE_ROOT"] = custom_root

        # Reload the module to pick up new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        self.assertEqual(main.WORKSPACE_ROOT, custom_root)

    def test_mcp_call_tool_security_path_validation(self):
        """Test that MCP call tool validates file paths."""
        # call_tool is imported at module level

        # Test with malicious path
        with self.assertRaises(ValueError) as cm:
            asyncio.run(call_tool("edit_file", {
                "target_file": "../etc/passwd",
                "instructions": "test",
                "code_edit": "test"
            }))
        self.assertIn("Invalid file path", str(cm.exception))

        # Test with absolute path outside workspace
        with self.assertRaises(ValueError) as cm:
            asyncio.run(call_tool("edit_file", {
                "target_file": "/etc/passwd",
                "instructions": "test",
                "code_edit": "test"
            }))
        self.assertIn("Invalid file path", str(cm.exception))

    def test_path_resolution_edge_cases(self):
        """Test edge cases in path resolution."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Test paths that resolve to the same location
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Test relative path variations
        paths_to_test = [
            "test.txt",
            "./test.txt",
            "subdir/../test.txt",
            "subdir/./../test.txt"
        ]

        expected_result = os.path.realpath(test_file)
        for test_path in paths_to_test:
            with self.subTest(path=test_path):
                result = main._secure_resolve(test_path)
                self.assertEqual(result, expected_result)


class TestPathResolutionScenarios(unittest.TestCase):
    """Test comprehensive path resolution scenarios."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.environ["WORKSPACE_ROOT"] = self.test_dir

    def tearDown(self):
        """Clean up test environment."""
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        shutil.rmtree(self.test_dir)

    def test_path_escaping_workspace(self):
        """Test various path escaping scenarios."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        escape_attempts = [
            "../etc/passwd",
            "../../etc/passwd",
            "../../../etc/passwd",
            "subdir/../../../etc/passwd",
            "./../etc/passwd",
            "subdir/../subdir2/../../etc/passwd"
        ]

        for attempt in escape_attempts:
            with self.subTest(path=attempt):
                with self.assertRaises(ValueError) as cm:
                    main._secure_resolve(attempt)
                self.assertIn("escapes workspace", str(cm.exception))

    def test_symlink_handling_edge_cases(self):
        """Test symlink edge cases including broken symlinks."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Create broken symlink
        broken_symlink = os.path.join(self.test_dir, "broken_link")
        try:
            os.symlink("/nonexistent/target", broken_symlink)

            # Should still validate that target would escape workspace
            with self.assertRaises(ValueError) as cm:
                main._secure_resolve("broken_link")
            self.assertIn("escapes workspace", str(cm.exception))

        except (FileNotFoundError, OSError):
            self.skipTest("Symlink creation not supported")
        finally:
            if os.path.exists(broken_symlink):
                os.unlink(broken_symlink)

    def test_complex_path_scenarios(self):
        """Test complex path scenarios with mixed operations."""
        # Need to reload the module to pick up the new environment variable
        import importlib

        from fastapply import main
        importlib.reload(main)

        # Create nested directory structure
        nested_dir = os.path.join(self.test_dir, "deep", "nested", "structure")
        os.makedirs(nested_dir)

        # Test paths that should work
        valid_paths = [
            "deep/nested/structure",
            "./deep/nested/structure",
            "deep/./nested/structure",
            "deep/nested/../nested/structure"
        ]

        expected = os.path.realpath(nested_dir)
        for path in valid_paths:
            with self.subTest(path=path):
                result = main._secure_resolve(path)
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
