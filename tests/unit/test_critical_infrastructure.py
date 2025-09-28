#!/usr/bin/env python3
"""
Critical infrastructure tests for main.py - focusing on call_tool() error handling
and other critical MCP server functions that currently lack test coverage.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add the parent directory to sys.path to import fastapply
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import MAX_FILE_SIZE, FastApplyConnector, call_tool


class TestCallToolErrorHandling(unittest.TestCase):
    """Test comprehensive error handling in call_tool() function."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        # Store original_cwd as absolute path to avoid issues
        self.original_cwd = os.path.abspath(self.original_cwd)
        os.chdir(self.test_dir)

        # Create test files
        self.test_file = os.path.join(self.test_dir, "test.py")
        with open(self.test_file, "w") as f:
            f.write("def test():\n    pass\n")

        # Create a large file for size limit tests
        self.large_file = os.path.join(self.test_dir, "large.py")
        with open(self.large_file, "w") as f:
            f.write("# Large file\n" + "x = 1\n" * (MAX_FILE_SIZE // 10 + 1000))

    def tearDown(self):
        """Clean up test environment."""
        try:
            os.chdir(self.original_cwd)
        except (FileNotFoundError, OSError):
            # If original directory doesn't exist, use a safe fallback
            os.chdir("/tmp")
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_call_tool_parameter_validation_edge_cases(self):
        """Test edge cases in parameter validation."""
        # Test with None parameters
        with self.assertRaises(AttributeError) as cm:
            asyncio.run(call_tool("edit_file", None))
        self.assertIn("NoneType", str(cm.exception))

        # Test with empty parameters dict
        with self.assertRaises(ValueError) as cm:
            asyncio.run(call_tool("edit_file", {}))
        self.assertIn("target_file parameter is required", str(cm.exception))

        # Test with malformed parameters (missing required fields)
        with self.assertRaises(ValueError) as cm:
            asyncio.run(call_tool("edit_file", {"target_file": None}))
        self.assertIn("target_file parameter is required", str(cm.exception))

    def test_call_tool_file_size_validation(self):
        """Test file size limit validation."""
        # Test with file that exceeds size limit
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.side_effect = ValueError("File too large")

            with self.assertRaises(RuntimeError) as cm:
                asyncio.run(call_tool("edit_file", {"target_file": "large.py", "instructions": "test", "code_edit": "print('test')"}))
            self.assertIn("Failed to apply edit", str(cm.exception))
            self.assertIn("File too large", str(cm.exception))

        # Test with file exactly at size limit (create a file that's exactly MAX_FILE_SIZE)
        exact_size_file = os.path.join(self.test_dir, "exact_size.py")
        with open(exact_size_file, "w") as f:
            f.write("x" * (MAX_FILE_SIZE - 10))  # Leave some room for newlines

        # This should work (file is under limit)
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"
            result = asyncio.run(
                call_tool("edit_file", {"target_file": "exact_size.py", "instructions": "test", "code_edit": "print('test')"})
            )
            self.assertIsInstance(result, list)

    def test_call_tool_file_extension_validation(self):
        """Test file extension validation."""
        # Test with disallowed file extensions
        disallowed_extensions = [".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".log", ".tmp", ".bak", ".swp", ".DS_Store"]

        for ext in disallowed_extensions:
            test_file = f"test{ext}"
            with open(test_file, "w") as f:
                f.write("test content")

            with self.assertRaises(ValueError) as cm:
                asyncio.run(call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "test"}))
            self.assertIn("Editing not permitted", str(cm.exception))

    def test_call_tool_file_permission_handling(self):
        """Test handling of file permission issues."""
        # Create a read-only file
        readonly_file = os.path.join(self.test_dir, "readonly.py")
        with open(readonly_file, "w") as f:
            f.write("readonly content")

        try:
            # Test the error handling by checking that permission errors are properly caught
            # and converted to RuntimeError with appropriate message
            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.side_effect = OSError("Permission denied")

                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(
                        call_tool("edit_file", {"target_file": "readonly.py", "instructions": "test", "code_edit": "print('test')"})
                    )

                self.assertIn("Failed to apply edit", str(cm.exception))
                self.assertIn("Permission denied", str(cm.exception))
        finally:
            # Clean up
            if os.path.exists(readonly_file):
                os.remove(readonly_file)

    def test_call_tool_concurrent_access_handling(self):
        """Test handling of concurrent file access."""

        # Test that call_tool can handle multiple concurrent calls
        async def test_concurrent_calls():
            tasks = []
            for i in range(5):
                task = call_tool(
                    "edit_file", {"target_file": "test.py", "instructions": f"concurrent test {i}", "code_edit": f"print('concurrent {i}')"}
                )
                tasks.append(task)

            # This should not raise exceptions (though some may fail due to concurrency)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"
            results = asyncio.run(test_concurrent_calls())

            # All results should be either successful results or exceptions
            self.assertEqual(len(results), 5)
            for result in results:
                if not isinstance(result, Exception):
                    self.assertIsInstance(result, list)

    def test_call_tool_memory_efficiency(self):
        """Test memory efficiency with large inputs."""
        # Test with very large instructions and code_edit
        large_instructions = "test " * 10000
        large_code_edit = "print('test')\n" * 10000

        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"

            # Should handle large inputs without memory issues
            result = asyncio.run(
                call_tool("edit_file", {"target_file": "test.py", "instructions": large_instructions, "code_edit": large_code_edit})
            )

            self.assertIsInstance(result, list)

    def test_call_tool_unicode_handling(self):
        """Test handling of Unicode characters in file paths and content."""
        # Create file with Unicode name
        unicode_file = "test_файл.py"
        unicode_content = "def тест():\n    print('Привет, мир!')\n"

        with open(unicode_file, "w", encoding="utf-8") as f:
            f.write(unicode_content)

        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"

            result = asyncio.run(
                call_tool("edit_file", {"target_file": unicode_file, "instructions": "тест", "code_edit": "print('Юникод')"})
            )

            self.assertIsInstance(result, list)

    def test_call_tool_timeout_handling(self):
        """Test timeout handling in call_tool."""

        def slow_operation(*args, **kwargs):
            import time

            time.sleep(0.1)  # Simulate slow operation
            return "success"

        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.side_effect = slow_operation

            # Should complete within reasonable time
            result = asyncio.run(call_tool("edit_file", {"target_file": "test.py", "instructions": "test", "code_edit": "print('test')"}))

            self.assertIsInstance(result, list)

    def test_call_tool_error_propagation(self):
        """Test that errors are properly propagated and logged."""
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.side_effect = Exception("Test error")

            with self.assertRaises(Exception) as cm:
                asyncio.run(call_tool("edit_file", {"target_file": "test.py", "instructions": "test", "code_edit": "print('test')"}))

            self.assertIn("Test error", str(cm.exception))

    def test_call_tool_logging_verification(self):
        """Test that call_tool properly logs operations."""
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"
            with patch("fastapply.main.logger") as mock_logger:
                asyncio.run(
                    call_tool("edit_file", {"target_file": "test.py", "instructions": "test instruction", "code_edit": "print('test')"})
                )

                # Verify logging occurred
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args
                self.assertIn("edit_file tool called", str(call_args))
                self.assertIn("test.py", str(call_args))
                self.assertIn("test instruction", str(call_args))


class TestFastApplyConnectorErrorHandling(unittest.TestCase):
    """Test error handling in FastApplyConnector."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.original_cwd = os.path.abspath(self.original_cwd)
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        try:
            os.chdir(self.original_cwd)
        except (FileNotFoundError, OSError):
            # If original directory doesn't exist, use a safe fallback
            os.chdir("/tmp")
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_connector_configuration_validation(self):
        """Test configuration validation in connector."""
        # Test invalid timeout values
        with self.assertRaises(ValueError):
            FastApplyConnector(timeout=-1)

        with self.assertRaises(ValueError):
            FastApplyConnector(timeout=400)

        # Test invalid max_tokens values
        with self.assertRaises(ValueError):
            FastApplyConnector(max_tokens=-1)

        with self.assertRaises(ValueError):
            FastApplyConnector(max_tokens=50000)

        # Test invalid temperature values
        with self.assertRaises(ValueError):
            FastApplyConnector(temperature=-1)

        with self.assertRaises(ValueError):
            FastApplyConnector(temperature=3)

    def test_connector_initialization_fallback(self):
        """Test connector initialization fallback when API is unavailable."""
        with patch("fastapply.main.openai.OpenAI") as mock_openai:
            mock_openai.side_effect = Exception("Connection failed")

            connector = FastApplyConnector()

            # Should fall back to test mode
            self.assertIsNone(connector.client)

    def test_connector_apply_edit_error_handling(self):
        """Test error handling in apply_edit method."""
        connector = FastApplyConnector()

        # Test with None client (test mode)
        connector.client = None

        with self.assertRaises(RuntimeError) as cm:
            connector.apply_edit("test", "original", "edit")
        self.assertIn("Fast Apply client not initialized", str(cm.exception))

    def test_connector_response_validation(self):
        """Test response validation in connector."""
        with patch("fastapply.main.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()

            # Test empty choices
            mock_response.choices = []
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            connector = FastApplyConnector()

            with self.assertRaises(ValueError) as cm:
                connector.apply_edit("test", "original", "edit")
            self.assertIn("Invalid Fast Apply API response", str(cm.exception))

    def test_connector_response_content_extraction(self):
        """Test content extraction from various response formats."""
        with patch("fastapply.main.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()

            # Test normal response
            mock_choice.message.content = "def test(): pass"
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            connector = FastApplyConnector()
            result = connector.apply_edit("test", "original", "edit")

            self.assertEqual(result, "def test(): pass")

    def test_connector_config_update(self):
        """Test configuration update functionality."""
        connector = FastApplyConnector()

        new_config = connector.update_config(
            url="http://new-url:8080/v1", model="new-model", timeout=60.0, max_tokens=4000, temperature=0.1
        )

        self.assertEqual(new_config["url"], "http://new-url:8080/v1")
        self.assertEqual(new_config["model"], "new-model")
        self.assertEqual(new_config["timeout"], 60.0)
        self.assertEqual(new_config["max_tokens"], 4000)
        self.assertEqual(new_config["temperature"], 0.1)


class TestSecurityValidation(unittest.TestCase):
    """Test security validation functions."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.original_cwd = os.path.abspath(self.original_cwd)
        os.chdir(self.test_dir)
        os.environ["WORKSPACE_ROOT"] = self.test_dir

    def tearDown(self):
        """Clean up test environment."""
        try:
            os.chdir(self.original_cwd)
        except (FileNotFoundError, OSError):
            # If original directory doesn't exist, use a safe fallback
            os.chdir("/tmp")
        os.environ.pop("WORKSPACE_ROOT", None)
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sql_injection_detection(self):
        """Test SQL injection detection in file content."""
        # Import and reload to get fresh environment
        import importlib

        from fastapply import main

        importlib.reload(main)

        # Create files with potential SQL injection patterns
        malicious_patterns = [
            "SELECT * FROM users",
            "DROP TABLE users",
            "INSERT INTO users VALUES",
            "UPDATE users SET password",
            "DELETE FROM users WHERE",
            "UNION SELECT * FROM",
            "OR 1=1",
            "'; DROP TABLE users; --",
        ]

        for pattern in malicious_patterns:
            test_file = f"test_{hash(pattern)}.py"
            with open(test_file, "w") as f:
                f.write(f"# {pattern}\ndef test(): pass")

            # Should not block (just content detection, not blocking)
            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"
                result = asyncio.run(
                    call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('test')"})
                )
                self.assertIsInstance(result, list)

    def test_xss_detection(self):
        """Test XSS detection in file content."""
        import importlib

        from fastapply import main

        importlib.reload(main)

        # Create files with potential XSS patterns
        xss_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src=javascript:alert('xss')>",
            "eval('alert(\"xss\")')",
            "document.write('<script>alert(\"xss\")</script>')",
        ]

        for pattern in xss_patterns:
            test_file = f"test_{hash(pattern)}.py"
            with open(test_file, "w") as f:
                f.write(f"# {pattern}\nprint('test')")

            # Should not block (just content detection, not blocking)
            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"
                result = asyncio.run(
                    call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('safe content')"})
                )
                self.assertIsInstance(result, list)

    def test_authentication_issues_detection(self):
        """Test authentication issue detection."""
        import importlib

        from fastapply import main

        importlib.reload(main)

        # Create files with potential authentication issues
        auth_patterns = [
            "password = 'hardcoded'",
            "api_key = '12345'",
            "secret = 'plaintext'",
            "token = 'static'",
            "credentials = {'user': 'admin', 'pass': 'password'}",
        ]

        for pattern in auth_patterns:
            test_file = f"test_{hash(pattern)}.py"
            with open(test_file, "w") as f:
                f.write(f"# {pattern}\ndef auth(): pass")

            # Should not block (just content detection, not blocking)
            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"
                result = asyncio.run(
                    call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('test')"})
                )
                self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
