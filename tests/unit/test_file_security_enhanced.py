#!/usr/bin/env python3
"""
Enhanced file security tests for main.py and security validation
Testing file path validation, size limits, and security checks that need coverage.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the parent directory to sys.path to import fastapply
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import MAX_FILE_SIZE, _is_allowed_edit_target, _secure_resolve, call_tool


class TestFileSecurityEnhanced(unittest.TestCase):
    """Enhanced file security tests."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        self.original_workspace = os.environ.get("WORKSPACE_ROOT")
        os.chdir(self.test_dir)
        os.environ["WORKSPACE_ROOT"] = self.test_dir

        # Create test files of various sizes
        self.small_file = os.path.join(self.test_dir, "small.py")
        with open(self.small_file, "w") as f:
            f.write("print('small file')")

        self.medium_file = os.path.join(self.test_dir, "medium.py")
        with open(self.medium_file, "w") as f:
            f.write("# medium file\n" + "x = 1\n" * 1000)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        if self.original_workspace:
            os.environ["WORKSPACE_ROOT"] = self.original_workspace
        else:
            os.environ.pop("WORKSPACE_ROOT", None)
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_secure_resolve_complex_scenarios(self):
        """Test complex secure resolve scenarios."""
        # Test relative path normalization
        complex_paths = [
            ("./test.py", True),  # Should succeed
            ("../test.py", False),  # Should fail - escapes workspace
            ("../../test.py", False),  # Should fail - escapes workspace
            ("subdir/../../test.py", False),  # Should fail - escapes workspace
            ("subdir/../test.py", True),  # Should succeed - normalizes to test.py
            ("subdir/./test.py", True),  # Should succeed - normalizes to subdir/test.py
            ("./subdir/../test.py", True),  # Should succeed - normalizes to test.py
        ]

        for path, should_succeed in complex_paths:
            if should_succeed:
                # Should resolve successfully for safe paths
                result = _secure_resolve(path)
                expected = os.path.realpath(os.path.join(self.test_dir, path))
                self.assertEqual(result, expected)
            else:
                # Should be blocked for unsafe paths
                with self.assertRaises(ValueError) as cm:
                    _secure_resolve(path)
                self.assertIn("escapes workspace", str(cm.exception))

    def test_secure_resolve_symlink_chains(self):
        """Test symlink chain detection and handling."""
        try:
            # Create a symlink chain that points outside workspace
            outside_file = os.path.join(os.path.dirname(self.test_dir), "outside.txt")
            symlink1 = os.path.join(self.test_dir, "link1")
            symlink2 = os.path.join(self.test_dir, "link2")

            with open(outside_file, "w") as f:
                f.write("outside content")

            os.symlink(outside_file, symlink1)
            os.symlink(symlink1, symlink2)

            # Both symlinks should be blocked
            with self.assertRaises(ValueError) as cm:
                _secure_resolve("link1")
            self.assertIn("escapes workspace", str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                _secure_resolve("link2")
            self.assertIn("escapes workspace", str(cm.exception))

        except (OSError, FileNotFoundError):
            self.skipTest("Symlink creation not supported")
        finally:
            # Cleanup
            for path in [symlink1, symlink2, outside_file]:
                if os.path.exists(path):
                    if os.path.islink(path):
                        os.unlink(path)
                    else:
                        os.remove(path)

    def test_secure_resolve_edge_cases(self):
        """Test edge cases in secure resolve."""
        # Test empty path
        result = _secure_resolve("")
        expected = os.path.realpath(self.test_dir)
        self.assertEqual(result, expected)

        # Test single dot
        result = _secure_resolve(".")
        expected = os.path.realpath(self.test_dir)
        self.assertEqual(result, expected)

        # Test double dot - should escape workspace and be blocked
        with self.assertRaises(ValueError) as cm:
            _secure_resolve("..")
        self.assertIn("escapes workspace", str(cm.exception))

        # Test path with multiple slashes
        result = _secure_resolve("subdir//test.py")
        expected = os.path.realpath(os.path.join(self.test_dir, "subdir", "test.py"))
        self.assertEqual(result, expected)

    def test_is_allowed_edit_target_comprehensive(self):
        """Test comprehensive file extension validation."""
        # Test allowed extensions
        allowed_files = [
            "test.py",
            "test.js",
            "test.ts",
            "test.jsx",
            "test.tsx",
            "test.md",
            "test.txt",
            "test.json",
            "test.yml",
            "test.yaml",
        ]

        for filename in allowed_files:
            self.assertTrue(_is_allowed_edit_target(filename), f"Should allow {filename}")

        # Test disallowed extensions
        disallowed_files = [
            "test.exe",
            "test.dll",
            "test.so",
            "test.dylib",
            "test.bin",
            "test.obj",
            "test.o",
            "test.a",
            "test.lib",
            "test.com",
            "test.bat",
            "test.cmd",
            "test.pif",
            "test.scr",
            "test.shs",
            "test.shb",
            "test.vbs",
            "test.vbe",
            "test.jse",
            "test.wsf",
            "test.wsc",
            "test.wsh",
            "test.msc",
            "test.jar",
            "test.class",
            "test.war",
            "test.ear",
            "test.apk",
            "test.ipa",
            "test.deb",
            "test.rpm",
            "test.dmg",
            "test.iso",
            "test.img",
            "test.vmdk",
            "test.ovf",
            "test.ova",
            "test.vhd",
            "test.vhdx",
            "test.qcow",
            "test.qcow2",
            "test.raw",
            "test.tar",
            "test.tar.gz",
            "test.tar.bz2",
            "test.tar.xz",
            "test.zip",
            "test.rar",
            "test.7z",
            "test.gz",
            "test.bz2",
            "test.xz",
            "test.lz4",
            "test.zst",
            "test.pdf",
            "test.doc",
            "test.docx",
            "test.xls",
            "test.xlsx",
            "test.ppt",
            "test.pptx",
            "test.odt",
            "test.ods",
            "test.odp",
            "test.rtf",
            "test.latex",
            "test.tex",
            "test.dvi",
            "test.ps",
            "test.eps",
            "test.svg",
            "test.png",
            "test.jpg",
            "test.jpeg",
            "test.gif",
            "test.bmp",
            "test.tiff",
            "test.webp",
            "test.ico",
            "test.mp3",
            "test.mp4",
            "test.avi",
            "test.mkv",
            "test.mov",
            "test.wmv",
            "test.flv",
            "test.webm",
            "test.m4v",
            "test.3gp",
            "test.wav",
            "test.flac",
            "test.ogg",
            "test.m4a",
            "test.wma",
            "test.aac",
            "test.log",
            "test.tmp",
            "test.temp",
            "test.bak",
            "test.backup",
            "test.old",
            "test.swp",
            "test.swo",
            "test.DS_Store",
            "test.Thumbs.db",
            "test.desktop.ini",
            "test.ntuser.dat",
            "test.reg",
            "test.pdb",
            "test.pch",
            "test.idb",
            "test.ids",
            "test.tds",
            "test.suo",
            "test.user",
            "test.aps",
            "test.ncb",
            "test.opt",
            "test.plg",
            "test.exp",
            "test.ilk",
            "test.map",
            "test.dmp",
            "test.core",
            "test.minidump",
        ]

        for filename in disallowed_files:
            self.assertFalse(_is_allowed_edit_target(filename), f"Should not allow {filename}")

    def test_file_size_validation_extreme_cases(self):
        """Test file size validation with extreme cases."""
        # Test file exactly at size limit
        exact_size_file = os.path.join(self.test_dir, "exact_size.py")
        exact_content = "x" * (MAX_FILE_SIZE - 10)  # Leave room for newlines
        with open(exact_size_file, "w") as f:
            f.write(exact_content)

        # Should be allowed
        file_size = os.path.getsize(exact_size_file)
        self.assertLessEqual(file_size, MAX_FILE_SIZE)

        # Test file just over size limit
        oversized_file = os.path.join(self.test_dir, "oversized.py")
        oversized_content = "x" * (MAX_FILE_SIZE + 100)
        with open(oversized_file, "w") as f:
            f.write(oversized_content)

        # Should be blocked in call_tool
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            with self.assertRaises(ValueError) as cm:
                asyncio.run(call_tool("edit_file", {"target_file": "oversized.py", "instructions": "test", "code_edit": "print('test')"}))
            self.assertIn("File too large", str(cm.exception))
            mock_connector.apply_edit.assert_not_called()

    def test_call_tool_file_type_validation(self):
        """Test file type validation in call_tool."""
        # Test with files that have allowed extensions
        allowed_extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json", ".yml", ".yaml"]

        for ext in allowed_extensions:
            test_file = f"test{ext}"
            with open(test_file, "w") as f:
                f.write("test content")

            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"

                # Should succeed
                result = asyncio.run(
                    call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('test')"})
                )
                self.assertIsInstance(result, list)

        # Test with files that have disallowed extensions
        disallowed_extensions = [".exe", ".dll", ".so", ".log", ".tmp"]

        for ext in disallowed_extensions:
            test_file = f"test{ext}"
            with open(test_file, "w") as f:
                f.write("test content")

            with self.assertRaises(ValueError) as cm:
                asyncio.run(call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('test')"}))
            self.assertIn("Editing not permitted", str(cm.exception))

    def test_call_tool_file_encoding_handling(self):
        """Test file encoding handling in call_tool."""
        # Test with different encodings - but only UTF-8 since that's what we support
        encodings = [
            ("utf-8", "utf-8 test"),
            ("ascii", "ascii test"),
        ]

        for encoding, content in encodings:
            test_file = f"test_{encoding.replace('-', '_')}.py"
            with open(test_file, "w", encoding=encoding) as f:
                f.write(f"# {encoding} test\nprint('{content}')")

            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"

                result = asyncio.run(
                    call_tool("edit_file", {"target_file": test_file, "instructions": "test", "code_edit": "print('encoded')"})
                )
                self.assertIsInstance(result, list)

        # Test that non-UTF-8 files properly raise IOError
        try:
            # Create a file with UTF-16 BOM
            with open("test_utf_16.py", "wb") as f:
                f.write(b'\xff\xfe')  # UTF-16 LE BOM
                f.write('t'.encode('utf-16-le'))
                f.write('e'.encode('utf-16-le'))
                f.write('s'.encode('utf-16-le'))
                f.write('t'.encode('utf-16-le'))

            with self.assertRaises(IOError) as cm:
                asyncio.run(
                    call_tool("edit_file", {"target_file": "test_utf_16.py", "instructions": "test", "code_edit": "print('test')"})
                )
            self.assertIn("Failed to read file", str(cm.exception))
            self.assertIn("invalid start byte", str(cm.exception))
        except (OSError, UnicodeError):
            # Skip if file system doesn't support this
            pass

    def test_call_tool_file_permission_scenarios(self):
        """Test file permission scenarios in call_tool."""
        # Test with read-only file
        readonly_file = os.path.join(self.test_dir, "readonly.py")
        with open(readonly_file, "w") as f:
            f.write("readonly content")

        original_mode = os.stat(readonly_file).st_mode
        try:
            os.chmod(readonly_file, 0o444)  # Read-only

            with patch("fastapply.main.fast_apply_connector") as mock_connector:
                mock_connector.apply_edit.return_value = "success"

                # Should still work (file reading is allowed)
                result = asyncio.run(
                    call_tool("edit_file", {"target_file": "readonly.py", "instructions": "test", "code_edit": "print('test')"})
                )
                self.assertIsInstance(result, list)

        finally:
            os.chmod(readonly_file, original_mode)

    def test_call_tool_special_file_names(self):
        """Test call_tool with special file names."""
        special_names = [
            "test with spaces.py",
            "test-with-dashes.py",
            "test_with_underscores.py",
            "test.with.dots.py",
            "test123.py",
            "TEST.py",
            "Test.PY",
            "тест.py",  # Unicode
            "测试.py",  # Chinese
            "🎯.py",  # Emoji
            "file (copy).py",
            "file 1.py",
            "file[1].py",
            "file(1).py",
        ]

        for filename in special_names:
            try:
                with open(filename, "w") as f:
                    f.write("print('special name')")

                with patch("fastapply.main.fast_apply_connector") as mock_connector:
                    mock_connector.apply_edit.return_value = "success"

                    result = asyncio.run(
                        call_tool("edit_file", {"target_file": filename, "instructions": "test", "code_edit": "print('test')"})
                    )
                    self.assertIsInstance(result, list)

            except (OSError, UnicodeError):
                # Skip files that can't be created on this filesystem
                continue

    def test_call_tool_concurrent_file_access(self):
        """Test concurrent file access handling."""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                with patch("fastapply.main.fast_apply_connector") as mock_connector:
                    mock_connector.apply_edit.return_value = f"worker {worker_id} result"

                    result = asyncio.run(
                        call_tool(
                            "edit_file",
                            {"target_file": "small.py", "instructions": f"worker {worker_id}", "code_edit": f"print('worker {worker_id}')"},
                        )
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should not have errors
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 10)

    def test_call_tool_memory_usage_large_files(self):
        """Test memory usage with large files."""
        # Create a large file (but within size limits)
        large_file = os.path.join(self.test_dir, "large_but_valid.py")
        large_content = "# Large file\n" + "x = 1\n" * (MAX_FILE_SIZE // 10)

        with open(large_file, "w") as f:
            f.write(large_content)

        # Verify file size
        file_size = os.path.getsize(large_file)
        self.assertLess(file_size, MAX_FILE_SIZE)

        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"

            # Should handle large file without memory issues
            result = asyncio.run(
                call_tool("edit_file", {"target_file": "large_but_valid.py", "instructions": "test", "code_edit": "print('small edit')"})
            )
            self.assertIsInstance(result, list)

    def test_security_validation_workflow(self):
        """Test complete security validation workflow."""
        # Test file with multiple security issues
        security_test_file = os.path.join(self.test_dir, "security_test.py")
        with open(security_test_file, "w") as f:
            f.write("""
import os
import hashlib

# Hardcoded credentials
API_KEY = "123456789"
password = "admin123"

# SQL injection vulnerability
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute_query(query)

# XSS vulnerability
def display_user_input(user_input):
    return "<div>" + user_input + "</div>""")

        # Security validation should pass through call_tool
        with patch("fastapply.main.fast_apply_connector") as mock_connector:
            mock_connector.apply_edit.return_value = "success"

            result = asyncio.run(
                call_tool(
                    "edit_file",
                    {"target_file": "security_test.py", "instructions": "fix security issues", "code_edit": "# Fixed security issues"},
                )
            )
            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
