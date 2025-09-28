#!/usr/bin/env python3
"""Tests for newly added Fast Apply MCP server features.
Covers:
- Extension allowlist rejection
- Optimistic concurrency control (force override)
- JSON output formatting for edit_file and dry_run_edit_file
- health_status tool output
- Request size pre-flight guard in FastApplyConnector.apply_edit
- Strict path security resolution
- _atomic_write directory auto-creation (new behavior)
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapply import main  # type: ignore  # noqa: E402
from fastapply.main import FastApplyConnector, _atomic_write, call_tool  # noqa: E402


class TestExtensionAllowlist(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        # Create a disallowed extension file
        with open("malware.exe", "w", encoding="utf-8") as f:
            f.write("dummy exe content")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def test_edit_file_disallowed_extension(self):
        with self.assertRaises(ValueError) as cm:
            # Patch connector to avoid real API call even though we expect early failure
            with patch.object(main.fast_apply_connector, "apply_edit"):
                asyncio.run(call_tool(
                    "edit_file",
                    {
                        "target_file": "malware.exe",
                        "instructions": "Do nothing",
                        "code_edit": "// attempt",
                    },
                ))
        self.assertIn("Editing not permitted", str(cm.exception))


class TestOptimisticConcurrency(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        with open("file.py", "w", encoding="utf-8") as f:
            f.write("print('v1')\n")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def test_concurrent_modification_detected(self):
        # Patch apply_edit to simulate a change
        fake_return = {
            "merged_code": "print('v2')\n",
            "has_changes": True,
            "udiff": "--- a\n+++ b\n",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }
        with patch.object(main.fast_apply_connector, "apply_edit", return_value=fake_return):
            # Modify file after call_tool reads it but before write.
            # We insert a side-effect by patching open during second read.
            original_open = open

            def open_side_effect(path, mode='r', *a, **kw):  # noqa: D401
                if path.endswith("file.py") and mode == 'r':
                    data = original_open(path, mode, *a, **kw).read()
                    # After initial read, mutate the file to trigger hash mismatch
                    with original_open(path, 'w', encoding='utf-8') as wf:
                        wf.write("print('CHANGED')\n")
                    # Return data from first read
                    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
                    tmp.write(data)
                    tmp.close()
                    return original_open(tmp.name, 'r', encoding='utf-8')
                return original_open(path, mode, *a, **kw)

            with patch("builtins.open", side_effect=open_side_effect):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(call_tool(
                        "edit_file",
                        {
                            "target_file": "file.py",
                            "instructions": "update",
                            "code_edit": "print('v2')\n",
                        },
                    ))
        self.assertIn("File changed on disk", str(cm.exception))

    def test_force_override(self):
        fake_return = {
            "merged_code": "print('v2')\n",
            "has_changes": True,
            "udiff": "",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }
        with patch.object(main.fast_apply_connector, "apply_edit", return_value=fake_return):
            # Simulate external change before calling
            with open("file.py", "w", encoding="utf-8") as f:
                f.write("print('modified externally')\n")
            # Should succeed with force
            result = asyncio.run(call_tool(
                "edit_file",
                {
                    "target_file": "file.py",
                    "instructions": "update",
                    "code_edit": "print('v2')\n",
                    "force": True,
                    "output_format": "json",
                },
            ))
        self.assertIn("text", result[0])
        payload = json.loads(result[0]["text"])  # json_dumps uses indentation
        self.assertTrue(payload["has_changes"])  # type: ignore


class TestJSONOutputs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        with open("sample.py", "w", encoding="utf-8") as f:
            f.write("print('hi')\n")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def test_dry_run_json(self):
        fake_return = {
            "merged_code": "print('hi there')\n",
            "has_changes": True,
            "udiff": "--- a\n+++ b\n",
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }
        with patch.object(main.fast_apply_connector, "apply_edit", return_value=fake_return):
            result = asyncio.run(call_tool(
                "dry_run_edit_file",
                {
                    "target_file": "sample.py",
                    "code_edit": "print('hi there')\n",
                    "instruction": "update",
                    "output_format": "json",
                },
            ))
        payload = json.loads(result[0]["text"])  # parse JSON
        self.assertIn("udiff", payload)
        self.assertTrue(payload["has_changes"])  # type: ignore


class TestHealthAndSecurity(unittest.TestCase):
    def test_health_status(self):
        result = asyncio.run(call_tool("health_status", {}))
        self.assertIn("text", result[0])
        txt = result[0]["text"]
        self.assertIn("strict_paths=", txt)
        self.assertIn("tools=", txt)

    def test_path_escape(self):
        # read_multiple_files returns error text per path instead of raising globally
        result = asyncio.run(call_tool("read_multiple_files", {"paths": ["../outside.txt"]}))
        self.assertIn("text", result[0])
        self.assertIn("Error", result[0]["text"])  # should contain error indicator


class TestRequestSizeGuard(unittest.TestCase):
    def test_request_size_limit(self):
        # Build a large instruction + code that exceeds small injected limit
        big_snippet = "x" * 5000
        connector = FastApplyConnector()
        # Provide fake client to avoid None error but request should fail before use
        class FakeChat:
            class ChatNamespace:
                class Completions:  # noqa: N801 (API mimic)
                    @staticmethod
                    def create(**kwargs):  # pragma: no cover - should not be reached
                        return None
                completions = Completions
            chat = ChatNamespace()
        connector.client = FakeChat()  # type: ignore

        # Patch MAX_REQUEST_BYTES directly since it's set at module import time
        with patch.object(main, 'MAX_REQUEST_BYTES', 1000):
            with self.assertRaises(ValueError) as cm:
                connector.apply_edit(
                    original_code=big_snippet,
                    code_edit=big_snippet,
                    instruction="expand",
                    file_path="file.py",
                )
            self.assertIn("Request size", str(cm.exception))
            self.assertIn("exceeds limit", str(cm.exception))


class TestAtomicWriteBehavior(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def test_atomic_write_creates_directories(self):
        nested_path = os.path.join("a", "b", "c.txt")
        _atomic_write(nested_path, "data")
        self.assertTrue(os.path.exists(nested_path))
        with open(nested_path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "data")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
