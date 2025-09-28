#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Fast Apply MCP server tests.
"""

import os
import shutil
import sys
import tempfile

import pytest

# Add the parent directory to sys.path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.main import FastApplyConnector


@pytest.fixture
def test_env():
    """Fixture that provides a test environment with temporary directory."""
    test_dir = tempfile.mkdtemp()
    original_workspace = os.environ.get("WORKSPACE_ROOT")

    os.environ["WORKSPACE_ROOT"] = test_dir
    connector = FastApplyConnector()

    yield {
        "test_dir": test_dir,
        "connector": connector,
        "original_workspace": original_workspace
    }

    # Cleanup
    if original_workspace:
        os.environ["WORKSPACE_ROOT"] = original_workspace
    else:
        os.environ.pop("WORKSPACE_ROOT", None)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_files():
    """Fixture that provides sample file contents."""
    return {
        "python_file": {
            "name": "test.py",
            "content": '''def hello():
    return "Hello, World!"

def main():
    print(hello())

if __name__ == "__main__":
    main()
'''
        },
        "javascript_file": {
            "name": "test.js",
            "content": '''function hello() {
    return "Hello, World!";
}

function main() {
    console.log(hello());
}

main();
'''
        },
        "json_file": {
            "name": "config.json",
            "content": '''{
    "name": "test-project",
    "version": "1.0.0",
    "description": "Test configuration"
}'''
        },
        "markdown_file": {
            "name": "README.md",
            "content": '''# Test Project

This is a test project for Fast Apply MCP server testing.

## Features

- Feature testing
- Integration testing
- Performance testing
'''
        }
    }


@pytest.fixture
def security_test_cases():
    """Fixture that provides security test cases."""
    return [
        ("../etc/passwd", "Path traversal attempt"),
        ("/etc/passwd", "Absolute path outside workspace"),
        ("../../windows/system32/config", "Windows path traversal"),
        ("~/../etc/passwd", "Home directory traversal"),
        ("./../etc/passwd", "Relative path traversal"),
        ("symlink_attack", "Symlink attack simulation"),
    ]


@pytest.fixture
def validation_test_cases():
    """Fixture that provides validation test cases."""
    return [
        ("", "Empty input"),
        (None, "None input"),
        ("valid_content", "Valid content"),
        ("x" * 1000000, "Large content (1MB)"),
        ("import os\nos.system('rm -rf /')", "Dangerous content"),
        ("eval(user_input)", "Eval usage"),
    ]


@pytest.fixture
def backup_test_cases():
    """Fixture that provides backup test cases."""
    return [
        ("small_file.txt", "Small content", "Small file backup test"),
        ("medium_file.py", "def function():\n    return 'medium content'\n", "Medium file backup test"),
        ("empty_file.txt", "", "Empty file backup test"),
        ("unicode_file.txt", "Hello, 世界! 🌍", "Unicode file backup test"),
    ]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-related"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add security marker to security tests
        if "security" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)

        # Add integration marker to integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Add performance marker to performance tests
        if "performance" in item.nodeid.lower() or "large" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        # Add slow marker to potentially slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["concurrent", "large_file", "multiple"]):
            item.add_marker(pytest.mark.slow)


# Custom pytest assertions and helpers
def assert_file_exists(filepath, msg=None):
    """Custom assertion to check if file exists."""
    if not os.path.exists(filepath):
        error_msg = msg or f"File does not exist: {filepath}"
        raise AssertionError(error_msg)


def assert_file_content_equals(filepath, expected_content, msg=None):
    """Custom assertion to check file content."""
    assert_file_exists(filepath)

    with open(filepath, 'r') as f:
        actual_content = f.read()

    if actual_content != expected_content:
        error_msg = msg or f"File content mismatch. Expected: {expected_content}, Got: {actual_content}"
        raise AssertionError(error_msg)


def assert_backup_created(original_file, msg=None):
    """Custom assertion to check if backup was created."""
    backup_file = original_file + ".bak"
    assert_file_exists(backup_file, msg or f"Backup file not created for: {original_file}")


def assert_operation_success(result, msg=None):
    """Custom assertion to check if operation was successful."""
    if not isinstance(result, dict):
        error_msg = msg or f"Expected dict result, got {type(result)}"
        raise AssertionError(error_msg)

    if not result.get("success", False):
        error_msg = msg or f"Operation failed: {result.get('error', 'Unknown error')}"
        raise AssertionError(error_msg)


def assert_no_security_violation(result, msg=None):
    """Custom assertion to check for security violations."""
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error", "").lower()
        if any(keyword in error_msg for keyword in ["invalid", "security", "escape", "traversal"]):
            error_msg = msg or f"Security violation detected: {result.get('error')}"
            raise AssertionError(error_msg)
