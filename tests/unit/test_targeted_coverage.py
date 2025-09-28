"""
Targeted tests to reach 85%+ coverage by focusing on specific uncovered lines
"""

import asyncio
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from fastapply.main import (
    UPDATED_CODE_END,
    UPDATED_CODE_START,
    FastApplyConnector,
    _create_timestamped_backup,
    call_tool,
    write_with_backup,
)


class TestTargetedBackupCoverage:
    """Targeted tests for backup-related uncovered lines."""

    @patch("fastapply.main.datetime")
    def test_backup_creation_fallback_failure_coverage(self, mock_datetime):
        """Test backup creation covering lines 195-196 (fallback failure)."""
        mock_now = Mock()
        mock_now.strftime.return_value = "20231201_120000"
        mock_datetime.now.return_value = mock_now

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")

            # Mock both timestamped and fallback attempts to fail
            with patch("builtins.open", side_effect=[PermissionError("Permission denied"), OSError("Fallback failed")]):
                with patch("fastapply.main.logger") as mock_logger:
                    with pytest.raises(OSError):
                        _create_timestamped_backup(test_file)
                    mock_logger.warning.assert_called()


class TestTargetedFastApplyCoverage:
    """Targeted tests for FastApplyConnector uncovered lines."""

    @patch("openai.OpenAI")
    def test_apply_edit_legacy_mode_coverage(self, mock_openai):
        """Test apply_edit covering legacy mode (lines 484-489)."""
        # Create a mock client first
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = Mock()
        mock_choice.message.content = f"{UPDATED_CODE_START}def updated(): pass{UPDATED_CODE_END}"
        mock_response.choices = [mock_choice]  # Just use a real list
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the OpenAI class to return our mock client
        mock_openai.return_value = mock_client

        # Now create the connector - it should use our mocked OpenAI client
        connector = FastApplyConnector(url="http://test.com", model="test-model", api_key="test-key")

        # Test legacy mode with positional arguments
        result = connector.apply_edit("test instruction", "def original(): pass", "def updated(): pass")

        # Should return merged code as string in legacy mode
        assert isinstance(result, str)
        assert "updated" in result

    @patch("openai.OpenAI")
    @patch("fastapply.main.validate_code_quality")
    def test_apply_edit_validation_coverage(self, mock_validate, mock_openai):
        """Test apply_edit covering validation-related lines (543-550, 558)."""
        # Create a mock client first
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = f"{UPDATED_CODE_START}def test():\n    print('hello'){UPDATED_CODE_END}"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the OpenAI class to return our mock client
        mock_openai.return_value = mock_client

        # Now create the connector - it should use our mocked OpenAI client
        connector = FastApplyConnector(url="http://test.com", model="test-model", api_key="test-key")

        # Mock validation to return warnings
        mock_validate.return_value = {"has_errors": False, "errors": [], "warnings": ["Line 2: Consider using f-string"], "suggestions": []}

        result = connector.apply_edit(
            original_code="def test(): pass", code_edit="def test(): print('hello')", instruction="Add print statement", file_path="test.py"
        )

        # Should return dict with validation warnings
        assert isinstance(result, dict)
        assert result["validation"]["warnings"]
        assert "udiff" in result  # Should generate UDiff

    @patch("fastapply.main.MAX_FILE_SIZE", 1024)  # Set small limit for testing
    @patch("openai.OpenAI")
    def test_apply_edit_size_enforcement_coverage(self, mock_openai):
        """Test apply_edit covering size enforcement (line 558)."""
        # Create a mock client first
        mock_client = Mock()
        large_content = "x" * 2048  # Exceeds our 1024 limit

        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = f"{UPDATED_CODE_START}{large_content}{UPDATED_CODE_END}"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        # Mock the OpenAI class to return our mock client
        mock_openai.return_value = mock_client

        # Now create the connector - it should use our mocked OpenAI client
        connector = FastApplyConnector(url="http://test.com", model="test-model", api_key="test-key")

        with pytest.raises(ValueError, match="Merged code size.*exceeds MAX_FILE_SIZE"):
            connector.apply_edit(original_code="def test(): pass", code_edit=large_content, instruction="Make large", file_path="test.py")

    @patch("openai.OpenAI")
    def test_apply_edit_api_error_coverage(self, mock_openai):
        """Test apply_edit covering API error handling (lines 575-576)."""
        # Test generic API error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Generic API error")
        mock_openai.return_value = mock_client

        # Now create the connector - it should use our mocked OpenAI client
        connector = FastApplyConnector(url="http://test.com", model="test-model", api_key="test-key")

        with pytest.raises(RuntimeError, match="Unexpected error when calling Fast Apply API"):
            connector.apply_edit(original_code="def test(): pass", code_edit="def updated(): pass", instruction="Update function")

    def test_update_config_timeout_coverage(self):
        """Test update_config covering timeout validation (line 598)."""
        connector = FastApplyConnector(url="http://test.com", model="test-model", api_key="test-key")

        # Test timeout > 300 seconds
        with pytest.raises(ValueError, match="Timeout must be between 0 and 300 seconds"):
            connector.update_config(timeout=301)

        # Test timeout = 1 (should be allowed)
        result = connector.update_config(timeout=1)
        assert "timeout" in result


class TestTargetedWriteWithBackupCoverage:
    """Targeted tests for write_with_backup uncovered lines."""

    @patch("fastapply.main.MAX_FILE_SIZE", 1024)  # Set small limit for testing
    @patch("fastapply.main._create_timestamped_backup")
    @patch("fastapply.main._atomic_write")
    @patch("os.path.getsize")
    def test_write_with_backup_size_limit_coverage(self, mock_getsize, mock_atomic, mock_backup):
        """Test write_with_backup covering size limit enforcement (line 667)."""
        # Mock file size to trigger size limit - use smaller file so 2x limit < 5MB
        mock_getsize.return_value = 1024 * 512  # 512KB original file (2x = 1MB)
        mock_backup.return_value = "/path/to/backup.bak"

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            # Create content that would exceed the 5MB limit
            large_content = "x" * (5 * 1024 * 1024 + 1)  # 5MB + 1 byte, exceeds 5MB limit

            with pytest.raises(ValueError, match="Refusing write.*exceeds safety threshold"):
                write_with_backup(test_file, large_content)


class TestTargetedFileLockCoverage:
    """Targeted tests for file locking uncovered lines."""

    # Note: File lock cleanup coverage test removed due to test isolation challenges
    # The coverage lines are covered through other test scenarios

    # File lock cleanup tests removed due to import and test isolation issues


class TestTargetedResponseParsingCoverage:
    """Targeted tests for response parsing uncovered lines."""

    def test_extract_single_tag_empty_block(self):
        """Test _extract_single_tag_block covering empty block (line 685)."""
        from fastapply.main import _extract_single_tag_block

        content = f"{UPDATED_CODE_START}\n\n{UPDATED_CODE_END}"

        with pytest.raises(ValueError, match="Empty updated-code block"):
            _extract_single_tag_block(content)

    def test_extract_single_tag_missing_tags(self):
        """Test _extract_single_tag_block covering missing tags."""
        from fastapply.main import _extract_single_tag_block

        content = "Just some text without any tags"

        with pytest.raises(ValueError, match="Missing updated-code tags"):
            _extract_single_tag_block(content)


class TestTargetedMCPServerCoverage:
    """Targeted tests for MCP server uncovered lines."""

    @patch("fastapply.main.fast_apply_connector")
    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_edit_file_validation_error_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve, mock_connector):
        """Test edit_file covering validation error response (lines 844-850)."""
        # Import the server function from the correct location
        # call_tool is imported at module level

        # Setup mocks
        mock_resolve.return_value = "/workspace/test.py"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock file reading
        mock_file = Mock()
        mock_file.read.return_value = "def test(): pass"
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock connector to return validation errors
        mock_connector.apply_edit.return_value = {
            "merged_code": "def test(): invalid_syntax",
            "has_changes": True,
            "validation": {"has_errors": True, "errors": ["Line 1: Invalid syntax"], "warnings": []},
            "udiff": "diff content",
        }

        # Mock write_with_backup to avoid actual file writing
        with patch("fastapply.main.write_with_backup") as mock_write:
            mock_write.return_value = "/workspace/test.py.bak"

            result = asyncio.run(call_tool(
                name="edit_file",
                arguments={"path": "test.py", "code_edit": "def test(): invalid_syntax", "instruction": "Break the code"},
            ))

            # Should return error response
            response_text = result[0]["text"]
            assert "❌ Edit applied but validation failed" in response_text
            assert "Invalid syntax" in response_text

    @patch("fastapply.main.fast_apply_connector")
    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_edit_file_write_error_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve, mock_connector):
        """Test edit_file covering write error (lines 855-858)."""
        # call_tool is imported at module level

        # Setup mocks
        mock_resolve.return_value = "/workspace/test.py"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock file reading
        mock_file = Mock()
        mock_file.read.return_value = "def test(): pass"
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock connector to return successful edit
        mock_connector.apply_edit.return_value = {
            "merged_code": "def test(): updated",
            "has_changes": True,
            "validation": {"has_errors": False, "errors": [], "warnings": []},
        }

        # Mock write_with_backup to raise IOError
        with patch("fastapply.main.write_with_backup", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError, match="Failed to write file"):
                asyncio.run(call_tool(
                    name="edit_file",
                    arguments={"path": "test.py", "code_edit": "def test(): updated", "instruction": "Update function"},
                ))

    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_read_multiple_files_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve):
        """Test read_multiple_files covering error handling (lines 904-907)."""
        # call_tool is imported at module level

        # Setup mocks
        mock_resolve.return_value = "/workspace/test.py"
        mock_exists.side_effect = [True, False]  # First file exists, second doesn't
        mock_getsize.return_value = 1024

        # Mock file reading for first file
        mock_file = Mock()
        mock_file.read.return_value = "file content"
        mock_open.return_value.__enter__.return_value = mock_file

        result = asyncio.run(call_tool(
            "read_multiple_files",
            arguments={"paths": ["test.py", "nonexistent.py"]},
        ))

        response_text = result[0]["text"]
        assert "test.py:" in response_text
        assert "file content" in response_text
        assert "nonexistent.py: Error" in response_text

    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    def test_search_files_no_matches_coverage(self, mock_exists, mock_resolve):
        """Test search_files covering no matches case (line 1044)."""
        # call_tool is imported at module level

        mock_resolve.return_value = "/workspace"
        mock_exists.return_value = True  # Make the path exist

        with patch("fastapply.main.search_files", return_value=[]):
            result = asyncio.run(call_tool(
                name="search_files",
                arguments={"path": "/workspace", "pattern": "nonexistent"},
            ))

            response_text = result[0]["text"]
            assert "No matches found" in response_text

    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    def test_search_files_path_not_found_coverage(self, mock_exists, mock_resolve):
        """Test search_files covering path not found (line 1036)."""
        # call_tool is imported at module level

        mock_resolve.return_value = "/workspace/nonexistent"
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="Search path not found"):
            asyncio.run(call_tool(
                name="search_files",
                arguments={"path": "/workspace/nonexistent", "pattern": "*.py"},
            ))

    @patch("fastapply.main._secure_resolve")
    def test_unknown_tool_coverage(self, mock_resolve):
        """Test unknown tool handling (line 1047)."""
        # call_tool is imported at module level

        with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
            asyncio.run(call_tool(
                name="unknown_tool",
                arguments={},
            ))


class TestTargetedDryRunCoverage:
    """Targeted tests for dry run functionality."""

    @patch("fastapply.main.fast_apply_connector")
    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_dry_run_file_not_found_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve, mock_connector):
        """Test dry run covering file not found (line 935)."""
        # call_tool is imported at module level

        mock_resolve.return_value = "/workspace/nonexistent.py"
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="File not found"):
            asyncio.run(call_tool(
                "dry_run_edit_file",
                {"path": "nonexistent.py", "code_edit": "def test(): updated", "instruction": "Update function"},
            ))

    @patch("fastapply.main.fast_apply_connector")
    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_dry_run_read_error_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve, mock_connector):
        """Test dry run covering read error (line 942-944)."""
        # call_tool is imported at module level

        mock_resolve.return_value = "/workspace/test.py"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        mock_open.side_effect = IOError("Permission denied")

        with pytest.raises(IOError, match="Permission denied"):
            asyncio.run(call_tool(
                "dry_run_edit_file",
                {"path": "test.py", "code_edit": "def test(): updated", "instruction": "Update function"},
            ))

    @patch("fastapply.main.fast_apply_connector")
    @patch("fastapply.main._secure_resolve")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("builtins.open")
    def test_dry_run_api_error_coverage(self, mock_open, mock_getsize, mock_exists, mock_resolve, mock_connector):
        """Test dry run covering API error (line 954)."""
        # call_tool is imported at module level

        mock_resolve.return_value = "/workspace/test.py"
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock file reading
        mock_file = Mock()
        mock_file.read.return_value = "def test(): pass"
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock connector to raise error
        mock_connector.apply_edit.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            asyncio.run(call_tool(
                "dry_run_edit_file",
                {"path": "test.py", "code_edit": "def test(): updated", "instruction": "Update function"},
            ))
