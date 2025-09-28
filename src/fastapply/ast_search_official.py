"""
Official ast-grep integration following the reference implementation.

This module provides ast-grep functionality using command-line execution
to match the official ast-grep MCP server patterns and behavior.
"""

import json
import os
import subprocess
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


class AstGrepError(Exception):
    """Base exception for ast-grep operations."""

    pass


def get_supported_languages() -> List[str]:
    """Get list of supported languages from ast-grep."""
    try:
        run_ast_grep(["--help"])
        # For now, return common languages. In real implementation,
        # this would parse ast-grep output for supported languages
        return [
            "bash",
            "c",
            "cpp",
            "csharp",
            "css",
            "dart",
            "go",
            "html",
            "java",
            "javascript",
            "json",
            "kotlin",
            "lua",
            "php",
            "python",
            "ruby",
            "rust",
            "scala",
            "swift",
            "typescript",
        ]
    except Exception:
        logger.warning("Failed to get supported languages, using default list")
        return ["python", "javascript", "typescript", "json"]


def run_command(command: List[str], cwd: Optional[str] = None, input_text: Optional[str] = None) -> str:
    """Run a shell command and return its output."""
    try:
        logger.debug("Running command", command=command, cwd=cwd)
        result = subprocess.run(command, cwd=cwd, input=input_text, text=True, capture_output=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Command failed", command=command, error=e.stderr)
        raise AstGrepError(f"Command failed: {e.stderr}")
    except FileNotFoundError:
        raise AstGrepError(f"Command not found: {command[0]}")


def run_ast_grep(args: List[str], cwd: Optional[str] = None, input_text: Optional[str] = None) -> str:
    """Run ast-grep command with given arguments."""
    command = ["ast-grep"] + args
    return run_command(command, cwd=cwd, input_text=input_text)


def format_matches_as_text(matches_json: str) -> str:
    """Format JSON matches as readable text."""
    try:
        matches = json.loads(matches_json)
        if not matches:
            return "No matches found."

        formatted_lines = []
        for match in matches:
            file_path = match.get("file", "unknown")
            line = match.get("line", 0)
            text = match.get("text", "").strip()

            formatted_lines.append(f"{file_path}:{line}: {text}")

        return "\n".join(formatted_lines)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to format matches", error=str(e))
        return matches_json


def dump_syntax_tree(code: str, language: str, dump_format: str = "pattern") -> str:
    """
    Dump syntax tree for the given code.

    Args:
        code: Source code to analyze
        language: Programming language
        dump_format: Output format ("pattern", "cst", "ast")

    Returns:
        Syntax tree representation
    """
    try:
        if dump_format not in ["pattern", "cst", "ast"]:
            raise AstGrepError(f"Invalid dump format: {dump_format}. Use 'pattern', 'cst', or 'ast'")

        # Use a simple pattern for debugging - ast-grep needs a pattern to show debug output
        args = ["run", "--pattern", "$A", "--lang", language, f"--debug-query={dump_format}", "--stdin"]
        output = run_ast_grep(args, input_text=code)
        return output.strip()

    except Exception as e:
        logger.error("Failed to dump syntax tree", error=str(e))
        raise AstGrepError(f"Syntax tree dump failed: {e}")


def test_match_code_rule(code: str, rule_yaml: str) -> str:
    """
    Test a YAML rule against code.

    Args:
        code: Source code to test
        rule_yaml: YAML rule definition

    Returns:
        JSON string with match results
    """
    try:
        # Create temporary rule file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as rule_file:
            rule_file.write(rule_yaml)
            rule_file.flush()

            try:
                args = ["scan", "--rule", rule_file.name, "--json", "--stdin"]
                output = run_ast_grep(args, input_text=code)
                return output.strip() or "[]"
            finally:
                os.unlink(rule_file.name)

    except Exception as e:
        logger.error("Failed to test rule", error=str(e))
        raise AstGrepError(f"Rule testing failed: {e}")


def find_code(pattern: str, language: str, path: str = ".", output_format: str = "json", max_results: Optional[int] = None) -> str:
    """
    Find code using ast-grep patterns.

    Args:
        pattern: Search pattern
        language: Programming language
        path: Directory to search
        output_format: Output format ("json" or "text")
        max_results: Maximum number of results

    Returns:
        Search results in requested format
    """
    try:
        args = ["run", "--pattern", pattern, "--lang", language]

        if output_format == "json":
            args.append("--json")

        # Add path as positional argument
        args.append(path)

        output = run_ast_grep(args, cwd=".")

        if not output.strip():
            return "[]" if output_format == "json" else "No matches found."

        if max_results and output_format == "json":
            try:
                matches = json.loads(output)
                if isinstance(matches, list) and len(matches) > max_results:
                    matches = matches[:max_results]
                    output = json.dumps(matches, indent=2)
            except json.JSONDecodeError:
                pass

        return output.strip() if output_format == "json" else format_matches_as_text(output)

    except Exception as e:
        logger.error("Failed to find code", error=str(e))
        raise AstGrepError(f"Code search failed: {e}")


def find_code_by_rule(rule_yaml: str, path: str = ".", output_format: str = "json", max_results: Optional[int] = None) -> str:
    """
    Find code using YAML rules.

    Args:
        rule_yaml: YAML rule definition
        path: Directory to search
        output_format: Output format ("json" or "text")
        max_results: Maximum number of results

    Returns:
        Search results in requested format
    """
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as rule_file:
            rule_file.write(rule_yaml)
            rule_file.flush()

            try:
                args = ["scan", "--rule", rule_file.name]

                if output_format == "json":
                    args.append("--json")

                if path != ".":
                    args.append(path)

                output = run_ast_grep(args, cwd=path if os.path.isdir(path) else ".")

                if not output.strip():
                    return "[]" if output_format == "json" else "No matches found."

                if max_results and output_format == "json":
                    try:
                        matches = json.loads(output)
                        if isinstance(matches, list) and len(matches) > max_results:
                            matches = matches[:max_results]
                            output = json.dumps(matches, indent=2)
                    except json.JSONDecodeError:
                        pass

                return output.strip() if output_format == "json" else format_matches_as_text(output)

            finally:
                os.unlink(rule_file.name)

    except Exception as e:
        logger.error("Failed to find code by rule", error=str(e))
        raise AstGrepError(f"Rule-based search failed: {e}")


def is_ast_grep_available() -> bool:
    """Check if ast-grep command is available."""
    try:
        run_command(["ast-grep", "--version"])
        return True
    except (AstGrepError, FileNotFoundError):
        logger.warning("ast-grep command not available")
        return False


# Backward compatibility - keep existing function names
def search_code_patterns(pattern: str, language: str, path: str, exclude_patterns: Optional[List[str]] = None):
    """Backward compatibility wrapper for find_code."""
    try:
        result_json = find_code(pattern, language, path, "json")
        matches = json.loads(result_json)

        # Convert to our legacy format for compatibility
        from ast_search import PatternSearchResult

        results = []
        for match in matches:
            result = PatternSearchResult(
                file_path=match.get("file", ""),
                line=match.get("line", 0),
                column=match.get("column", 0),
                text=match.get("text", ""),
                matches=match.get("metaVars", {}),
            )
            results.append(result)
        return results
    except Exception as e:
        from ast_search import AstSearchError

        raise AstSearchError(f"Pattern search failed: {e}")
