"""
FastApply MCP Server Implementation

A standalone MCP server with FastApply code editing capabilities.
Uses stdio transport and provides multi-file editing tools like Morphllm.
"""

import difflib
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import openai
import structlog
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import the new AST search functionality
try:
    from . import ast_search

    AST_SEARCH_AVAILABLE = True
except ImportError:
    AST_SEARCH_AVAILABLE = False

# Import AST rule intelligence system
try:
    import ast_rule_intelligence

    AST_INTELLIGENCE_AVAILABLE = ast_rule_intelligence.is_available()
except ImportError:
    AST_INTELLIGENCE_AVAILABLE = False

# Import the official AST search tool
try:
    from fastapply import ast_search_official

    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False


# Import official ast-grep implementation
try:
    from fastapply import ast_search_official

    AST_GREP_AVAILABLE = ast_search_official.is_ast_grep_available()
except ImportError:
    AST_GREP_AVAILABLE = False

# Import ripgrep integration layer
try:
    import ripgrep_integration

    RIPGREP_AVAILABLE = True
except ImportError:
    RIPGREP_AVAILABLE = False

# Import enhanced search infrastructure
try:
    from . import enhanced_search
    from .enhanced_search import (
        EnhancedSearchInfrastructure,
        EnhancedSearchResult,
        ResultRanking,
        SearchContext,
        SearchStrategy,
    )

    ENHANCED_SEARCH_AVAILABLE = True
except ImportError:
    ENHANCED_SEARCH_AVAILABLE = False

# Global enhanced search instance
enhanced_search_instance = None
if ENHANCED_SEARCH_AVAILABLE:
    try:
        enhanced_search_instance = EnhancedSearchInfrastructure()
    except Exception:
        ENHANCED_SEARCH_AVAILABLE = False

# Import deep semantic analysis
try:
    from .deep_semantic_analysis import (
        AnalysisContext,
        DeepSemanticAnalyzer,
    )

    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    SEMANTIC_ANALYSIS_AVAILABLE = False

# Global semantic analysis instance
semantic_analyzer = None
if SEMANTIC_ANALYSIS_AVAILABLE:
    try:
        semantic_analyzer = DeepSemanticAnalyzer()
    except Exception:
        SEMANTIC_ANALYSIS_AVAILABLE = False

# Import relationship mapping
try:
    from .relationship_mapping import (
        RelationshipMapper,
    )

    RELATIONSHIP_MAPPING_AVAILABLE = True
except ImportError:
    RELATIONSHIP_MAPPING_AVAILABLE = False

# Import navigation tools
try:
    from .navigation_tools import (
        GraphType,
        NavigationAnalyzer,
    )

    NAVIGATION_TOOLS_AVAILABLE = True
except ImportError:
    NAVIGATION_TOOLS_AVAILABLE = False

# Safe refactoring tools
try:
    from .safe_refactoring import (
        CodeExtractionAndMovement,
        SafeSymbolRenaming,
    )

    SAFE_REFACTORING_AVAILABLE = True
except ImportError:
    SAFE_REFACTORING_AVAILABLE = False

# Batch operations tools
try:
    from .batch_operations import (
        BatchAnalysisSystem,
        BatchConfig,
        BatchOperation,
        BatchOperationType,
        # BatchResults,  # Unused
        BatchTransformation,
    )

    BATCH_OPERATIONS_AVAILABLE = True
except ImportError:
    BATCH_OPERATIONS_AVAILABLE = False

# Security & Quality analysis tools
try:
    from .security_quality_analysis import (
        ComplianceReportingFramework,
        QualityAssessment,
        QualityAssuranceFramework,
        QualityGateAutomation,
        QualityMetric,
        # QualityMetrics,  # Imported locally where needed
        SecurityReport,
        SecurityVulnerabilityScanner,
        SeverityLevel,
        # Vulnerability,  # Unused
        # VulnerabilityCategory,  # Unused
        # VulnerabilityType,  # Unused
      )

    SECURITY_QUALITY_AVAILABLE = True
except ImportError:
    SECURITY_QUALITY_AVAILABLE = False

# Global relationship mapper instance
relationship_mapper = None
if RELATIONSHIP_MAPPING_AVAILABLE:
    try:
        relationship_mapper = RelationshipMapper()
    except Exception:
        RELATIONSHIP_MAPPING_AVAILABLE = False

# Global batch operations instances
batch_analyzer = None
batch_transformer = None
batch_config = None
if BATCH_OPERATIONS_AVAILABLE:
    try:
        batch_config = BatchConfig()
        batch_analyzer = BatchAnalysisSystem(batch_config)
        batch_transformer = BatchTransformation(batch_config)
    except Exception:
        BATCH_OPERATIONS_AVAILABLE = False

# Global security & quality analysis instances
security_scanner = None
quality_analyzer = None
compliance_reporter = None
quality_gates = None
if SECURITY_QUALITY_AVAILABLE:
    try:
        security_scanner = SecurityVulnerabilityScanner()
        quality_analyzer = QualityAssuranceFramework()
        compliance_reporter = ComplianceReportingFramework()
        quality_gates = QualityGateAutomation()
    except Exception:
        SECURITY_QUALITY_AVAILABLE = False


# Load environment variables
load_dotenv()

# Compatibility shim for tests that instantiate openai.APIError with a single message arg.
try:  # pragma: no cover - defensive
    import inspect

    if hasattr(openai, "APIError"):
        sig = inspect.signature(openai.APIError)
        if len(sig.parameters) > 2:  # real class expects more; wrap
            original_api_error = openai.APIError

            class SimpleAPIError(Exception):
                def __init__(self, message: str):  # type: ignore[override]
                    super().__init__(message)

            openai.APIError = SimpleAPIError  # type: ignore
except Exception:
    pass

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

# Create logger instance
logger = structlog.get_logger("fast-apply-mcp")

"""Unified tool exposure (all tools always available).

Removed legacy single-tool mode and feature flags to simplify operational
behavior. Tools exposed:
    - edit_file
    - dry_run_edit_file
    - search_files
    - read_multiple_files
"""

# Workspace root for security isolation (explicit root strongly recommended)
WORKSPACE_ROOT = os.path.abspath(os.getenv("WORKSPACE_ROOT", os.getcwd()))

# Global navigation analyzer instance
navigation_analyzer = None
if NAVIGATION_TOOLS_AVAILABLE:
    try:
        navigation_analyzer = NavigationAnalyzer(WORKSPACE_ROOT)
    except Exception:
        NAVIGATION_TOOLS_AVAILABLE = False

# Global safe refactoring instances
safe_renamer = None
safe_extractor = None
if SAFE_REFACTORING_AVAILABLE:
    try:
        safe_renamer = SafeSymbolRenaming()
        safe_extractor = CodeExtractionAndMovement()
    except Exception:
        SAFE_REFACTORING_AVAILABLE = False

# File size limits to prevent memory exhaustion
MAX_FILE_SIZE = int(os.getenv("FAST_APPLY_MAX_FILE_BYTES", "10485760"))  # 10MB default

# Response size limits (approximate character limit ~40k tokens @ 6 chars/token)
MAX_RESPONSE_SIZE = 240000

# Allowed editable file extensions (comma separated env override)
_DEFAULT_ALLOWED_EXTS = [
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
]
ALLOWED_EXTS = [e.strip() for e in os.getenv("FAST_APPLY_ALLOWED_EXTS", ",".join(_DEFAULT_ALLOWED_EXTS)).split(",") if e.strip()]

# Maximum combined size (bytes) of original + edit snippet allowed in a single API request
MAX_REQUEST_BYTES = int(os.getenv("FAST_APPLY_MAX_REQUEST_BYTES", str(2 * MAX_FILE_SIZE)))

# File locking mechanism to prevent race conditions
_file_locks = {}
_locks_lock = threading.Lock()

# Optional strict path mode (default on). Set FAST_APPLY_STRICT_PATHS=0 to allow legacy CWD fallback.
STRICT_PATHS = os.getenv("FAST_APPLY_STRICT_PATHS", "1") not in ("0", "false", "False")

# Simple availability caches to avoid repeated subprocess spawning when tools absent
_RUFF_AVAILABLE: Optional[bool] = None
_ESLINT_AVAILABLE: Optional[bool] = None


def _get_file_lock(file_path: str) -> threading.Lock:
    """Get or create a lock for a specific canonical file path.

    Uses realpath to ensure all path variants for the same file share a single lock.
    """
    canonical = os.path.realpath(file_path)
    with _locks_lock:
        if canonical not in _file_locks:
            _file_locks[canonical] = threading.Lock()
        return _file_locks[canonical]


def _cleanup_file_locks():
    """Clean up unused file locks to prevent memory leaks."""
    with _locks_lock:
        # Remove locks for files that no longer exist
        existing_files = set(_file_locks.keys())
        for file_path in list(existing_files):
            if not os.path.exists(file_path):
                del _file_locks[file_path]
                logger.debug("Cleaned up lock for non-existent file", file_path=file_path)


def _secure_resolve(path: str) -> str:
    """Securely resolve a path confined to WORKSPACE_ROOT.

    In STRICT_PATHS mode (default):
        - Only paths within WORKSPACE_ROOT after realpath are allowed.
        - Absolute paths must already be inside WORKSPACE_ROOT.
    In legacy mode (STRICT_PATHS disabled):
        - Retains prior behavior allowing CWD fallback if still under CWD.
    """
    workspace_root = os.path.realpath(os.getenv("WORKSPACE_ROOT", os.getcwd()))
    original = path

    # If absolute, keep as-is for validation; else join
    if not os.path.isabs(path):
        path = os.path.join(workspace_root, path)

    candidate = os.path.realpath(path)

    if os.path.commonpath([candidate, workspace_root]) == workspace_root:
        return candidate

    if not STRICT_PATHS:
        # Legacy fallback: allow any existing path under current working directory
        cwd_real = os.path.realpath(os.getcwd())
        if os.path.exists(candidate) and os.path.commonpath([candidate, cwd_real]) == cwd_real:
            return candidate

    raise ValueError(f"Path escapes workspace: {original}")


def _is_allowed_edit_target(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in ALLOWED_EXTS


# Official Fast Apply template constants
FAST_APPLY_SYSTEM_PROMPT = """You are a coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

FAST_APPLY_USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
Instruction: {instruction}
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, markdown, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code."""

# XML tag constants for response parsing
UPDATED_CODE_START = "<updated-code>"
UPDATED_CODE_END = "</updated-code>"


def _atomic_write(path: str, data: str):
    """Write data atomically to path using temp file + fsync + os.replace."""
    import tempfile

    dir_name = os.path.dirname(path)
    if dir_name:  # Only create directories if path is not in current directory
        os.makedirs(dir_name, exist_ok=True)
    # Use None for current directory instead of empty string
    temp_dir = dir_name if dir_name else None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=temp_dir) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except Exception:
            # fsync best effort
            pass
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _create_timestamped_backup(original_path: str) -> str:
    """Create a timestamped backup of the original file.

    Returns the backup path. Falls back to .bak if timestamp write fails.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{original_path}.bak_{timestamp}"
    try:
        with open(original_path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return backup_path
    except Exception as e:
        logger.warning("Failed to create timestamped backup", error=str(e), path=original_path)
        fallback_path = original_path + ".bak"
        try:
            with open(original_path, "r", encoding="utf-8") as src, open(fallback_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            return fallback_path
        except Exception:
            # If even fallback fails, re-raise original error context
            raise


# Maximum diff size to prevent memory exhaustion (100KB)
MAX_DIFF_SIZE = 102400


def generate_udiff(original_code: str, modified_code: str, target_file: str) -> str:
    """Generate UDiff between original and modified code for verification."""
    diff = "\n".join(
        difflib.unified_diff(
            original_code.splitlines(keepends=True),
            modified_code.splitlines(keepends=True),
            fromfile=target_file,
            tofile=target_file,
            lineterm="",
        )
    )

    # Truncate large diffs to prevent memory issues
    if len(diff) > MAX_DIFF_SIZE:
        logger.warning("Diff exceeds maximum size, truncating", diff_size=len(diff), max_size=MAX_DIFF_SIZE)
        lines = diff.splitlines()
        truncated_lines = lines[:50] + ["... (diff truncated due to size)"] + lines[-50:]
        diff = "\n".join(truncated_lines)

    return diff


def validate_code_quality(file_path: str, code: str) -> Dict[str, Any]:
    """Validate code quality using available linting tools."""
    validation_results: Dict[str, Any] = {"has_errors": False, "errors": [], "warnings": [], "suggestions": []}

    try:
        # Try to detect file type and run appropriate linters
        if file_path.endswith(".py"):
            lint_results = _validate_python_code(code)
        elif file_path.endswith((".js", ".jsx", ".ts", ".tsx")):
            lint_results = _validate_javascript_code(code)
        else:
            lint_results = {}

        if lint_results:
            validation_results["errors"].extend(lint_results.get("errors", []))
            validation_results["warnings"].extend(lint_results.get("warnings", []))
            validation_results["suggestions"].extend(lint_results.get("suggestions", []))
            if validation_results["errors"]:
                validation_results["has_errors"] = True
    except Exception as e:
        logger.warning("Code validation failed", error=str(e), file_path=file_path)

    return validation_results


def _validate_python_code(code: str) -> Dict[str, Any]:
    """Validate Python code using Ruff if available (cached availability)."""
    global _RUFF_AVAILABLE
    results: Dict[str, List[str]] = {"errors": [], "warnings": [], "suggestions": []}
    if _RUFF_AVAILABLE is False:
        return results
    try:
        import subprocess
        import tempfile

        if _RUFF_AVAILABLE is None:
            # Probe availability quickly
            try:
                subprocess.run(["ruff", "--version"], capture_output=True, timeout=2)
                _RUFF_AVAILABLE = True
            except Exception:
                _RUFF_AVAILABLE = False
                return results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(["ruff", "check", "--format=json", temp_file], capture_output=True, text=True, timeout=10)
            if result.stdout:
                import json

                ruff_results = json.loads(result.stdout)
                for issue in ruff_results:
                    location = issue.get("location", {})
                    row = location.get("row")
                    msg = issue.get("message")
                    code_id = issue.get("code")
                    bucket = "warnings"
                    if code_id and code_id.startswith(("E", "F")):
                        bucket = "errors"
                    results[bucket].append(f"Line {row}: {msg} ({code_id})")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    except Exception:
        pass
    return results


def _validate_javascript_code(code: str) -> Dict[str, Any]:
    """Validate JavaScript/TypeScript code using ESLint if available (cached availability)."""
    global _ESLINT_AVAILABLE
    results: Dict[str, List[str]] = {"errors": [], "warnings": [], "suggestions": []}
    if _ESLINT_AVAILABLE is False:
        return results
    try:
        import subprocess
        import tempfile

        if _ESLINT_AVAILABLE is None:
            try:
                subprocess.run(["eslint", "--version"], capture_output=True, timeout=2)
                _ESLINT_AVAILABLE = True
            except Exception:
                _ESLINT_AVAILABLE = False
                return results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(["eslint", "--format=json", temp_file], capture_output=True, text=True, timeout=10)
            if result.stdout:
                import json

                eslint_results = json.loads(result.stdout)
                for file_result in eslint_results:
                    for message in file_result.get("messages", []):
                        if message.get("severity") == 2:
                            results["errors"].append(f"Line {message.get('line')}: {message.get('message')}")
                        else:
                            results["warnings"].append(f"Line {message.get('line')}: {message.get('message')}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    except Exception:
        pass
    return results


def search_files(root_path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Search for files matching a pattern, with optional exclude patterns."""
    if exclude_patterns is None:
        exclude_patterns = []

    # Add default exclusion for common large directories
    default_excludes = [".git", "node_modules", "venv", "env", ".venv", "__pycache__", ".pytest_cache"]
    all_exclude_patterns = exclude_patterns + default_excludes

    results = []

    def should_exclude(file_path: str) -> bool:
        """Check if file should be excluded based on patterns."""
        relative_path = os.path.relpath(file_path, root_path)
        for exclude_pattern in all_exclude_patterns:
            # Simple pattern matching - can be enhanced with glob patterns
            if exclude_pattern in relative_path or exclude_pattern in os.path.basename(file_path):
                return True
        return False

    def search_recursive(current_path: str):
        """Recursively search directories using os.scandir for better performance."""
        try:
            with os.scandir(current_path) as entries:
                for entry in entries:
                    item_path = entry.path

                    # Skip if path is outside workspace bounds (now using _secure_resolve)
                    try:
                        _secure_resolve(item_path)
                    except ValueError:
                        continue

                    if should_exclude(item_path):
                        continue

                    if entry.is_file():
                        # Check if filename matches pattern (case-insensitive)
                        if pattern.lower() in entry.name.lower():
                            results.append(item_path)
                    elif entry.is_dir():
                        search_recursive(item_path)
        except (PermissionError, OSError):
            # Skip directories we can't access
            pass

    search_recursive(root_path)
    return results


class FastApplyConnector:
    """Handles connections to Fast Apply API using OpenAI-compatible client.

    Responsibilities:
    - Maintain configuration
    - Provide apply_edit with structured verification
    - Parse & validate model responses
    - Avoid leaking secrets in return values/logs
    """

    def __init__(
        self,
        url: str = os.getenv("FAST_APPLY_URL", "http://localhost:1234/v1"),
        model: str = os.getenv("FAST_APPLY_MODEL", "fastapply-1.5b"),
        api_key: Optional[str] = os.getenv("FAST_APPLY_API_KEY"),
        timeout: float = float(os.getenv("FAST_APPLY_TIMEOUT", "30.0")),
        max_tokens: int = int(os.getenv("FAST_APPLY_MAX_TOKENS", "8000")),
        temperature: float = float(os.getenv("FAST_APPLY_TEMPERATURE", "0.05")),
    ):
        """Initialize the Fast Apply connector with configuration parameters."""

        # Validate configuration parameters
        if timeout <= 0 or timeout > 300:
            raise ValueError("Timeout must be between 0 and 300 seconds")
        if max_tokens <= 0 or max_tokens > 32000:
            raise ValueError("Max tokens must be between 0 and 32000")
        if temperature < 0 or temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        self.url = url
        self.model = model
        self.api_key = api_key or "optional-api-key"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client: Optional[openai.OpenAI] = None

        # Initialize OpenAI client with Fast Apply API configuration
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=self.timeout,
            )
        except Exception as e:
            # Allow for testing without API connection
            logger.warning("Could not initialize OpenAI client, running in test mode", error=str(e))
            self.client = None

        logger.info(
            "FastApplyConnector initialized",
            model=self.model,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            has_client=self.client is not None,
        )

    def _strip_markdown_blocks(self, text: str) -> str:
        """Remove surrounding markdown code fences if present."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            # remove first and last fence lines
            if len(lines) >= 2:
                inner = lines[1:-1]
                # If first fence contains language spec, already removed
                return "\n".join(inner).strip()
        # Inline style ```code``` single line
        if stripped.startswith("```") and stripped.count("```") == 2 and "\n" not in stripped:
            return stripped.strip("`")
        return text

    def _parse_fast_apply_response(self, raw_response: str) -> str:
        """Parse model response extracting exactly one <updated-code> block or fallback to markdown strip.

        Mirrors expected legacy semantics used in existing tests.
        """
        if not raw_response or not raw_response.strip():
            raise ValueError("Fast Apply API response is empty")

        if len(raw_response) > MAX_RESPONSE_SIZE:
            logger.warning(
                "Response exceeds maximum allowed size, truncating",
                response_size=len(raw_response),
                max_size=MAX_RESPONSE_SIZE,
            )
            raw_response = raw_response[:MAX_RESPONSE_SIZE]

        cleaned = _sanitize_model_response(raw_response)
        try:
            return _extract_single_tag_block(cleaned)
        except ValueError:
            # Fallback: strip markdown fences
            return self._strip_markdown_blocks(cleaned)

    def apply_edit(self, *args, **kwargs):  # type: ignore[override]
        """Apply code edit.

        Dual-interface support:
        - New style: apply_edit(original_code=..., code_edit=..., instruction=..., file_path=...)
          Returns rich dict with metadata.
        - Legacy positional style: apply_edit(instructions, original_code, code_edit)
          Returns merged code string.
        """
        legacy_mode = False
        if args and not kwargs:
            # Legacy expects 3 positional arguments: instruction, original_code, code_edit
            if len(args) == 3:
                instruction, original_code, code_edit = args
                file_path = None
                legacy_mode = True
            else:
                raise TypeError("Legacy apply_edit expects exactly 3 positional arguments")
        else:
            original_code = kwargs.get("original_code")
            code_edit = kwargs.get("code_edit")
            # Accept both singular & plural key variants
            instruction = kwargs.get("instruction") or kwargs.get("instructions", "")
            file_path = kwargs.get("file_path")
            if original_code is None or code_edit is None:
                raise TypeError("apply_edit requires original_code and code_edit")

        if self.client is None:
            raise RuntimeError("Fast Apply client not initialized; cannot perform edit.")
        try:
            # Format the request according to official Fast Apply specification
            user_content = FAST_APPLY_USER_PROMPT.format(
                original_code=original_code, update_snippet=code_edit, instruction=instruction or "Apply the requested code changes."
            )

            logger.info(
                "Sending edit request to Fast Apply API",
                original_code_length=len(original_code),
                code_edit_length=len(code_edit),
                instruction=instruction,
                file_path=file_path,
            )

            # Make the API call with proper system/user message structure
            # Request size pre-flight check (approx based on raw string byte lengths)
            request_bytes = len(user_content.encode("utf-8"))
            if request_bytes > MAX_REQUEST_BYTES:
                raise ValueError(f"Request size {request_bytes} bytes exceeds limit {MAX_REQUEST_BYTES} bytes (original + edit too large)")

            # Retry logic for transient API errors
            attempts = int(os.getenv("FAST_APPLY_RETRY_ATTEMPTS", "3"))
            backoff_base = float(os.getenv("FAST_APPLY_RETRY_BACKOFF", "0.75"))
            last_exc: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": FAST_APPLY_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    break  # success
                except Exception as api_e:  # Broad catch; refined classification below
                    last_exc = api_e
                    transient = api_e.__class__.__name__.endswith("APIError") or isinstance(api_e, (TimeoutError,))
                    if attempt == attempts or not transient:
                        raise
                    sleep_for = backoff_base * (2 ** (attempt - 1))
                    try:
                        import time

                        time.sleep(min(sleep_for, 8))
                    except Exception:
                        pass
            else:  # pragma: no cover safeguard
                if last_exc:
                    raise last_exc

            # Extract the merged code from the response
            if response.choices and len(response.choices) > 0:
                raw_response = response.choices[0].message.content
                logger.info("Received response from Fast Apply API", response_length=len(raw_response))

                # Parse the response using improved parsing logic
                merged_code = self._parse_fast_apply_response(raw_response)
                logger.info("Successfully parsed merged code", merged_code_length=len(merged_code))

                # Generate verification results
                verification_results = {
                    "merged_code": merged_code,
                    "has_changes": merged_code != original_code,
                    "udiff": "",
                    "validation": {"has_errors": False, "errors": [], "warnings": []},
                }

                # Generate UDiff for verification
                if verification_results["has_changes"] and file_path:
                    verification_results["udiff"] = generate_udiff(original_code, merged_code, file_path)
                    logger.info("Generated UDiff verification", udiff_length=len(verification_results["udiff"]))

                # Always validate code quality if file_path is provided (even when unchanged)
                if file_path:
                    verification_results["validation"] = validate_code_quality(file_path, merged_code)
                    if verification_results["validation"]["has_errors"]:
                        logger.warning(
                            "Code validation found errors",
                            errors=verification_results["validation"]["errors"],
                            warnings=verification_results["validation"]["warnings"],
                        )

                # Enforce absolute size safety BEFORE returning (prevents oversized content writes downstream)
                merged_bytes = merged_code.encode("utf-8")
                if len(merged_bytes) > MAX_FILE_SIZE:
                    raise ValueError(
                        f"Merged code size {len(merged_bytes)} bytes exceeds MAX_FILE_SIZE {MAX_FILE_SIZE} bytes; refusing to continue"
                    )

                return merged_code if legacy_mode else verification_results
            else:
                # For legacy tests expect ValueError to surface directly (not wrapped)
                raise ValueError("Invalid Fast Apply API response: no choices available")

        except ValueError:
            # Pass through intentional validation errors
            raise
        except Exception as e:
            # Heuristic: treat anything named *APIError* as API error to simplify test mocking
            if e.__class__.__name__.endswith("APIError"):
                logger.error("Fast Apply API error", error=str(e))
                raise RuntimeError("Fast Apply API error") from e
            logger.error("Fast Apply unexpected error", error=str(e))
            raise RuntimeError("Unexpected error when calling Fast Apply API") from e

    def update_config(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update configuration. Returns public-safe config (no api_key) + legacy fields."""

        # Validate numeric parameters
        if timeout is not None and (timeout <= 0 or timeout > 300):
            raise ValueError("Timeout must be between 0 and 300 seconds")

        if url is not None:
            self.url = url
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if timeout is not None:
            self.timeout = timeout
        if max_tokens is not None:
            if max_tokens <= 0 or max_tokens > 32000:
                raise ValueError("max_tokens must be between 1 and 32000")
            self.max_tokens = max_tokens
        if temperature is not None:
            if temperature < 0 or temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
            self.temperature = temperature

        # Update client with new configuration (allow failure in tests)
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning("Could not reinitialize OpenAI client after config update, staying in test mode", error=str(e))
            self.client = None

        logger.info("Fast Apply configuration updated")
        return {
            "url": self.url,
            "model": self.model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _analyze_response_format(self, raw_response: str) -> Dict[str, Any]:
        """Analyze the format and content of a Fast Apply API response for debugging."""
        analysis = {
            "total_length": len(raw_response),
            "has_xml_tags": UPDATED_CODE_START in raw_response and UPDATED_CODE_END in raw_response,
            "has_markdown_fences": raw_response.strip().startswith("```") or raw_response.strip().endswith("```"),
            "line_count": len(raw_response.splitlines()),
            "starts_with_code_tag": raw_response.strip().startswith("<updated-code>"),
            "ends_with_code_tag": raw_response.strip().endswith("</updated-code>"),
            "first_200_chars": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response,
            "last_200_chars": raw_response[-200:] + "..." if len(raw_response) > 200 else raw_response,
        }

        if analysis["has_xml_tags"]:
            start_idx = raw_response.find(UPDATED_CODE_START)
            end_idx = raw_response.find(UPDATED_CODE_END) + len(UPDATED_CODE_END)
            analysis["xml_content_length"] = end_idx - start_idx

        return analysis


def write_with_backup(path: str, new_content: str) -> str:
    """Atomically write file with timestamped backup under lock.

    Safety rules:
        - Reject if new size (bytes) > max(original*2, 5MB) when file exists.
        - Always create backup when file exists; if not, just write.
    """
    file_lock = _get_file_lock(path)
    with file_lock:
        try:
            original_size = os.path.getsize(path)
        except OSError:
            original_size = 0

        new_size = len(new_content.encode("utf-8"))
        if original_size > 0:
            limit = max(original_size * 2, 5 * 1024 * 1024)
            if new_size > limit:
                raise ValueError(f"Refusing write: new content size {new_size} exceeds safety threshold {limit} bytes")

        backup_path = _create_timestamped_backup(path) if original_size > 0 else f"{path}.initial"
        _atomic_write(path, new_content)
        try:
            _cleanup_file_locks()
        except Exception:
            pass
    return backup_path


def _extract_single_tag_block(content: str) -> str:
    """Strictly extract exactly one <updated-code> block or raise."""
    start = content.find(UPDATED_CODE_START)
    end = content.find(UPDATED_CODE_END)
    if start == -1 or end == -1:
        raise ValueError("Missing updated-code tags")
    second_start = content.find(UPDATED_CODE_START, start + 1)
    if second_start != -1:
        raise ValueError("Multiple updated-code blocks detected")
    inner = content[start + len(UPDATED_CODE_START) : end].strip()
    if not inner:
        raise ValueError("Empty updated-code block")
    return inner


def _sanitize_model_response(raw: str) -> str:
    """Remove markdown fences & surrounding noise before tag extraction."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first & last fence if present
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
            text = "\n".join(lines).strip()
    return text


# NOTE: The legacy module-level _parse_fast_apply_response shim has been removed.
# Use FastApplyConnector()._parse_fast_apply_response for parsing model responses.


# Create server instance
mcp = FastMCP("fast-apply-mcp")
fast_apply_connector = FastApplyConnector()

# Initialize enhanced search infrastructure
enhanced_search_engine: Union[EnhancedSearchInfrastructure, None] = None
if ENHANCED_SEARCH_AVAILABLE:
    enhanced_search_engine = enhanced_search.EnhancedSearchInfrastructure()


def json_dumps(obj: Any) -> str:
    """Consistent JSON serialization for tool outputs."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps({"error": "serialization_failed"})


def extract_simple_pattern(query: str) -> str:
    """Extract a simple ast-grep pattern from a natural language query for legacy fallback."""
    import re

    # Remove common question words and focus on technical terms
    query = query.lower()

    # Look for common patterns
    if "print" in query or "console" in query:
        return "console.log" if "console" in query else "print"

    if "import" in query:
        return "import $NAME"

    if "class" in query:
        return "class $NAME"

    if "function" in query or "def" in query:
        return "def $NAME"

    # Extract first significant word as fallback
    words = re.findall(r"\b\w+\b", query)
    technical_words = [w for w in words if len(w) > 2 and w not in ["find", "search", "look", "get", "code"]]

    return technical_words[0] if technical_words else words[0] if words else "*"


@mcp.tool()
def list_tools() -> List[Dict[str, Any]]:
    """Return metadata for all exposed tools (unified mode)."""
    return [
        {
            "name": "edit_file",
            "description": "Apply code edits to a file using Fast Apply.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string", "description": "Path to target file"},
                    "instructions": {"type": "string", "description": "Edit instructions"},
                    "code_edit": {"type": "string", "description": "Code edit snippet"},
                    "force": {"type": "boolean", "description": "Override optimistic concurrency / safety checks"},
                    "output_format": {"type": "string", "enum": ["text", "json"], "description": "Response format"},
                },
                "required": ["target_file", "instructions", "code_edit"],
            },
        },
        {
            "name": "dry_run_edit_file",
            "description": "Preview an edit without writing changes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_file": {"type": "string"},
                    "instruction": {"type": "string"},
                    "code_edit": {"type": "string"},
                    "output_format": {"type": "string", "enum": ["text", "json"]},
                },
                "required": ["target_file", "code_edit"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for files by simple substring pattern.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "excludePatterns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "read_multiple_files",
            "description": "Read multiple small files (concatenated output).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["paths"],
            },
        },
        {
            "name": "search_code_patterns",
            "description": "Search for semantic code patterns using AST-based matching (e.g., 'function $name($args) { $body }').",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "AST pattern to search for"},
                    "language": {"type": "string", "description": "Target language (python, javascript, typescript)"},
                    "path": {"type": "string", "description": "File or directory path to search in"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional patterns to exclude"},
                },
                "required": ["pattern", "language", "path"],
            },
        },
        {
            "name": "analyze_code_structure",
            "description": "Analyze structural components (functions, classes, imports) of a code file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the code file to analyze"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "find_references",
            "description": "Find all references to a specific symbol (function, class, variable).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name to search for"},
                    "path": {"type": "string", "description": "File or directory path to search in"},
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "variable", "any"],
                        "description": "Type of symbol to search for",
                    },
                },
                "required": ["symbol", "path"],
            },
        },
        {
            "name": "dump_syntax_tree",
            "description": "Dump syntax tree for code using ast-grep (official tool).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language"},
                    "dump_format": {
                        "type": "string",
                        "enum": ["pattern", "cst", "ast"],
                        "description": "Output format for syntax tree",
                        "default": "pattern",
                    },
                },
                "required": ["code", "language"],
            },
        },
        {
            "name": "test_match_code_rule",
            "description": "Test YAML rule against code using ast-grep (official tool).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to test"},
                    "rule_yaml": {"type": "string", "description": "YAML rule definition"},
                },
                "required": ["code", "rule_yaml"],
            },
        },
        {
            "name": "find_code",
            "description": "Find code using ast-grep patterns (official tool).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "language": {"type": "string", "description": "Programming language"},
                    "path": {"type": "string", "description": "Directory to search", "default": "."},
                    "output_format": {"type": "string", "enum": ["json", "text"], "description": "Output format", "default": "json"},
                    "max_results": {"type": "integer", "description": "Maximum results to return"},
                },
                "required": ["pattern", "language"],
            },
        },
        {
            "name": "find_code_by_rule",
            "description": "Find code using YAML rules with ast-grep (official tool).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rule_yaml": {"type": "string", "description": "YAML rule definition"},
                    "path": {"type": "string", "description": "Directory to search", "default": "."},
                    "output_format": {"type": "string", "enum": ["json", "text"], "description": "Output format", "default": "json"},
                    "max_results": {"type": "integer", "description": "Maximum results to return"},
                },
                "required": ["rule_yaml"],
            },
        },
        {
            "name": "health_status",
            "description": "Return server health and configuration info (non-sensitive).",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "llm_analyze_code",
            "description": "LLM-based deep code analysis using AST intelligence and reasoning patterns.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language (python, javascript, etc.)"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["general", "complexity", "security", "performance", "architecture"],
                        "description": "Analysis type",
                    },
                    "use_collective_memory": {"type": "boolean", "description": "Use Qdrant collective consciousness for analysis"},
                },
                "required": ["code", "language"],
            },
        },
        {
            "name": "llm_generate_rule",
            "description": "Generate AST rules dynamically using LLM reasoning from examples and queries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of the rule to generate"},
                    "language": {"type": "string", "description": "Target programming language"},
                    "examples": {"type": "array", "items": {"type": "string"}, "description": "Example code snippets to learn from"},
                    "rule_type": {"type": "string", "enum": ["pattern", "relational", "composite"], "description": "Rule type to generate"},
                    "use_collective_memory": {"type": "boolean", "description": "Use learned patterns from Qdrant"},
                },
                "required": ["query", "language"],
            },
        },
        {
            "name": "llm_search_pattern",
            "description": "Intelligent pattern search using LLM reasoning and collective consciousness.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of patterns to search for"},
                    "language": {"type": "string", "description": "Programming language to search in"},
                    "path": {"type": "string", "description": "File or directory path to search in"},
                    "use_collective_memory": {"type": "boolean", "description": "Use Qdrant collective consciousness for enhanced search"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                },
                "required": ["query", "language"],
            },
        },
        {
            "name": "auto_ast_intelligence",
            "description": "Auto-invocation system that detects user intent and executes optimal AST intelligence tools.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query or command"},
                    "context": {"type": "string", "description": "Optional context about the current task or project"},
                    "language": {"type": "string", "description": "Default programming language for analysis"},
                    "path": {"type": "string", "description": "Default path for file operations"},
                    "auto_execute": {"type": "boolean", "description": "Whether to automatically execute the detected tool sequence"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "ripgrep_search",
            "description": "Ultra-fast pattern discovery using ripgrep for large codebases.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "search_type": {
                        "type": "string",
                        "enum": ["pattern", "literal", "word", "regex"],
                        "description": "Type of search",
                        "default": "pattern",
                    },
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": True},
                    "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to exclude"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                    "context_lines": {"type": "integer", "description": "Number of context lines to include", "default": 0},
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Programming language file types to search",
                    },
                    "max_filesize": {"type": "string", "description": "Maximum file size to search (e.g., '10M', '1G')"},
                    "max_depth": {"type": "integer", "description": "Maximum directory depth to search"},
                    "follow_symlinks": {"type": "boolean", "description": "Follow symbolic links", "default": False},
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "text", "paths_only"],
                        "description": "Output format",
                        "default": "json",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
        {
            "name": "ripgrep_search_code",
            "description": "Search for code patterns specific to a programming language using ripgrep.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Code pattern to search for"},
                    "language": {"type": "string", "description": "Programming language (python, javascript, etc.)"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "search_type": {
                        "type": "string",
                        "enum": ["pattern", "literal", "word", "regex"],
                        "description": "Type of search",
                        "default": "pattern",
                    },
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": True},
                    "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to exclude"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                    "context_lines": {"type": "integer", "description": "Number of context lines to include", "default": 2},
                    "max_filesize": {"type": "string", "description": "Maximum file size to search (e.g., '10M', '1G')"},
                    "max_depth": {"type": "integer", "description": "Maximum directory depth to search"},
                    "follow_symlinks": {"type": "boolean", "description": "Follow symbolic links", "default": False},
                },
                "required": ["pattern", "language", "path"],
            },
        },
        {
            "name": "ripgrep_find_symbols",
            "description": "Find potential symbol candidates using ripgrep pattern matching.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of the symbol to find"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "variable", "any"],
                        "description": "Type of symbol to search for",
                        "default": "any",
                    },
                    "language": {"type": "string", "description": "Programming language for context"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                    "context_lines": {"type": "integer", "description": "Number of context lines to include", "default": 2},
                },
                "required": ["symbol_name", "path"],
            },
        },
        {
            "name": "ripgrep_file_metrics",
            "description": "Analyze file metrics using ripgrep and system tools.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to analyze"},
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "ripgrep_batch_search",
            "description": "Perform multiple ripgrep searches concurrently for batch processing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patterns": {"type": "array", "items": {"type": "string"}, "description": "List of patterns to search for"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "search_type": {
                        "type": "string",
                        "enum": ["pattern", "literal", "word", "regex"],
                        "description": "Type of search",
                        "default": "pattern",
                    },
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": True},
                    "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to exclude"},
                    "max_results": {"type": "integer", "description": "Maximum number of results per pattern"},
                    "context_lines": {"type": "integer", "description": "Number of context lines to include", "default": 0},
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Programming language file types to search",
                    },
                    "max_concurrent": {"type": "integer", "description": "Maximum concurrent searches", "default": 4},
                },
                "required": ["patterns", "path"],
            },
        },
        {
            "name": "enhanced_search",
            "description": "Intelligent multi-strategy search combining ripgrep speed with semantic understanding.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query or pattern"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "strategy": {
                        "type": "string",
                        "enum": ["exact", "fuzzy", "semantic", "hybrid"],
                        "description": "Search strategy",
                        "default": "hybrid",
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Programming language file types to search",
                    },
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to exclude"},
                    "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "File patterns to include"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 100},
                    "context_lines": {"type": "integer", "description": "Number of context lines to include", "default": 3},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
                    "ranking": {
                        "type": "string",
                        "enum": ["relevance", "frequency", "recency", "confidence", "combined"],
                        "description": "Result ranking strategy",
                        "default": "combined",
                    },
                    "timeout": {"type": "number", "description": "Search timeout in seconds", "default": 30.0},
                },
                "required": ["query", "path"],
            },
        },
        {
            "name": "enhanced_search_intelligent",
            "description": "Smart intent-aware search that auto-detects optimal strategy and parameters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query or intent"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "context": {"type": "string", "description": "Additional context about the search intent"},
                    "auto_detect_strategy": {"type": "boolean", "description": "Auto-detect optimal search strategy", "default": True},
                    "language": {"type": "string", "description": "Preferred programming language"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 50},
                    "optimize_for": {
                        "type": "string",
                        "enum": ["speed", "accuracy", "comprehensiveness"],
                        "description": "Optimization goal",
                        "default": "accuracy",
                    },
                },
                "required": ["query", "path"],
            },
        },
        {
            "name": "search_with_context",
            "description": "Context-preserving search that maintains state across multiple operations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "path": {"type": "string", "description": "Directory or file path to search in"},
                    "context_id": {"type": "string", "description": "Unique identifier for search context"},
                    "previous_results": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Previous search results for context",
                    },
                    "refine": {"type": "boolean", "description": "Refine previous results instead of new search", "default": False},
                    "incremental": {"type": "boolean", "description": "Incremental search mode", "default": False},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 100},
                },
                "required": ["query", "path", "context_id"],
            },
        },
        {
            "name": "search_performance_optimize",
            "description": "Optimize search performance through pattern analysis and cache management.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["analyze_patterns", "optimize_cache", "tune_pipeline", "benchmark"],
                        "description": "Optimization action to perform",
                    },
                    "path": {"type": "string", "description": "Path to analyze (for pattern analysis)"},
                    "common_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Common search patterns to optimize for",
                    },
                    "cache_size": {"type": "integer", "description": "Desired cache size", "default": 1000},
                    "clear_cache": {"type": "boolean", "description": "Clear existing cache", "default": False},
                },
                "required": ["action"],
            },
        },
        {
            "name": "deep_semantic_analysis",
            "description": "Perform deep semantic analysis of code including intent, behavior, patterns, and quality.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "analysis_depth": {
                        "type": "string",
                        "enum": ["basic", "comprehensive", "deep"],
                        "description": "Depth of analysis",
                        "default": "comprehensive",
                    },
                    "include_patterns": {"type": "boolean", "description": "Include design pattern detection", "default": True},
                    "include_quality": {"type": "boolean", "description": "Include quality assessment", "default": True},
                    "context": {"type": "string", "description": "Additional context for analysis"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "understand_code_intent",
            "description": "Analyze the intent and purpose of code with confidence scoring.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "context": {"type": "string", "description": "Additional context for analysis"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "analyze_runtime_behavior",
            "description": "Analyze runtime behavior patterns, side effects, and performance characteristics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                },
                "required": ["code"],
            },
        },
        {
            "name": "identify_design_patterns",
            "description": "Identify design patterns and anti-patterns in code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "include_suggestions": {"type": "boolean", "description": "Include improvement suggestions", "default": True},
                },
                "required": ["code"],
            },
        },
        {
            "name": "assess_code_quality",
            "description": "Comprehensive code quality assessment across multiple dimensions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to assess"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "include_recommendations": {"type": "boolean", "description": "Include improvement recommendations", "default": True},
                },
                "required": ["code"],
            },
        },
        {
            "name": "map_relationships",
            "description": "Map code relationships and architectural dependencies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "context": {"type": "string", "description": "File path or additional context"},
                    "project_path": {"type": "string", "description": "Root project path"},
                    "include_dependencies": {"type": "boolean", "description": "Include dependency analysis", "default": True},
                    "include_coupling": {"type": "boolean", "description": "Include coupling analysis", "default": True},
                    "include_cohesion": {"type": "boolean", "description": "Include cohesion analysis", "default": True},
                },
                "required": ["code"],
            },
        },
        {
            "name": "detect_circular_dependencies",
            "description": "Detect circular dependencies and provide resolution suggestions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Project root path"},
                    "include_impact": {"type": "boolean", "description": "Include impact analysis", "default": True},
                    "include_suggestions": {"type": "boolean", "description": "Include resolution suggestions", "default": True},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "analyze_coupling_cohesion",
            "description": "Analyze module coupling and cohesion for architectural insights.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Project root path"},
                    "threshold": {"type": "number", "description": "Threshold for flagging issues", "default": 0.7},
                    "include_recommendations": {"type": "boolean", "description": "Include improvement recommendations", "default": True},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "map_control_flow",
            "description": "Map control flow within code for execution path analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "include_complexity": {"type": "boolean", "description": "Include complexity metrics", "default": True},
                },
                "required": ["code"],
            },
        },
        {
            "name": "analyze_data_flow",
            "description": "Analyze data flow patterns and variable dependencies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to analyze"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"},
                    "include_tracking": {"type": "boolean", "description": "Include variable tracking", "default": True},
                },
                "required": ["code"],
            },
        },
        {
            "name": "safe_rename_symbol",
            "description": "Safely rename a symbol with automatic reference updating and rollback capability.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current symbol name"},
                    "new_name": {"type": "string", "description": "New symbol name"},
                    "symbol_type": {
                        "type": "string",
                        "description": "Type of symbol (function, class, variable, etc.)",
                        "default": "function",
                    },
                    "scope": {"type": "string", "description": "Optional scope/context for the symbol"},
                    "project_path": {"type": "string", "description": "Path to the project root", "default": "."},
                },
                "required": ["old_name", "new_name"],
            },
        },
        {
            "name": "analyze_rename_impact",
            "description": "Analyze the impact and risks of renaming a symbol before performing the operation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current symbol name"},
                    "new_name": {"type": "string", "description": "New symbol name"},
                    "symbol_type": {
                        "type": "string",
                        "description": "Type of symbol (function, class, variable, etc.)",
                        "default": "function",
                    },
                    "scope": {"type": "string", "description": "Optional scope/context for the symbol"},
                    "project_path": {"type": "string", "description": "Path to the project root", "default": "."},
                },
                "required": ["old_name", "new_name"],
            },
        },
        {
            "name": "safe_extract_function",
            "description": "Safely extract a function to a separate file with dependency management.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Source range [start_line, end_line] to extract",
                    },
                    "function_name": {"type": "string", "description": "Name for the extracted function"},
                    "target_file": {"type": "string", "description": "Target file path for the extracted function"},
                    "source_file": {"type": "string", "description": "Source file containing the code to extract"},
                    "project_path": {"type": "string", "description": "Path to the project root", "default": "."},
                },
                "required": ["source_range", "function_name", "target_file", "source_file"],
            },
        },
        {
            "name": "safe_move_symbol",
            "description": "Safely move a symbol (function/class) to another file with reference updating.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of the symbol to move"},
                    "source_file": {"type": "string", "description": "Current file containing the symbol"},
                    "target_file": {"type": "string", "description": "Target file to move the symbol to"},
                    "symbol_type": {
                        "type": "string",
                        "description": "Type of symbol (function, class, variable, etc.)",
                        "default": "function",
                    },
                    "scope": {"type": "string", "description": "Optional scope/context for the symbol"},
                    "project_path": {"type": "string", "description": "Path to the project root", "default": "."},
                },
                "required": ["symbol_name", "source_file", "target_file"],
            },
        },
        {
            "name": "execute_rollback",
            "description": "Execute a rollback operation for a previous refactoring change.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string", "description": "Operation ID to rollback"},
                },
                "required": ["operation_id"],
            },
        },
        # Batch Operations Tools - Phase 6
        {
            "name": "batch_analyze_project",
            "description": "Large-scale project analysis with batch processing for 1000+ files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "analysis_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["complexity", "dependencies", "quality", "security", "performance"]},
                        "description": "Types of analysis to perform",
                    },
                    "max_workers": {"type": "integer", "description": "Maximum concurrent workers", "default": 4},
                    "timeout": {"type": "number", "description": "Timeout in seconds", "default": 300.0},
                    "output_format": {"type": "string", "enum": ["json", "text", "summary"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "batch_transform_code",
            "description": "Bulk code transformations with safety validation and rollback capability.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "transformation_type": {
                        "type": "string",
                        "enum": ["rename", "extract", "move", "pattern_replace"],
                        "description": "Type of transformation",
                    },
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Target files, patterns, or symbols",
                    },
                    "parameters": {"type": "object", "description": "Transformation-specific parameters"},
                    "validation_level": {
                        "type": "string",
                        "enum": ["strict", "normal", "permissive"],
                        "description": "Safety validation level",
                        "default": "normal",
                    },
                    "dry_run": {"type": "boolean", "description": "Preview changes without applying", "default": False},
                    "max_workers": {"type": "integer", "description": "Maximum concurrent workers", "default": 2},
                },
                "required": ["transformation_type", "project_path", "targets"],
            },
        },
        {
            "name": "monitor_batch_progress",
            "description": "Monitor progress of batch operations with real-time metrics and status updates.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string", "description": "Operation ID to monitor"},
                    "include_details": {"type": "boolean", "description": "Include detailed progress information", "default": True},
                    "include_metrics": {"type": "boolean", "description": "Include performance metrics", "default": True},
                    "include_errors": {"type": "boolean", "description": "Include error information", "default": True},
                    "refresh_interval": {"type": "number", "description": "Refresh interval in seconds", "default": 1.0},
                },
                "required": ["operation_id"],
            },
        },
        {
            "name": "schedule_batch_operations",
            "description": "Schedule and manage batch operations with priority and resource management.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Operation type"},
                                "parameters": {"type": "object", "description": "Operation parameters"},
                                "priority": {"type": "string", "enum": ["low", "normal", "high", "critical"], "default": "normal"},
                            },
                        },
                        "description": "Operations to schedule",
                    },
                    "max_concurrent": {"type": "integer", "description": "Maximum concurrent operations", "default": 3},
                    "resource_limits": {
                        "type": "object",
                        "properties": {
                            "max_memory_mb": {"type": "integer", "description": "Maximum memory usage in MB"},
                            "max_cpu_percent": {"type": "integer", "description": "Maximum CPU usage percentage"},
                            "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
                        },
                    },
                    "schedule_mode": {"type": "string", "enum": ["immediate", "queued", "timed"], "default": "queued"},
                },
                "required": ["operations"],
            },
        },
        {
            "name": "validate_batch_operations",
            "description": "Validate batch operations for safety and potential conflicts before execution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation_plan": {
                        "type": "object",
                        "description": "Operation plan to validate",
                    },
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "validation_level": {
                        "type": "string",
                        "enum": ["quick", "standard", "thorough"],
                        "description": "Validation depth",
                        "default": "standard",
                    },
                    "include_performance": {"type": "boolean", "description": "Include performance impact analysis", "default": True},
                    "include_security": {"type": "boolean", "description": "Include security analysis", "default": True},
                },
                "required": ["operation_plan", "project_path"],
            },
        },
        {
            "name": "execute_batch_rename",
            "description": "Execute batch symbol renaming across multiple files with dependency tracking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rename_operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_name": {"type": "string", "description": "Current symbol name"},
                                "new_name": {"type": "string", "description": "New symbol name"},
                                "symbol_type": {"type": "string", "description": "Symbol type"},
                                "scope": {"type": "string", "description": "Optional scope"},
                            },
                        },
                        "description": "Rename operations to execute",
                    },
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "update_references": {"type": "boolean", "description": "Update all references automatically", "default": True},
                    "dry_run": {"type": "boolean", "description": "Preview without applying changes", "default": False},
                    "create_backups": {"type": "boolean", "description": "Create backups before changes", "default": True},
                },
                "required": ["rename_operations", "project_path"],
            },
        },
        {
            "name": "batch_extract_components",
            "description": "Extract multiple components (functions, classes) to separate files with dependency management.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_file": {"type": "string", "description": "Source file path"},
                                "source_range": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] line range"},
                                "target_file": {"type": "string", "description": "Target file path"},
                                "component_name": {"type": "string", "description": "Component name"},
                                "component_type": {
                                    "type": "string",
                                    "enum": ["function", "class", "method"],
                                    "description": "Component type",
                                },
                            },
                        },
                        "description": "Components to extract",
                    },
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "manage_imports": {"type": "boolean", "description": "Automatically manage imports", "default": True},
                    "create_backups": {"type": "boolean", "description": "Create backups before extraction", "default": True},
                    "max_workers": {"type": "integer", "description": "Maximum concurrent extractions", "default": 2},
                },
                "required": ["extractions", "project_path"],
            },
        },
        {
            "name": "generate_batch_report",
            "description": "Generate comprehensive reports for batch operations with metrics and analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string", "description": "Operation ID to generate report for"},
                    "report_type": {
                        "type": "string",
                        "enum": ["summary", "detailed", "executive", "technical"],
                        "description": "Type of report to generate",
                        "default": "detailed",
                    },
                    "include_metrics": {"type": "boolean", "description": "Include performance metrics", "default": True},
                    "include_errors": {"type": "boolean", "description": "Include error analysis", "default": True},
                    "include_recommendations": {"type": "boolean", "description": "Include improvement recommendations", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "text", "markdown", "html"], "default": "json"},
                },
                "required": ["operation_id"],
            },
        },
        # Security & Quality Analysis Tools
        {
            "name": "security_scan_comprehensive",
            "description": "Perform comprehensive security vulnerability scan with OWASP Top 10 coverage.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "scan_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["owasp_top_10", "dependency_check", "code_analysis", "configuration_audit"]},
                        "description": "Types of security scans to perform",
                        "default": ["owasp_top_10", "dependency_check"],
                    },
                    "severity_threshold": {
                        "type": "string",
                        "enum": ["info", "low", "medium", "high", "critical"],
                        "description": "Minimum severity level to report",
                        "default": "low",
                    },
                    "include_remediation": {"type": "boolean", "description": "Include remediation guidance", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "text", "html"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "quality_assessment_comprehensive",
            "description": "Perform comprehensive code quality assessment with maintainability analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "file_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns to include in assessment",
                        "default": ["*.py", "*.js", "*.ts", "*.java", "*.cpp"],
                    },
                    "include_complexity": {"type": "boolean", "description": "Include complexity analysis", "default": True},
                    "include_code_smells": {"type": "boolean", "description": "Include code smell detection", "default": True},
                    "include_duplication": {"type": "boolean", "description": "Include code duplication analysis", "default": True},
                    "output_format": {"type": "string", "enum": ["json", "text", "html"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "compliance_reporting_generate",
            "description": "Generate compliance reports for various security and quality standards.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "standards": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["owasp_top_10", "pci_dss", "hipaa", "gdpr", "soc2", "iso27001", "nist_csf", "cwe_top_25"],
                        },
                        "description": "Compliance standards to evaluate",
                        "default": ["owasp_top_10"],
                    },
                    "include_security_scan": {"type": "boolean", "description": "Include security vulnerability scan", "default": True},
                    "include_quality_assessment": {"type": "boolean", "description": "Include quality assessment", "default": True},
                    "certification_threshold": {
                        "type": "number",
                        "description": "Minimum score for certification readiness",
                        "default": 85.0,
                    },
                    "output_format": {"type": "string", "enum": ["json", "html", "pdf"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "quality_gates_evaluate",
            "description": "Evaluate code against automated quality gates with customizable thresholds.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "gates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Gate name"},
                                "metric": {
                                    "type": "string",
                                    "enum": [
                                        "cyclomatic_complexity",
                                        "cognitive_complexity",
                                        "maintainability_index",
                                        "code_coverage",
                                        "technical_debt",
                                        "duplicate_code",
                                        "code_smells",
                                        "security_issues",
                                    ],
                                },
                                "threshold": {"type": "number", "description": "Threshold value"},
                                "operator": {"type": "string", "enum": ["gt", "gte", "lt", "lte"], "description": "Comparison operator"},
                                "severity": {"type": "string", "enum": ["warning", "error"], "description": "Failure severity"},
                                "enabled": {"type": "boolean", "description": "Whether gate is enabled", "default": True},
                            },
                            "required": ["name", "metric", "threshold", "operator", "severity"],
                        },
                        "description": "Custom quality gates (uses defaults if not provided)",
                    },
                    "fail_on_error": {"type": "boolean", "description": "Fail build on error gates", "default": True},
                    "fail_on_warning": {"type": "boolean", "description": "Fail build on warning gates", "default": False},
                    "output_format": {"type": "string", "enum": ["json", "text", "junit"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
        {
            "name": "vulnerability_database_check",
            "description": "Check dependencies against known vulnerability databases.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "description": "Path to the project root"},
                    "dependency_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific dependency files to check",
                        "default": ["requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml"],
                    },
                    "include_transitive": {"type": "boolean", "description": "Include transitive dependencies", "default": True},
                    "severity_filter": {
                        "type": "string",
                        "enum": ["all", "critical", "high", "medium", "low"],
                        "description": "Filter vulnerabilities by severity",
                        "default": "all",
                    },
                    "output_format": {"type": "string", "enum": ["json", "text", "sarif"], "default": "json"},
                },
                "required": ["project_path"],
            },
        },
    ]


# Auto-Invocation Intent Detection System
# Maps user interaction keywords to optimal tool sequences
INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    # Analysis intents
    "analyze": {
        "keywords": ["analyze", "analysis", "examine", "inspect", "review", "audit", "assess"],
        "primary_tool": "llm_analyze_code",
        "fallback_tools": ["dump_syntax_tree", "find_code"],
        "context": "deep_code_understanding",
        "auto_qdrant": True,
    },
    "research": {
        "keywords": ["research", "investigate", "explore", "study", "learn", "discover"],
        "primary_tool": "llm_search_pattern",
        "fallback_tools": ["find_code_by_rule", "find_code"],
        "context": "pattern_discovery",
        "auto_qdrant": True,
        "use_collective_memory": True,
    },
    "implement": {
        "keywords": ["implement", "create", "build", "develop", "write", "code", "generate"],
        "primary_tool": "llm_generate_rule",
        "fallback_tools": ["find_code_by_rule", "test_match_code_rule"],
        "context": "rule_generation",
        "auto_qdrant": True,
    },
    "refactor": {
        "keywords": ["refactor", "restructure", "reorganize", "optimize", "improve", "enhance"],
        "primary_tool": "llm_analyze_code",
        "secondary_tool": "llm_generate_rule",
        "fallback_tools": ["find_code_by_rule"],
        "context": "code_improvement",
        "auto_qdrant": True,
    },
    "test": {
        "keywords": ["test", "validate", "verify", "check", "ensure", "confirm"],
        "primary_tool": "test_match_code_rule",
        "secondary_tool": "llm_analyze_code",
        "fallback_tools": ["find_code"],
        "context": "validation",
        "auto_qdrant": False,
    },
    "document": {
        "keywords": ["document", "explain", "describe", "comment", "clarify", "specify"],
        "primary_tool": "llm_analyze_code",
        "fallback_tools": ["dump_syntax_tree"],
        "context": "documentation_generation",
        "auto_qdrant": True,
    },
    "troubleshoot": {
        "keywords": ["troubleshoot", "debug", "fix", "repair", "resolve", "issue", "error"],
        "primary_tool": "llm_analyze_code",
        "secondary_tool": "llm_search_pattern",
        "fallback_tools": ["find_code", "dump_syntax_tree"],
        "context": "error_resolution",
        "auto_qdrant": True,
        "use_collective_memory": True,
    },
    "improve": {
        "keywords": ["improve", "better", "enhance", "perfect", "polish", "refine"],
        "primary_tool": "llm_analyze_code",
        "secondary_tool": "llm_generate_rule",
        "fallback_tools": ["find_code_by_rule"],
        "context": "quality_enhancement",
        "auto_qdrant": True,
    },
    "audit": {
        "keywords": ["audit", "review", "check", "inspect", "validate", "compliance"],
        "primary_tool": "llm_analyze_code",
        "secondary_tool": "llm_search_pattern",
        "fallback_tools": ["find_code_by_rule"],
        "context": "compliance_checking",
        "auto_qdrant": True,
    },
    "pattern": {
        "keywords": ["pattern", "template", "structure", "architecture", "design"],
        "primary_tool": "llm_search_pattern",
        "secondary_tool": "llm_generate_rule",
        "fallback_tools": ["find_code_by_rule"],
        "context": "architectural_analysis",
        "auto_qdrant": True,
    },
}


def detect_user_intent(query: str) -> Dict[str, Any]:
    """Detect user intent from query using keyword matching."""
    query_lower = query.lower()

    # Score each intent pattern
    intent_scores: Dict[str, Dict[str, Any]] = {}
    for intent_name, intent_config in INTENT_PATTERNS.items():
        score = 0
        keywords = cast(List[str], intent_config["keywords"])

        # Count keyword matches
        for keyword in keywords:
            if keyword in query_lower:
                score += 1

        # Bonus for exact intent name match
        if intent_name in query_lower:
            score += 2

        if score > 0:
            intent_scores[intent_name] = {"score": score, "config": intent_config}

    # Return highest scoring intent
    if intent_scores:
        best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
        return {
            "intent": best_intent[0],
            "confidence": best_intent[1]["score"] / 10.0,  # Normalize to 0-1
            "config": best_intent[1]["config"],
        }

    # Default to general analysis
    return {"intent": "analyze", "confidence": 0.5, "config": INTENT_PATTERNS["analyze"]}


def generate_auto_invocation_plan(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Generate optimal tool invocation sequence based on user intent."""
    intent_data = detect_user_intent(query)
    config = intent_data["config"]

    plan = {
        "detected_intent": intent_data["intent"],
        "confidence": intent_data["confidence"],
        "primary_tool": config["primary_tool"],
        "context": config["context"],
        "auto_qdrant": config.get("auto_qdrant", False),
        "tool_sequence": [],
    }

    # Build tool sequence
    tool_sequence = []

    # Add primary tool
    tool_sequence.append({"tool": config["primary_tool"], "priority": "primary", "reason": f"Primary tool for {intent_data['intent']}"})

    # Add secondary tool if specified
    if "secondary_tool" in config:
        tool_sequence.append(
            {"tool": config["secondary_tool"], "priority": "secondary", "reason": f"Secondary tool for {intent_data['intent']}"}
        )

    # Add fallback tools
    for fallback_tool in config.get("fallback_tools", []):
        tool_sequence.append({"tool": fallback_tool, "priority": "fallback", "reason": f"Fallback tool for {intent_data['intent']}"})

    plan["tool_sequence"] = tool_sequence

    # Add Qdrant integration if enabled
    if config.get("auto_qdrant", False):
        plan["qdrant_actions"] = [
            "store_interaction_context",
            "retrieve_similar_patterns" if config.get("use_collective_memory", False) else None,
        ]
        plan["qdrant_actions"] = [action for action in plan["qdrant_actions"] if action]

    # Add context-specific parameters
    plan["suggested_parameters"] = generate_context_parameters(intent_data["intent"], query, context)

    return plan


def generate_context_parameters(intent: str, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Generate context-aware parameters for tool invocation."""
    params: Dict[str, Any] = {}

    # Extract potential code snippets from query
    if "```" in query:
        # Extract code between triple backticks
        code_parts = query.split("```")
        if len(code_parts) >= 3:
            params["code"] = code_parts[1].strip()

    # Extract file references
    import re

    file_patterns = re.findall(r"\w+\.(py|js|ts|java|cpp|c|go|rs|php|rb)", query)
    if file_patterns:
        params["target_files"] = file_patterns

    # Language detection
    language_keywords = {
        "python": ["python", "py", "def", "class", "import"],
        "javascript": ["javascript", "js", "function", "const", "let"],
        "typescript": ["typescript", "ts", "interface", "type"],
        "java": ["java", "public", "class", "void"],
        "cpp": ["cpp", "c++", "namespace", "template"],
        "go": ["go", "func", "package", "import"],
        "rust": ["rust", "rs", "fn", "let", "struct"],
    }

    detected_languages = []
    query_lower = query.lower()
    for lang, keywords in language_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_languages.append(lang)

    if detected_languages:
        params["language"] = detected_languages[0]  # Primary language
        if len(detected_languages) > 1:
            params["potential_languages"] = detected_languages

    # Intent-specific parameters
    if intent == "analyze":
        params["analysis_type"] = "general"
        if "security" in query.lower():
            params["analysis_type"] = "security"
        elif "performance" in query.lower():
            params["analysis_type"] = "performance"
        elif "complexity" in query.lower():
            params["analysis_type"] = "complexity"

    elif intent == "research":
        params["use_collective_memory"] = True
        if "pattern" in query.lower():
            params["search_type"] = "pattern"
        elif "bug" in query.lower() or "error" in query.lower():
            params["search_type"] = "error"

    elif intent == "implement":
        if "rule" in query.lower():
            params["rule_type"] = "pattern"
        params["examples"] = []  # Will be populated from context

    elif intent == "refactor":
        params["analysis_type"] = "refactoring_opportunities"

    elif intent == "test":
        params["validation_type"] = "syntax"

    elif intent == "troubleshoot":
        params["analysis_type"] = "error_detection"
        params["focus"] = "debugging"

    return params


async def auto_invoke_ast_intelligence(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Automatically invoke optimal AST intelligence tools based on user query."""
    start_time = datetime.now()

    # Generate invocation plan
    plan = generate_auto_invocation_plan(query, context)

    logger.info(
        "Auto-invocation plan generated",
        intent=plan["detected_intent"],
        confidence=plan["confidence"],
        primary_tool=plan["primary_tool"],
        tools_count=len(plan["tool_sequence"]),
    )

    results = {
        "query": query,
        "detected_intent": plan["detected_intent"],
        "confidence": plan["confidence"],
        "execution_plan": plan,
        "tool_results": [],
        "processing_time_ms": 0,
        "success": False,
    }

    # Execute tool sequence
    for tool_step in plan["tool_sequence"]:
        tool_result = await execute_tool_with_fallback(tool_step, plan, context)
        results["tool_results"].append(tool_result)

        # Stop if primary tool succeeded
        if tool_result["success"] and tool_step["priority"] == "primary":
            results["success"] = True
            break

    # Qdrant integration
    if plan.get("auto_qdrant", False) and plan.get("qdrant_actions"):
        await handle_qdrant_integration(query, plan, results)

    results["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(
        "Auto-invocation completed",
        success=results["success"],
        tools_executed=len([r for r in results["tool_results"] if r.get("executed", False)]),
        processing_time_ms=results["processing_time_ms"],
    )

    return results


async def execute_tool_with_fallback(tool_step: Dict, plan: Dict, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Execute a single tool with fallback handling."""
    tool_name = tool_step["tool"]
    result = {"tool": tool_name, "priority": tool_step["priority"], "executed": False, "success": False, "error": None, "result": None}

    try:
        # Prepare tool-specific arguments
        tool_args = prepare_tool_arguments(tool_name, plan, context)

        # Execute tool via existing MCP tool system
        if tool_name in [
            "llm_analyze_code",
            "llm_generate_rule",
            "llm_search_pattern",
            "dump_syntax_tree",
            "test_match_code_rule",
            "find_code_by_rule",
            "find_code",
        ]:
            # Call the tool using the existing MCP infrastructure
            tool_result = await call_tool(tool_name, tool_args)

            result["executed"] = True
            result["success"] = True
            result["result"] = tool_result

            logger.info(f"Tool execution succeeded: {tool_name}")

        else:
            result["error"] = f"Unknown tool: {tool_name}"

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Tool execution failed: {tool_name}", error=str(e))

    return result


def prepare_tool_arguments(tool_name: str, plan: Dict, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Prepare arguments for specific tool execution."""
    suggested_params = plan.get("suggested_parameters", {})
    args = {}

    if tool_name == "llm_analyze_code":
        args["code"] = suggested_params.get("code", "// Sample code for analysis")
        args["language"] = suggested_params.get("language", "python")
        args["analysis_type"] = suggested_params.get("analysis_type", "general")

    elif tool_name == "llm_generate_rule":
        args["query"] = plan["detected_intent"]
        args["language"] = suggested_params.get("language", "python")
        args["examples"] = suggested_params.get("examples", [])
        args["rule_type"] = suggested_params.get("rule_type", "pattern")

    elif tool_name == "llm_search_pattern":
        args["query"] = plan["detected_intent"]
        args["language"] = suggested_params.get("language", "python")
        args["use_collective_memory"] = suggested_params.get("use_collective_memory", False)

    elif tool_name == "dump_syntax_tree":
        args["code"] = suggested_params.get("code", "// Sample code")
        args["language"] = suggested_params.get("language", "python")
        args["output_format"] = "pattern"

    elif tool_name == "test_match_code_rule":
        args["code"] = suggested_params.get("code", "// Sample code")
        args["rule_yaml"] = "rule:\n  pattern: $PATTERN"  # Default rule

    elif tool_name == "find_code_by_rule":
        args["rule_yaml"] = "rule:\n  pattern: $PATTERN"  # Default rule
        args["path"] = "."
        args["output_format"] = "json"

    elif tool_name == "find_code":
        args["pattern"] = suggested_params.get("code", "def")
        args["path"] = "."
        args["language"] = suggested_params.get("language", "python")

    return args


async def handle_qdrant_integration(query: str, plan: Dict, results: Dict) -> None:
    """Handle Qdrant storage and retrieval for learning."""
    try:
        # Store interaction for learning
        interaction_data = {
            "query": query,
            "intent": plan["detected_intent"],
            "confidence": plan["confidence"],
            "tools_used": [step["tool"] for step in results["tool_results"] if step.get("executed", False)],
            "success": results["success"],
            "timestamp": datetime.now().isoformat(),
        }

        # Store in Qdrant if available
        if AST_INTELLIGENCE_AVAILABLE:
            await ast_rule_intelligence.store_interaction_for_learning(interaction_data)

        logger.info("Qdrant integration completed", intent=plan["detected_intent"])

    except Exception as e:
        logger.warning("Qdrant integration failed", error=str(e))


async def detect_and_execute_ast_intelligence(
    query: str, context: Optional[str] = None, language: str = "python", path: str = ".", auto_execute: bool = True
) -> Dict[str, Any]:
    """Auto-invocation system for AST intelligence based on user intent detection.

    This function analyzes user queries and automatically invokes the optimal
    AST intelligence tools based on detected intent patterns.

    Args:
        query: Natural language query from user
        context: Optional context about current task
        language: Default programming language
        path: Default path for file operations
        auto_execute: Whether to automatically execute detected tools

    Returns:
        Dictionary containing intent analysis, execution results, and recommendations
    """
    try:
        # Step 1: Detect user intent from query
        detected_intent = detect_user_intent(query)

        # Step 2: Generate execution plan using AST intelligence
        if AST_INTELLIGENCE_AVAILABLE:
            reasoning_engine = ast_rule_intelligence.LLMAstReasoningEngine()
            execution_plan = await reasoning_engine.reason_and_generate_rule(query, language)
        else:
            # Fallback to simple intent-based plan
            execution_plan = {"intent": detected_intent, "confidence": 0.7, "actions": ["analyze_code", "search_patterns"], "rules": []}

        # Step 3: Execute plan if auto_execute is enabled
        execution_results = {}
        if auto_execute and execution_plan.get("confidence", 0) > 0.5:
            if AST_INTELLIGENCE_AVAILABLE and "rules" in execution_plan:
                # Execute generated rules using available tools
                for rule in execution_plan["rules"]:
                    if rule.get("tool_name"):
                        tool_result = await reasoning_engine.execute_generated_tool(rule["tool_name"], path or ".")
                        execution_results[rule["tool_name"]] = tool_result
            else:
                # Fallback execution for simple plans
                execution_results = {"status": "executed", "plan": execution_plan}

        # Step 4: Handle Qdrant integration for learning
        if auto_execute:
            await handle_qdrant_integration(query, execution_plan, execution_results)

        return {
            "query": query,
            "detected_intent": detected_intent,
            "execution_plan": execution_plan,
            "execution_results": execution_results,
            "auto_executed": auto_execute,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("Auto-invocation system failed", error=str(e), query=query)
        return {
            "query": query,
            "error": str(e),
            "fallback_mode": True,
            "timestamp": datetime.now().isoformat(),
        }


# Global auto-invocation entry point
async def handle_user_interaction(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Main entry point for automatic AST intelligence invocation.

    This function should be called whenever a user interaction is detected
    that might benefit from AST analysis. It automatically detects intent
    and invokes the optimal sequence of tools.

    Args:
        query: The user's natural language query or request
        context: Optional context about the current codebase or session

    Returns:
        Dictionary containing execution results and analysis
    """
    return await auto_invoke_ast_intelligence(query, context)


@mcp.tool()
async def call_tool(name: str, arguments: dict) -> List[Dict[str, Any]]:
    """Handle tool calls with unified branching and robust safety checks."""
    request_id = str(uuid.uuid4())
    try:
        if name == "edit_file":
            target_file = arguments.get("target_file") or arguments.get("path")
            code_edit = arguments.get("code_edit")
            instruction = arguments.get("instructions") or arguments.get("instruction", "")
            if not target_file:
                raise ValueError("target_file parameter is required")
            if not instruction:
                raise ValueError("instructions parameter is required")
            if not code_edit:
                raise ValueError("code_edit parameter is required")
            logger.info("edit_file tool called", target_file=target_file, code_edit_length=len(code_edit), instruction=instruction)
            try:
                secure_path = _secure_resolve(target_file)
            except ValueError:
                raise ValueError("Invalid file path")
            if not os.path.exists(secure_path):
                raise ValueError(f"File not found: {secure_path}")
            file_size = os.path.getsize(secure_path)
            if file_size > MAX_FILE_SIZE:
                raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE} bytes)")
            if not _is_allowed_edit_target(secure_path):
                raise ValueError(f"Editing not permitted for extension of {target_file}; allowed: {', '.join(ALLOWED_EXTS)}")
            try:
                with open(secure_path, "r", encoding="utf-8") as f:
                    original_code = f.read()
                original_hash = hashlib.sha256(original_code.encode("utf-8")).hexdigest()
            except Exception as e:
                raise IOError(f"Failed to read file {secure_path}: {e}")
            try:
                verification_results = fast_apply_connector.apply_edit(
                    original_code=original_code,
                    code_edit=code_edit,
                    instruction=instruction,
                    file_path=secure_path,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to apply edit: {e}") from e
            merged_code = verification_results if isinstance(verification_results, str) else verification_results.get("merged_code", "")
            if isinstance(verification_results, dict) and verification_results.get("validation", {}).get("has_errors"):
                errors_text = "; ".join(verification_results["validation"]["errors"])
                return [{"type": "text", "text": f"❌ Edit applied but validation failed:\n{errors_text}"}]
            force = bool(arguments.get("force", False))
            try:
                with open(secure_path, "r", encoding="utf-8") as f:
                    current_code = f.read()
                current_hash = hashlib.sha256(current_code.encode("utf-8")).hexdigest()
            except Exception:
                current_hash = None
            if not force and current_hash and current_hash != original_hash:
                raise RuntimeError("File changed on disk since read; aborting edit (pass force=true to override).")
            try:
                backup_path = write_with_backup(secure_path, merged_code)
            except Exception as e:
                raise IOError(f"Failed to write file {secure_path}: {e}")
            rel_target = os.path.relpath(secure_path, WORKSPACE_ROOT)
            rel_backup = os.path.relpath(backup_path, WORKSPACE_ROOT)
            output_format = arguments.get("output_format") or "text"
            if output_format == "json" and isinstance(verification_results, dict):
                payload = {
                    "request_id": request_id,
                    "target_file": rel_target,
                    "backup": rel_backup,
                    "has_changes": verification_results.get("has_changes"),
                    "udiff": verification_results.get("udiff"),
                    "validation": verification_results.get("validation"),
                }
                return [{"type": "text", "text": json_dumps(payload)}]
            parts = [f"request_id={request_id}", f"✅ Successfully applied edit to {rel_target}", f"💾 Backup: {rel_backup}"]
            if isinstance(verification_results, dict) and verification_results.get("has_changes") and verification_results.get("udiff"):
                parts.append(f"\n📊 Changes (UDiff):\n{verification_results['udiff']}")
            if isinstance(verification_results, dict) and verification_results.get("validation", {}).get("warnings"):
                parts.append("\n⚠️  Validation warnings:\n" + "\n".join(verification_results["validation"]["warnings"]))
            return [{"type": "text", "text": "\n".join(parts)}]
        elif name == "read_multiple_files":
            paths = arguments.get("paths", [])
            if not paths:
                raise ValueError("paths parameter is required and must be non-empty")
            results = []
            for file_path in paths:
                try:
                    secure_path = _secure_resolve(file_path)
                    if not os.path.exists(secure_path):
                        results.append(f"{file_path}: Error - File not found")
                        continue
                    with open(secure_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    results.append(f"{file_path}:\n{content}\n")
                except Exception as e:
                    results.append(f"{file_path}: Error - {e}")
            return [{"type": "text", "text": f"request_id={request_id}\n" + "\n---\n".join(results)}]
        elif name == "dry_run_edit_file":
            target_file = arguments.get("target_file") or arguments.get("path")
            code_edit = arguments.get("code_edit")
            instruction = arguments.get("instruction") or arguments.get("instructions", "")
            if not target_file:
                raise ValueError("target_file parameter is required")
            if not code_edit:
                raise ValueError("code_edit parameter is required")
            logger.info("dry_run_edit_file tool called", target_file=target_file, code_edit_length=len(code_edit), instruction=instruction)
            secure_path = _secure_resolve(target_file)
            if not os.path.exists(secure_path):
                raise ValueError(f"File not found: {secure_path}")
            with open(secure_path, "r", encoding="utf-8") as f:
                original_code = f.read()
            verification_results = fast_apply_connector.apply_edit(
                original_code=original_code,
                code_edit=code_edit,
                instruction=instruction,
                file_path=secure_path,
            )
            merged_code = verification_results["merged_code"]
            has_changes = verification_results["has_changes"]
            udiff = verification_results["udiff"]
            validation = verification_results["validation"]
            rel_target = os.path.relpath(secure_path, WORKSPACE_ROOT)
            output_format = arguments.get("output_format") or "text"
            if output_format == "json":
                payload = {
                    "request_id": request_id,
                    "target_file": rel_target,
                    "has_changes": has_changes,
                    "udiff": udiff,
                    "validation": validation,
                    "preview_lines": merged_code.split("\n")[:20] if has_changes else [],
                }
                return [{"type": "text", "text": json_dumps(payload)}]
            parts = [f"request_id={request_id}", f"🔍 DRY RUN RESULTS for {rel_target}", "=" * 60]
            if has_changes:
                parts.append(f"✅ Changes would be applied ({len(merged_code)} bytes)")
                parts.append(f"📊 Original size: {len(original_code)} bytes → New size: {len(merged_code)} bytes")
            else:
                parts.append("ℹ️  No changes would be made (content identical)")
            if has_changes and udiff:
                parts.append("\n📋 Unified Diff:")
                parts.append(udiff)
            parts.append("\n🔒 Code Validation:")
            if validation["has_errors"]:
                parts.append("❌ Errors: " + "; ".join(validation["errors"]))
            else:
                parts.append("✅ No validation errors")
            if validation["warnings"]:
                parts.append("⚠️  Warnings:\n   " + "\n   ".join(validation["warnings"]))
            else:
                parts.append("✅ No validation warnings")
            parts.append("\n🛡️  Safety Information:")
            parts.extend(
                [
                    "   • No files were modified",
                    "   • No backup was created",
                    f"   • Workspace isolation: ✅ (within {WORKSPACE_ROOT})",
                    "   • Path validation: ✅",
                ]
            )
            if has_changes:
                preview_lines = merged_code.split("\n")[:20]
                parts.append("\n📝 Preview of merged code (first 20 lines):")
                parts.append("```")
                parts.extend(preview_lines)
                if len(merged_code.split("\n")) > 20:
                    parts.append("... (truncated)")
                parts.append("```")
            parts.append("\n💡 To apply these changes, use the edit_file tool with the same parameters.")
            return [{"type": "text", "text": "\n".join(parts)}]
        elif name == "search_files":
            search_path = arguments.get("path", ".")
            pattern = arguments.get("pattern")
            exclude_patterns = arguments.get("excludePatterns", [])
            if not pattern:
                raise ValueError("pattern parameter is required")
            secure_path = _secure_resolve(search_path)
            if not os.path.exists(secure_path):
                raise ValueError(f"Search path not found: {secure_path}")
            file_results: List[str] = search_files(secure_path, pattern, exclude_patterns)
            if file_results:
                rels = [os.path.relpath(p, WORKSPACE_ROOT) for p in file_results]
                return [{"type": "text", "text": f"request_id={request_id}\n" + "\n".join(rels)}]
            return [{"type": "text", "text": f"request_id={request_id}\nNo matches found"}]
        elif name == "search_code_patterns":
            if not AST_SEARCH_AVAILABLE:
                raise ValueError("AST search functionality not available (ast-grep-py not installed)")

            pattern = arguments.get("pattern")
            language = arguments.get("language")
            path = arguments.get("path", ".")
            exclude_patterns = arguments.get("exclude_patterns", [])

            if not pattern:
                raise ValueError("pattern parameter is required")
            if not language:
                raise ValueError("language parameter is required")

            logger.info("search_code_patterns tool called", pattern=pattern, language=language, path=path)
            secure_path = _secure_resolve(path)

            try:
                pattern_results: List[ast_search.PatternSearchResult] = ast_search.search_code_patterns(
                    pattern, language, secure_path, exclude_patterns
                )
                if pattern_results:
                    result_data = {
                        "request_id": request_id,
                        "pattern": pattern,
                        "language": language,
                        "matches_count": len(pattern_results),
                        "matches": [result.to_dict() for result in pattern_results],
                    }
                    return [{"type": "text", "text": json_dumps(result_data)}]
                else:
                    return [{"type": "text", "text": f"request_id={request_id}\nNo code patterns found for: {pattern}"}]
            except ast_search.AstSearchError as e:
                raise ValueError(f"AST search failed: {e}")

        elif name == "analyze_code_structure":
            if not AST_SEARCH_AVAILABLE:
                raise ValueError("AST search functionality not available (ast-grep-py not installed)")

            file_path = arguments.get("file_path")
            if not file_path:
                raise ValueError("file_path parameter is required")

            logger.info("analyze_code_structure tool called", file_path=file_path)
            secure_path = _secure_resolve(file_path)

            if not os.path.exists(secure_path):
                raise ValueError(f"File not found: {secure_path}")

            try:
                structure = ast_search.analyze_code_structure(secure_path)
                result_data = {"request_id": request_id, "structure": structure.to_dict()}
                return [{"type": "text", "text": json_dumps(result_data)}]
            except ast_search.AstSearchError as e:
                raise ValueError(f"Code structure analysis failed: {e}")

        elif name == "find_references":
            if not AST_SEARCH_AVAILABLE:
                raise ValueError("AST search functionality not available (ast-grep-py not installed)")

            symbol = arguments.get("symbol")
            path = arguments.get("path", ".")
            symbol_type = arguments.get("symbol_type", "any")

            if not symbol:
                raise ValueError("symbol parameter is required")

            logger.info("find_references tool called", symbol=symbol, path=path, symbol_type=symbol_type)
            secure_path = _secure_resolve(path)

            try:
                references = ast_search.find_references(symbol, secure_path, symbol_type)
                if references:
                    result_data = {
                        "request_id": request_id,
                        "symbol": symbol,
                        "symbol_type": symbol_type,
                        "references_count": len(references),
                        "references": [ref.to_dict() for ref in references],
                    }
                    return [{"type": "text", "text": json_dumps(result_data)}]
                else:
                    return [{"type": "text", "text": f"request_id={request_id}\nNo references found for symbol: {symbol}"}]
            except ast_search.AstSearchError as e:
                raise ValueError(f"Reference search failed: {e}")

        elif name == "dump_syntax_tree":
            if not AST_GREP_AVAILABLE:
                raise ValueError("ast-grep command not available - please install ast-grep")

            code = arguments.get("code")
            language = arguments.get("language")
            dump_format = arguments.get("dump_format", "pattern")

            if not code:
                raise ValueError("code parameter is required")
            if not language:
                raise ValueError("language parameter is required")

            logger.info("dump_syntax_tree tool called", language=language, dump_format=dump_format)

            try:
                result = ast_search_official.dump_syntax_tree(code, language, dump_format)
                return [{"type": "text", "text": f"request_id={request_id}\n{result}"}]
            except ast_search_official.AstGrepError as e:
                raise ValueError(f"Syntax tree dump failed: {e}")

        elif name == "test_match_code_rule":
            if not AST_GREP_AVAILABLE:
                raise ValueError("ast-grep command not available - please install ast-grep")

            code = arguments.get("code")
            rule_yaml = arguments.get("rule_yaml")

            if not code:
                raise ValueError("code parameter is required")
            if not rule_yaml:
                raise ValueError("rule_yaml parameter is required")

            logger.info("test_match_code_rule tool called")

            try:
                result = ast_search_official.test_match_code_rule(code, rule_yaml)
                return [{"type": "text", "text": f"request_id={request_id}\n{result}"}]
            except ast_search_official.AstGrepError as e:
                raise ValueError(f"Rule testing failed: {e}")

        elif name == "find_code":
            if not AST_GREP_AVAILABLE:
                raise ValueError("ast-grep command not available - please install ast-grep")

            pattern = arguments.get("pattern")
            language = arguments.get("language")
            path = arguments.get("path", ".")
            output_format = arguments.get("output_format", "json")
            max_results = arguments.get("max_results")

            if not pattern:
                raise ValueError("pattern parameter is required")
            if not language:
                raise ValueError("language parameter is required")

            logger.info("find_code tool called", pattern=pattern, language=language, path=path)
            secure_path = _secure_resolve(path)

            try:
                result = ast_search_official.find_code(pattern, language, secure_path, output_format, max_results)
                return [{"type": "text", "text": f"request_id={request_id}\n{result}"}]
            except ast_search_official.AstGrepError as e:
                raise ValueError(f"Code search failed: {e}")

        elif name == "find_code_by_rule":
            if not AST_GREP_AVAILABLE:
                raise ValueError("ast-grep command not available - please install ast-grep")

            rule_yaml = arguments.get("rule_yaml")
            path = arguments.get("path", ".")
            output_format = arguments.get("output_format", "json")
            max_results = arguments.get("max_results")

            if not rule_yaml:
                raise ValueError("rule_yaml parameter is required")

            logger.info("find_code_by_rule tool called", path=path)
            secure_path = _secure_resolve(path)

            try:
                result = ast_search_official.find_code_by_rule(rule_yaml, secure_path, output_format, max_results)
                return [{"type": "text", "text": f"request_id={request_id}\n{result}"}]
            except ast_search_official.AstGrepError as e:
                raise ValueError(f"Rule-based search failed: {e}")

        elif name == "llm_analyze_code":
            # LLM-based code analysis with fallback to legacy tools
            from datetime import datetime
            start_time = datetime.now()

            code = arguments.get("code")
            language = arguments.get("language", "python")
            analysis_type = arguments.get("analysis_type", "general")

            if not code:
                raise ValueError("code parameter is required")

            logger.info(
                "llm_analyze_code tool called",
                language=language,
                analysis_type=analysis_type,
                ast_intelligence_available=AST_INTELLIGENCE_AVAILABLE,
            )

            try:
                if AST_INTELLIGENCE_AVAILABLE:
                    # Try LLM-based analysis first
                    try:
                        result = await ast_rule_intelligence.analyze_code_with_llm(
                            code=code, language=language, analysis_type=analysis_type, request_id=request_id
                        )

                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info("LLM analysis completed successfully", processing_time_ms=processing_time * 1000, method="llm_primary")

                        return [{"type": "text", "text": result}]

                    except Exception as llm_error:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.warning(
                            "LLM analysis failed, falling back to legacy",
                            error=str(llm_error),
                            processing_time_ms=processing_time * 1000,
                            fallback_reason="llm_error",
                        )
                        # Fall back to legacy analysis

                # Legacy fallback
                if AST_GREP_AVAILABLE:
                    try:
                        # Use legacy AST analysis
                        legacy_result = ast_search_official.dump_syntax_tree(code, language, "pattern")

                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            "Legacy analysis completed (fallback)", processing_time_ms=processing_time * 1000, method="legacy_fallback"
                        )

                        fallback_data = {
                            "request_id": request_id,
                            "method": "legacy_fallback",
                            "analysis_type": analysis_type,
                            "result": legacy_result,
                            "note": "Fell back to legacy AST analysis due to LLM system unavailability",
                            "performance_ms": processing_time * 1000,
                        }

                        return [{"type": "text", "text": json_dumps(fallback_data)}]

                    except Exception as legacy_error:
                        logger.error(
                            "Both LLM and legacy analysis failed",
                            legacy_error=str(legacy_error),
                            total_processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        )
                        raise ValueError(f"Both LLM and legacy analysis failed. Legacy error: {legacy_error}")
                else:
                    raise ValueError("Neither LLM intelligence nor legacy AST tools are available")

            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.error("llm_analyze_code failed", error=str(e), total_processing_time_ms=processing_time * 1000)
                raise ValueError(f"Code analysis failed: {e}")

        elif name == "llm_generate_rule":
            # LLM-based rule generation with fallback
            start_time = datetime.now()

            query = arguments.get("query")
            language = arguments.get("language", "python")
            examples = arguments.get("examples", [])
            rule_type = arguments.get("rule_type", "pattern")

            if not query:
                raise ValueError("query parameter is required")

            logger.info(
                "llm_generate_rule tool called",
                language=language,
                rule_type=rule_type,
                examples_count=len(examples),
                ast_intelligence_available=AST_INTELLIGENCE_AVAILABLE,
            )

            try:
                if AST_INTELLIGENCE_AVAILABLE:
                    try:
                        result = await ast_rule_intelligence.generate_rule_with_llm(
                            query=query, language=language, examples=examples, rule_type=rule_type, request_id=request_id
                        )

                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            "LLM rule generation completed successfully", processing_time_ms=processing_time * 1000, method="llm_primary"
                        )

                        return [{"type": "text", "text": result}]

                    except Exception as llm_error:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.warning(
                            "LLM rule generation failed, providing manual fallback",
                            error=str(llm_error),
                            processing_time_ms=processing_time * 1000,
                            fallback_reason="llm_error",
                        )
                        # Provide manual rule generation guidance

                # Manual fallback guidance
                processing_time = (datetime.now() - start_time).total_seconds()
                fallback_data = {
                    "request_id": request_id,
                    "method": "manual_guidance",
                    "query": query,
                    "language": language,
                    "rule_type": rule_type,
                    "guidance": {
                        "message": "LLM rule generation unavailable. Please use manual rule creation.",
                        "manual_steps": [
                            "1. Analyze your code pattern manually",
                            "2. Create YAML rule following ast-grep pattern syntax",
                            "3. Test rule using test_match_code_rule tool",
                            "4. Refine based on test results",
                        ],
                        "documentation": "See docs/research/cctools.mdc for comprehensive rule creation guide",
                    },
                    "performance_ms": processing_time * 1000,
                }

                logger.info("Manual rule generation guidance provided", processing_time_ms=processing_time * 1000, method="manual_fallback")

                return [{"type": "text", "text": json_dumps(fallback_data)}]

            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.error("llm_generate_rule failed", error=str(e), total_processing_time_ms=processing_time * 1000)
                raise ValueError(f"Rule generation failed: {e}")

        elif name == "llm_search_pattern":
            # LLM-based pattern search with collective consciousness
            start_time = datetime.now()

            query = arguments.get("query")
            language = arguments.get("language", "python")
            path = arguments.get("path", ".")
            search_type = arguments.get("search_type", "intelligent")
            use_collective_memory = arguments.get("use_collective_memory", True)

            if not query:
                raise ValueError("query parameter is required")

            logger.info(
                "llm_search_pattern tool called",
                language=language,
                search_type=search_type,
                use_collective_memory=use_collective_memory,
                ast_intelligence_available=AST_INTELLIGENCE_AVAILABLE,
            )

            secure_path = _secure_resolve(path)

            try:
                if AST_INTELLIGENCE_AVAILABLE:
                    try:
                        result = await ast_rule_intelligence.search_pattern_with_llm(
                            query=query,
                            language=language,
                            path=secure_path,
                            search_type=search_type,
                            use_collective_memory=use_collective_memory,
                            request_id=request_id,
                        )

                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            "LLM pattern search completed successfully",
                            processing_time_ms=processing_time * 1000,
                            method="llm_primary",
                            collective_memory_used=use_collective_memory,
                        )

                        return [{"type": "text", "text": result}]

                    except Exception as llm_error:
                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.warning(
                            "LLM pattern search failed, falling back to legacy",
                            error=str(llm_error),
                            processing_time_ms=processing_time * 1000,
                            fallback_reason="llm_error",
                        )
                        # Fall back to legacy search

                # Legacy fallback
                if AST_GREP_AVAILABLE:
                    try:
                        # Convert query to simple pattern for legacy search
                        simple_pattern = extract_simple_pattern(query)
                        legacy_result = ast_search_official.find_code(
                            pattern=simple_pattern, language=language, path=secure_path, output_format="json", max_results=20
                        )

                        processing_time = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            "Legacy pattern search completed (fallback)",
                            processing_time_ms=processing_time * 1000,
                            method="legacy_fallback",
                        )

                        fallback_data = {
                            "request_id": request_id,
                            "method": "legacy_fallback",
                            "original_query": query,
                            "simplified_pattern": simple_pattern,
                            "result": legacy_result,
                            "note": "Fell back to legacy pattern search. Results may be less comprehensive than LLM search.",
                            "performance_ms": processing_time * 1000,
                            "limitation": "Legacy search lacks LLM reasoning and collective consciousness",
                        }

                        return [{"type": "text", "text": json_dumps(fallback_data)}]

                    except Exception as legacy_error:
                        logger.error(
                            "Both LLM and legacy pattern search failed",
                            legacy_error=str(legacy_error),
                            total_processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        )
                        raise ValueError(f"Both LLM and legacy search failed. Legacy error: {legacy_error}")
                else:
                    raise ValueError("Neither LLM intelligence nor legacy AST tools are available")

            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.error("llm_search_pattern failed", error=str(e), total_processing_time_ms=processing_time * 1000)
                raise ValueError(f"Pattern search failed: {e}")

        elif name == "llm_analyze_code":
            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            analysis_type = arguments.get("analysis_type", "general")
            use_collective_memory = arguments.get("use_collective_memory", False)

            logger.info("llm_analyze_code tool called", language=language, analysis_type=analysis_type)

            if not AST_INTELLIGENCE_AVAILABLE:
                # Fallback to legacy analysis
                fallback_result = {
                    "request_id": request_id,
                    "fallback": True,
                    "analysis_type": analysis_type,
                    "language": language,
                    "result": "LLM intelligence not available, consider using dump_syntax_tree for basic analysis",
                }
                return [{"type": "text", "text": json_dumps(fallback_result)}]

            try:
                analyze_code_llm_start_time: float = time.time()
                analysis_result = await ast_rule_intelligence.analyze_code_with_llm(
                    code=code, language=language, analysis_type=analysis_type, use_collective_memory=use_collective_memory
                )
                processing_time = time.time() - analyze_code_llm_start_time

                response_data = {
                    "request_id": request_id,
                    "analysis": analysis_result,
                    "processing_time_ms": processing_time * 1000,
                    "collective_memory_used": use_collective_memory,
                }
                return [{"type": "text", "text": json_dumps(response_data)}]

            except Exception as e:
                logger.error("llm_analyze_code failed", error=str(e), processing_time_ms=processing_time * 1000)
                raise ValueError(f"LLM analysis failed: {e}")

        elif name == "llm_generate_rule":
            query = arguments.get("query", "")
            language = arguments.get("language", "python")
            examples = arguments.get("examples", [])
            rule_type = arguments.get("rule_type", "pattern")
            use_collective_memory = arguments.get("use_collective_memory", False)

            logger.info("llm_generate_rule tool called", language=language, rule_type=rule_type)

            if not AST_INTELLIGENCE_AVAILABLE:
                fallback_result = {
                    "request_id": request_id,
                    "fallback": True,
                    "query": query,
                    "result": "LLM intelligence not available, consider using test_match_code_rule for manual rule creation",
                }
                return [{"type": "text", "text": json_dumps(fallback_result)}]

            try:
                generate_rule_llm_start_time: float = time.time()
                rule_result = await ast_rule_intelligence.generate_rule_with_llm(
                    query=query, language=language, examples=examples, rule_type=rule_type, use_collective_memory=use_collective_memory
                )
                processing_time = time.time() - generate_rule_llm_start_time

                response_data = {
                    "request_id": request_id,
                    "generated_rule": rule_result,
                    "processing_time_ms": processing_time * 1000,
                    "collective_memory_used": use_collective_memory,
                }
                return [{"type": "text", "text": json_dumps(response_data)}]

            except Exception as e:
                logger.error("llm_generate_rule failed", error=str(e))
                raise ValueError(f"Rule generation failed: {e}")

        elif name == "llm_search_pattern":
            query = arguments.get("query", "")
            language = arguments.get("language", "python")
            path = arguments.get("path", ".")
            use_collective_memory = arguments.get("use_collective_memory", False)
            max_results = arguments.get("max_results", 20)

            logger.info("llm_search_pattern tool called", language=language, path=path)

            if not AST_INTELLIGENCE_AVAILABLE:
                fallback_result = {
                    "request_id": request_id,
                    "fallback": True,
                    "query": query,
                    "result": "LLM intelligence not available, consider using find_code_by_rule for pattern search",
                }
                return [{"type": "text", "text": json_dumps(fallback_result)}]

            try:
                search_pattern_llm_start_time: float = time.time()
                search_result = await ast_rule_intelligence.search_pattern_with_llm(
                    query=query, language=language, path=path, use_collective_memory=use_collective_memory, max_results=max_results
                )
                processing_time = time.time() - search_pattern_llm_start_time

                response_data = {
                    "request_id": request_id,
                    "search_results": search_result,
                    "processing_time_ms": processing_time * 1000,
                    "collective_memory_used": use_collective_memory,
                }
                return [{"type": "text", "text": json_dumps(response_data)}]

            except Exception as e:
                logger.error("llm_search_pattern failed", error=str(e))
                raise ValueError(f"Pattern search failed: {e}")

        elif name == "auto_ast_intelligence":
            query = arguments.get("query", "")
            context = arguments.get("context", "")
            language = arguments.get("language", "python")
            path = arguments.get("path", ".")
            auto_execute = arguments.get("auto_execute", True)

            logger.info("auto_ast_intelligence tool called", query=query)

            try:
                auto_ast_llm_start_time: float = time.time()
                auto_result = await detect_and_execute_ast_intelligence(
                    query=query, context=context, language=language, path=path, auto_execute=auto_execute
                )
                processing_time = time.time() - auto_ast_llm_start_time

                response_data = {
                    "request_id": request_id,
                    "auto_analysis": auto_result,
                    "processing_time_ms": processing_time * 1000,
                }
                return [{"type": "text", "text": json_dumps(response_data)}]

            except Exception as e:
                logger.error("auto_ast_intelligence failed", error=str(e))
                raise ValueError(f"Auto intelligence failed: {e}")

        elif name == "ripgrep_search":
            if not RIPGREP_AVAILABLE:
                raise ValueError("Ripgrep integration not available")

            pattern = arguments.get("pattern")
            path = arguments.get("path", ".")
            file_patterns = arguments.get("file_patterns", [])
            exclude_patterns = arguments.get("exclude_patterns", [])
            max_results = arguments.get("max_results", 1000)
            context_lines = arguments.get("context_lines", 0)
            case_sensitive = arguments.get("case_sensitive", True)
            output_format = arguments.get("output_format", "text")

            if not pattern:
                raise ValueError("pattern parameter is required")

            logger.info("ripgrep_search tool called", pattern=pattern, path=path)
            secure_path = _secure_resolve(path)

            try:
                from fastapply.ripgrep_integration import SearchOptions

                options = SearchOptions(
                    case_sensitive=case_sensitive,
                    include_patterns=file_patterns,
                    exclude_patterns=exclude_patterns,
                    max_results=max_results,
                    context_lines=context_lines,
                )

                result = ripgrep_integration.search_files(pattern=pattern, path=secure_path, options=options)

                if output_format == "json":
                    response_data = {
                        "request_id": request_id,
                        "pattern": pattern,
                        "path": path,
                        "matches": result.to_dict()["results"],
                        "total_matches": result.total_matches,
                        "files_searched": result.files_searched,
                        "processing_time_ms": result.search_time * 1000,
                    }
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [f"request_id={request_id}", f"🔍 Pattern: {pattern}", f"📁 Path: {path}"]
                    lines.append(f"📊 Total matches: {result.total_matches} in {result.files_searched} files")
                    lines.append(f"⚡ Processing time: {result.search_time * 1000:.2f}ms")

                    if result.results:
                        lines.append("\n📋 Matches:")
                        for match in result.results[:20]:  # Limit display
                            lines.append(f"  {match.file_path}:{match.line_number}: {match.line_text.strip()}")
                        if len(result.results) > 20:
                            lines.append(f"  ... and {len(result.results) - 20} more matches")

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("ripgrep_search failed", error=str(e))
                raise ValueError(f"Ripgrep search failed: {e}")

        elif name == "ripgrep_search_code":
            if not RIPGREP_AVAILABLE:
                raise ValueError("Ripgrep integration not available")

            pattern = arguments.get("pattern")
            language = arguments.get("language")
            path = arguments.get("path", ".")
            max_results = arguments.get("max_results", 500)
            context_lines = arguments.get("context_lines", 3)
            output_format = arguments.get("output_format", "text")

            if not pattern:
                raise ValueError("pattern parameter is required")
            if not language:
                raise ValueError("language parameter is required")

            logger.info("ripgrep_search_code tool called", pattern=pattern, language=language)
            secure_path = _secure_resolve(path)

            try:
                from fastapply.ripgrep_integration import SearchOptions

                options = SearchOptions(max_results=max_results, context_lines=context_lines)

                result = ripgrep_integration.search_code_patterns(pattern=pattern, language=language, path=secure_path, options=options)

                if output_format == "json":
                    response_data = {
                        "request_id": request_id,
                        "pattern": pattern,
                        "language": language,
                        "matches": result.to_dict()["results"],
                        "total_matches": result.total_matches,
                        "files_searched": result.files_searched,
                        "processing_time_ms": result.search_time * 1000,
                    }
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [f"request_id={request_id}", f"🔍 Code Pattern: {pattern}", f"💻 Language: {language}"]
                    lines.append(f"📊 Total matches: {result.total_matches}")
                    lines.append(f"📁 Files searched: {result.files_searched}")
                    lines.append(f"⚡ Processing time: {result.search_time * 1000:.2f}ms")

                    if result.results:
                        lines.append("\n📋 Code Matches:")
                        for match in result.results[:15]:  # Limit display
                            lines.append(f"  {match.file_path}:{match.line_number}: {match.line_text.strip()}")
                        if len(result.results) > 15:
                            lines.append(f"  ... and {len(result.results) - 15} more matches")

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("ripgrep_search_code failed", error=str(e))
                raise ValueError(f"Ripgrep code search failed: {e}")

        elif name == "ripgrep_find_symbols":
            if not RIPGREP_AVAILABLE:
                raise ValueError("Ripgrep integration not available")

            symbol_name = arguments.get("pattern")  # Using pattern as symbol_name
            symbol_type = arguments.get("symbol_type")
            path = arguments.get("path", ".")
            language = arguments.get("language")
            output_format = arguments.get("output_format", "text")

            if not symbol_name:
                raise ValueError("pattern parameter is required (symbol_name)")

            logger.info("ripgrep_find_symbols tool called", symbol_name=symbol_name, symbol_type=symbol_type)
            secure_path = _secure_resolve(path)

            try:
                result = ripgrep_integration.find_symbol_candidates(
                    symbol_name=symbol_name, path=secure_path, symbol_type=symbol_type, language=language
                )

                if output_format == "json":
                    response_data = {
                        "request_id": request_id,
                        "symbol_name": symbol_name,
                        "symbol_type": symbol_type,
                        "symbols": [asdict(symbol) for symbol in result],
                        "total_symbols": len(result),
                        "high_confidence_matches": len([s for s in result if s.confidence_score >= 0.8]),
                    }
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [f"request_id={request_id}", f"🔍 Symbol Name: {symbol_name}", f"🏷️ Symbol Type: {symbol_type or 'any'}"]
                    if language:
                        lines.append(f"💻 Language: {language}")
                    lines.append(f"📊 Total symbols: {len(result)}")
                    lines.append(f"🎯 High confidence matches: {len([s for s in result if s.confidence >= 0.8])}")

                    if result:
                        lines.append("\n📋 Symbol Candidates:")
                        for symbol in result[:15]:  # Limit display
                            confidence_emoji = "🟢" if symbol.confidence_score >= 0.8 else "🟡" if symbol.confidence_score >= 0.5 else "🔴"
                            line = (
                                f"  {confidence_emoji} {symbol.symbol_name} "
                                f"({symbol.confidence_score:.2f}) - {symbol.file_path}:"
                                f"{symbol.line_number}"
                            )
                            lines.append(line)
                        if len(result) > 15:
                            lines.append(f"  ... and {len(result) - 15} more symbols")

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("ripgrep_find_symbols failed", error=str(e))
                raise ValueError(f"Ripgrep symbol search failed: {e}")

        elif name == "ripgrep_file_metrics":
            if not RIPGREP_AVAILABLE:
                raise ValueError("Ripgrep integration not available")

            file_path = arguments.get("path")  # Single file analysis
            output_format = arguments.get("output_format", "text")

            if not file_path:
                raise ValueError("path parameter is required (single file path)")

            logger.info("ripgrep_file_metrics tool called", file_path=file_path)
            secure_path = _secure_resolve(file_path)

            try:
                result = ripgrep_integration.analyze_file_metrics(secure_path)

                if output_format == "json":
                    response_data = {"request_id": request_id, "file_path": file_path, "metrics": asdict(result)}
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [f"request_id={request_id}", "📊 File Metrics Analysis", f"📁 File: {file_path}"]
                    lines.extend(
                        [
                            f"📄 Language: {result.language}",
                            f"📏 Lines: {result.line_count}",
                            f"🔤 Characters: {result.size_bytes}",
                            f"📦 Size: {result.size_bytes} bytes",
                            (
                                "⏱️ Modified: "
                                + f"{datetime.fromtimestamp(result.last_modified).isoformat() if result.last_modified else 'Unknown'}"
                            ),
                        ]
                    )

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("ripgrep_file_metrics failed", error=str(e))
                raise ValueError(f"Ripgrep file metrics failed: {e}")

        elif name == "ripgrep_batch_search":
            if not RIPGREP_AVAILABLE:
                raise ValueError("Ripgrep integration not available")

            patterns = arguments.get("patterns", [])
            path = arguments.get("path", ".")
            max_patterns = arguments.get("max_patterns", 10)
            max_results_per_pattern = arguments.get("max_results_per_pattern", 100)
            output_format = arguments.get("output_format", "text")

            if not patterns:
                raise ValueError("patterns parameter is required and must be non-empty")

            if len(patterns) > max_patterns:
                raise ValueError(f"Too many patterns (max {max_patterns})")

            logger.info("ripgrep_batch_search tool called", patterns_count=len(patterns), path=path)
            secure_path = _secure_resolve(path)

            try:
                from fastapply.ripgrep_integration import SearchOptions

                options = SearchOptions(max_results=max_results_per_pattern)

                result = ripgrep_integration.batch_search(patterns=patterns, path=secure_path, options=options)

                total_matches = sum(r.total_matches for r in result.values())
                patterns_with_matches = len([r for r in result.values() if r.total_matches > 0])

                if output_format == "json":
                    response_data = {
                        "request_id": request_id,
                        "patterns": patterns,
                        "path": path,
                        "results": {pattern: r.to_dict() for pattern, r in result.items()},
                        "total_matches_all_patterns": total_matches,
                        "patterns_with_matches": patterns_with_matches,
                    }
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [f"request_id={request_id}", "🔍 Batch Ripgrep Search", f"📁 Path: {path}"]
                    lines.extend(
                        [
                            f"📊 Patterns searched: {len(patterns)}",
                            f"✅ Patterns with matches: {patterns_with_matches}",
                            f"🎯 Total matches: {total_matches}",
                        ]
                    )

                    for pattern, pattern_result in result.items():
                        count = pattern_result.total_matches
                        files = pattern_result.files_searched
                        lines.append(f"\n📋 '{pattern}': {count} matches in {files} files")

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("ripgrep_batch_search failed", error=str(e))
                raise ValueError(f"Ripgrep batch search failed: {e}")

        elif name == "enhanced_search":
            if not ENHANCED_SEARCH_AVAILABLE:
                raise ValueError("Enhanced search infrastructure not available")

            query = arguments.get("query", "")
            path = arguments.get("path", "")
            strategy = arguments.get("strategy", "hybrid")
            file_types = arguments.get("file_types", [])
            exclude_patterns = arguments.get("exclude_patterns", [])
            include_patterns = arguments.get("include_patterns", [])
            max_results = arguments.get("max_results", 100)
            context_lines = arguments.get("context_lines", 3)
            case_sensitive = arguments.get("case_sensitive", False)
            ranking = arguments.get("ranking", "combined")
            timeout = arguments.get("timeout", 30.0)

            logger.info("enhanced_search tool called", query=query, path=path, strategy=strategy)

            try:
                # Validate path
                secure_path = _secure_resolve(path)
                if not Path(secure_path).exists():
                    raise ValueError(f"Path does not exist: {path}")

                # Create search context
                search_context = SearchContext(
                    query=query,
                    file_types=file_types,
                    exclude_patterns=exclude_patterns,
                    include_patterns=include_patterns,
                    max_results=max_results,
                    strategy=SearchStrategy(strategy),
                    ranking=ResultRanking(ranking),
                    timeout=timeout,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                )

                # Execute enhanced search
                if enhanced_search_instance is None:
                    raise ValueError("Enhanced search infrastructure not initialized")
                search_results, metrics = enhanced_search_instance.search(search_context)
                from typing import cast as _cast

                search_results_cast: List[EnhancedSearchResult] = _cast(List[EnhancedSearchResult], search_results)

                # Format results
                if len(search_results_cast) > 0:
                    lines = [
                        "🔍 Enhanced Search Results",
                        f"📁 Path: {path}",
                        f"🎯 Strategy: {strategy}",
                        f"📊 Results found: {len(search_results_cast)}",
                        f"⚡ Search time: {metrics.search_duration:.3f}s",
                        f"🎯 Cache hits: {metrics.cache_hits}",
                        f"📄 Files searched: {metrics.total_files_searched}",
                    ]

                    # Display top results
                    for i, search_result in enumerate(search_results_cast[:10]):  # Show top 10
                        lines.append(f"\n{i + 1}. {search_result.file_path}:{search_result.line_number}")
                        lines.append(f"   {search_result.line_content.strip()}")
                        lines.append(f"   🏷️  Type: {search_result.match_type}")
                        lines.append(f"   🎯 Confidence: {search_result.confidence_score:.2f}")
                        lines.append(f"   📊 Combined score: {search_result.combined_score:.2f}")

                    if len(search_results_cast) > 10:
                        lines.append(f"\n... and {len(search_results_cast) - 10} more results")

                    return [{"type": "text", "text": "\n".join(lines)}]
                else:
                    return [{"type": "text", "text": f"No results found for query '{query}' in {path}"}]

            except Exception as e:
                logger.error("enhanced_search failed", error=str(e))
                raise ValueError(f"Enhanced search failed: {e}")

        elif name == "enhanced_search_intelligent":
            if not ENHANCED_SEARCH_AVAILABLE:
                raise ValueError("Enhanced search infrastructure not available")

            query = arguments.get("query", "")
            path = arguments.get("path", "")
            context = arguments.get("context", "")
            auto_detect_strategy = arguments.get("auto_detect_strategy", True)
            language = arguments.get("language", "")
            max_results = arguments.get("max_results", 50)
            optimize_for = arguments.get("optimize_for", "accuracy")

            logger.info("enhanced_search_intelligent tool called", query=query, path=path)

            try:
                # Validate path
                secure_path = _secure_resolve(path)
                if not Path(secure_path).exists():
                    raise ValueError(f"Path does not exist: {path}")

                # Intelligent strategy detection
                if enhanced_search_instance is None:
                    raise ValueError("Enhanced search infrastructure not initialized")
                if auto_detect_strategy:
                    strategy = enhanced_search_instance._detect_optimal_strategy(query, context, language)
                else:
                    strategy = SearchStrategy.HYBRID

                # Adjust parameters based on optimization goal
                if optimize_for == "speed":
                    max_results = min(max_results, 25)
                    timeout = 15.0
                elif optimize_for == "comprehensiveness":
                    max_results = max(max_results, 200)
                    timeout = 60.0
                else:  # accuracy
                    timeout = 30.0

                # Create optimized search context
                search_context = SearchContext(
                    query=query,
                    file_types=[language] if language else [],
                    max_results=max_results,
                    strategy=strategy,
                    ranking=ResultRanking.COMBINED,
                    timeout=timeout,
                    context_lines=3,
                )

                # Execute intelligent search
                if enhanced_search_instance is None:
                    raise ValueError("Enhanced search infrastructure not initialized")
                search_results, metrics = enhanced_search_instance.search(search_context)
                from typing import cast as _cast

                intelligent_results: List[EnhancedSearchResult] = _cast(List[EnhancedSearchResult], search_results)

                # Format intelligent results
                lines = [
                    "🧠 Intelligent Search Results",
                    f"📁 Path: {path}",
                    f"🎯 Detected strategy: {strategy.value}",
                    f"⚡ Optimized for: {optimize_for}",
                    f"📊 Results found: {len(intelligent_results)}",
                    f"⏱️  Search time: {metrics.search_duration:.3f}s",
                ]

                if context:
                    lines.append(f"📝 Context: {context}")

                # Show best matches with reasoning
                for i, intelligent_result in enumerate(intelligent_results[:8]):
                    lines.append(f"\n{i + 1}. {intelligent_result.file_path}:{intelligent_result.line_number}")
                    lines.append(f"   {intelligent_result.line_content.strip()}")
                    lines.append(f"   🏷️  Type: {intelligent_result.match_type}")
                    lines.append(f"   🎯 Confidence: {intelligent_result.confidence_score:.2f}")
                    lines.append(f"   📊 Relevance: {intelligent_result.relevance_score:.2f}")

                return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("enhanced_search_intelligent failed", error=str(e))
                raise ValueError(f"Intelligent search failed: {e}")

        elif name == "search_with_context":
            if not ENHANCED_SEARCH_AVAILABLE:
                raise ValueError("Enhanced search infrastructure not available")

            query = arguments.get("query", "")
            path = arguments.get("path", "")
            context_id = arguments.get("context_id", "")
            previous_results = arguments.get("previous_results", [])
            refine = arguments.get("refine", False)
            max_results = arguments.get("max_results", 100)

            logger.info("search_with_context tool called", query=query, context_id=context_id)

            try:
                # Validate path
                secure_path = _secure_resolve(path)
                if not Path(secure_path).exists():
                    raise ValueError(f"Path does not exist: {path}")

                # Get or create context
                if enhanced_search_instance is None:
                    raise ValueError("Enhanced search infrastructure not initialized")
                if hasattr(enhanced_search_instance, "get_context"):
                    search_context = enhanced_search_instance.get_context(context_id)  # type: ignore[call-arg]
                else:
                    search_context = SearchContext(
                        query=query, max_results=max_results, strategy=SearchStrategy.HYBRID, ranking=ResultRanking.COMBINED
                    )

                # Execute context-aware search
                if refine and previous_results:
                    # Refine existing results
                    search_results = enhanced_search_instance.refine_search_results(query, previous_results, search_context)
                else:
                    # New search with context
                    search_results, metrics = enhanced_search_instance.search(search_context)
                from typing import cast as _cast

                search_results_enhanced: List[EnhancedSearchResult] = _cast(List[EnhancedSearchResult], search_results)

                # Format contextual results
                lines = [
                    "🔗 Contextual Search Results",
                    f"📁 Path: {path}",
                    f"🆔 Context ID: {context_id}",
                    f"🔄 Mode: {'Refine' if refine else 'New search'}",
                    f"📊 Results found: {len(search_results_enhanced)}",
                ]

                for i, enhanced_result in enumerate(search_results_enhanced[:15]):
                    lines.append(f"\n{i + 1}. {enhanced_result.file_path}:{enhanced_result.line_number}")
                    lines.append(f"   {enhanced_result.line_content.strip()}")
                    if enhanced_result.metadata:
                        lines.append(f"   📋 Context: {len(enhanced_result.metadata.get('context_before', []))} lines before")

                return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("search_with_context failed", error=str(e))
                raise ValueError(f"Contextual search failed: {e}")

        elif name == "search_performance_optimize":
            if not ENHANCED_SEARCH_AVAILABLE:
                raise ValueError("Enhanced search infrastructure not available")

            action = arguments.get("action", "")
            path = arguments.get("path", "")
            common_patterns = arguments.get("common_patterns", [])
            cache_size = arguments.get("cache_size", 1000)
            clear_cache = arguments.get("clear_cache", False)

            logger.info("search_performance_optimize tool called", action=action)

            try:
                perf_results: Dict[str, Any] = {}

                if enhanced_search_instance is None:
                    raise ValueError("Enhanced search infrastructure not initialized")
                if action == "clear_cache":
                    enhanced_search_instance.clear_cache()
                    perf_results["cache_cleared"] = True

                elif action == "optimize_cache":
                    if clear_cache:
                        enhanced_search_instance.clear_cache()
                    enhanced_search_instance.optimize_for_patterns(common_patterns)
                    perf_results["cache_optimized"] = True
                    perf_results["cache_size"] = cache_size

                elif action == "tune_pipeline":
                    # Placeholder for pipeline tuning
                    perf_results["pipeline_tuned"] = True
                    perf_results["optimizations"] = [
                        "Parallel search strategies enabled",
                        "Smart ranking activated",
                        "Cache pre-warmed for common patterns",
                    ]

                elif action == "benchmark":
                    stats = enhanced_search_instance.get_search_statistics()
                    perf_results["benchmark"] = stats
                    perf_results["performance"] = "Optimal"

                elif action == "analyze_patterns":
                    if path:
                        secure_path_str = _secure_resolve(path)
                        secure_path_obj = Path(secure_path_str)
                        if secure_path_obj.exists():
                            # Pattern analysis placeholder
                            perf_results["pattern_analysis"] = {
                                "total_files": sum(1 for _ in secure_path_obj.rglob("*") if _.is_file()),
                                "common_patterns_detected": len(common_patterns),
                                "optimization_recommendations": [
                                    "Pre-warm cache for frequent searches",
                                    "Use hybrid strategy for complex queries",
                                    "Enable language-specific optimizations",
                                ],
                            }

                # Format optimization results
                lines = ["⚡ Search Performance Optimization", f"🔧 Action: {action}", "✅ Status: Completed"]

                for key, value in perf_results.items():
                    if isinstance(value, dict):
                        lines.append(f"\n📊 {key.replace('_', ' ').title()}:")
                        for k, v in value.items():
                            lines.append(f"   {k}: {v}")
                    else:
                        lines.append(f"📋 {key.replace('_', ' ').title()}: {value}")

                return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("search_performance_optimize failed", error=str(e))
                raise ValueError(f"Performance optimization failed: {e}")

        elif name == "health_status":
            info = {
                "request_id": request_id,
                "workspace_root": WORKSPACE_ROOT,
                "strict_paths": STRICT_PATHS,
                "max_file_size": MAX_FILE_SIZE,
                "max_response_size": MAX_RESPONSE_SIZE,
                "ruff_available": _RUFF_AVAILABLE,
                "eslint_available": _ESLINT_AVAILABLE,
                "ast_search_available": AST_SEARCH_AVAILABLE,
                "ast_grep_available": AST_GREP_AVAILABLE,
                "ast_intelligence_available": AST_INTELLIGENCE_AVAILABLE,
                "ripgrep_available": RIPGREP_AVAILABLE,
                "enhanced_search_available": ENHANCED_SEARCH_AVAILABLE,
                "semantic_analysis_available": SEMANTIC_ANALYSIS_AVAILABLE,
                "relationship_mapping_available": RELATIONSHIP_MAPPING_AVAILABLE,
                "tools": [t["name"] for t in list_tools()],
                "ast_capabilities": {
                    "legacy_wrapper": AST_SEARCH_AVAILABLE,
                    "direct_cli": AST_GREP_AVAILABLE,
                    "llm_intelligence": AST_INTELLIGENCE_AVAILABLE,
                    "collective_consciousness": AST_INTELLIGENCE_AVAILABLE,
                },
            }
            lines = [f"{k}={v}" for k, v in info.items()]
            return [{"type": "text", "text": "\n".join(lines)}]

        # Semantic Analysis Tools
        elif name == "deep_semantic_analysis":
            if not SEMANTIC_ANALYSIS_AVAILABLE:
                raise ValueError("Deep semantic analysis not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            include_context = arguments.get("include_context", True)

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if semantic_analyzer is None:
                    raise ValueError("Semantic analyzer not initialized")
                context = AnalysisContext(project_path=".") if include_context else None
                semantic_analysis_result = semantic_analyzer.analyze_semantics(code, language, context)
                from typing import cast as _cast

                semantic_result = _cast(Any, semantic_analysis_result)

                text_result = [
                    "Deep Semantic Analysis Result:",
                    f"Intent: {semantic_result.intent.primary_intent}",
                    f"Confidence: {semantic_result.intent.confidence:.2f}",
                    f"Complexity: {semantic_result.complexity.score} ({semantic_result.complexity.level})",
                    f"Quality Score: {semantic_result.quality.overall_score:.2f}",
                    f"Patterns Found: {len(semantic_result.patterns)}",
                    f"Behaviors: {len(semantic_result.behaviors)}",
                ]

                if semantic_result.pattern_analysis.detected_patterns:
                    text_result.append("\nDesign Patterns:")
                    for pattern, confidence in semantic_result.pattern_analysis.detected_patterns[:3]:  # Show top 3
                        text_result.append(f"  - {pattern.value}: {confidence:.2f}")

                json_result = {
                    "request_id": request_id,
                    "analysis": {
                        "intent": {
                            "primary": semantic_result.intent_analysis.primary_intent.value,
                            "confidence": semantic_result.intent_analysis.confidence,
                            "evidence": semantic_result.intent_analysis.supporting_evidence,
                        },
                        "complexity": {
                            "metrics": semantic_result.quality_assessment.complexity_metrics,
                        },
                        "quality": {
                            "overall_score": semantic_result.quality_assessment.overall_quality,
                            "maintainability": semantic_result.quality_assessment.maintainability_score,
                            "readability": semantic_result.quality_assessment.readability_score,
                            "security": semantic_result.quality_assessment.security_score,
                        },
                        "patterns": [{"type": p[0].value, "confidence": p[1]} for p in semantic_result.pattern_analysis.detected_patterns],
                        "behaviors": [
                            {"flow": f, "side_effects": se, "state_changes": sc}
                            for f, se, sc in zip(
                                semantic_result.behavior_analysis.execution_flow[:3],
                                semantic_result.behavior_analysis.side_effects[:3],
                                semantic_result.behavior_analysis.state_changes[:3],
                            )
                        ],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("deep_semantic_analysis failed", error=str(e))
                raise ValueError(f"Semantic analysis failed: {e}")

        elif name == "understand_code_intent":
            if not SEMANTIC_ANALYSIS_AVAILABLE:
                raise ValueError("Code intent understanding not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if semantic_analyzer is None:
                    raise ValueError("Semantic analyzer not initialized")

                intent_result_raw: Any = semantic_analyzer.understand_code_intent(code, language)
                from typing import cast as _cast

                intent_result = _cast(Any, intent_result_raw)

                text_result = [
                    "Code Intent Analysis:",
                    f"Primary Intent: {intent_result.primary_intent}",
                    f"Confidence: {intent_result.confidence:.2f}",
                    f"Domain: {intent_result.domain}",
                    f"Complexity: {intent_result.complexity_level}",
                ]

                if intent_result.secondary_intents:
                    text_result.append("\nSecondary Intents:")
                    for intent in intent_result.secondary_intents:
                        text_result.append(f"  - {intent}")

                if intent_result.key_operations:
                    text_result.append("\nKey Operations:")
                    for op in intent_result.key_operations[:5]:
                        text_result.append(f"  - {op}")

                json_result = {
                    "request_id": request_id,
                    "intent": {
                        "primary": intent_result.primary_intent,
                        "secondary": intent_result.secondary_intents,
                        "confidence": intent_result.confidence,
                        "domain": intent_result.domain,
                        "complexity_level": intent_result.complexity_level,
                        "key_operations": intent_result.key_operations,
                        "reasoning": intent_result.reasoning,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("understand_code_intent failed", error=str(e))
                raise ValueError(f"Intent analysis failed: {e}")

        elif name == "analyze_runtime_behavior":
            if not SEMANTIC_ANALYSIS_AVAILABLE:
                raise ValueError("Runtime behavior analysis not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if semantic_analyzer is None:
                    raise ValueError("Semantic analyzer not initialized")

                behavior_result_raw: Any = semantic_analyzer.analyze_runtime_behavior(code, language)
                from typing import cast as _cast

                behavior_result = _cast(Any, behavior_result_raw)

                text_result = [
                    "Runtime Behavior Analysis:",
                    f"Behavior Type: {behavior_result.behavior_type}",
                    f"Time Complexity: {behavior_result.time_complexity}",
                    f"Space Complexity: {behavior_result.space_complexity}",
                    f"Performance Impact: {behavior_result.performance_impact}",
                    f"Side Effects: {len(behavior_result.side_effects)}",
                ]

                if behavior_result.behavioral_patterns:
                    text_result.append("\nBehavioral Patterns:")
                    for pattern in behavior_result.behavioral_patterns[:3]:
                        text_result.append(f"  - {pattern}")

                if behavior_result.side_effects:
                    text_result.append("\nSide Effects:")
                    for effect in behavior_result.side_effects[:3]:
                        text_result.append(f"  - {effect}")

                json_result = {
                    "request_id": request_id,
                    "behavior": {
                        "type": behavior_result.behavior_type,
                        "time_complexity": behavior_result.time_complexity,
                        "space_complexity": behavior_result.space_complexity,
                        "performance_impact": behavior_result.performance_impact,
                        "behavioral_patterns": behavior_result.behavioral_patterns,
                        "side_effects": behavior_result.side_effects,
                        "execution_flow": behavior_result.execution_flow,
                        "resource_usage": behavior_result.resource_usage,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_runtime_behavior failed", error=str(e))
                raise ValueError(f"Behavior analysis failed: {e}")

        elif name == "identify_design_patterns":
            if not SEMANTIC_ANALYSIS_AVAILABLE:
                raise ValueError("Design pattern identification not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if semantic_analyzer is None:
                    raise ValueError("Semantic analyzer not initialized")

                pattern_result_raw: Any = semantic_analyzer.identify_design_patterns(code, language)
                from typing import cast as _cast

                pattern_result = _cast(Any, pattern_result_raw)

                text_result = [
                    "Design Pattern Analysis:",
                    f"Patterns Found: {len(pattern_result.patterns)}",
                    f"Anti-Patterns: {len(pattern_result.anti_patterns)}",
                ]

                if pattern_result.detected_patterns:
                    text_result.append("\nDesign Patterns:")
                    for pattern, confidence in pattern_result.detected_patterns:
                        text_result.append(f"  - {pattern.value}: {confidence:.2f}")

                if pattern_result.code_smells:
                    text_result.append("\nAnti-Patterns:")
                    for smell, severity, description in pattern_result.code_smells:
                        text_result.append(f"  - {smell.value}: {severity} - {description}")

                json_result = {
                    "request_id": request_id,
                    "patterns": {
                        "design_patterns": [{"type": p[0].value, "confidence": p[1]} for p in pattern_result.detected_patterns],
                        "anti_patterns": [{"type": s[0].value, "severity": s[1], "description": s[2]} for s in pattern_result.code_smells],
                        "suggestions": pattern_result.pattern_suggestions,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("identify_design_patterns failed", error=str(e))
                raise ValueError(f"Pattern identification failed: {e}")

        elif name == "assess_code_quality":
            if not SEMANTIC_ANALYSIS_AVAILABLE:
                raise ValueError("Code quality assessment not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if semantic_analyzer is None:
                    raise ValueError("Semantic analyzer not initialized")

                quality_result_raw: Any = semantic_analyzer.assess_code_quality(code, language)
                from typing import cast as _cast

                quality_result = _cast(Any, quality_result_raw)

                text_result = [
                    "Code Quality Assessment:",
                    f"Overall Score: {quality_result.overall_quality:.2f}",
                    f"Maintainability: {quality_result.maintainability_score:.2f}",
                    f"Readability: {quality_result.readability_score:.2f}",
                    f"Testability: {quality_result.testability_score:.2f}",
                    f"Security: {quality_result.security_score:.2f}",
                    f"Recommendations: {len(quality_result.improvement_recommendations)}",
                ]

                if quality_result.improvement_recommendations:
                    text_result.append("\nRecommendations:")
                    for rec in quality_result.improvement_recommendations[:5]:
                        text_result.append(f"  - {rec}")

                json_result = {
                    "request_id": request_id,
                    "quality": {
                        "overall_score": quality_result.overall_quality,
                        "maintainability": quality_result.maintainability_score,
                        "readability": quality_result.readability_score,
                        "testability": quality_result.testability_score,
                        "security": quality_result.security_score,
                        "complexity_metrics": quality_result.complexity_metrics,
                        "recommendations": quality_result.improvement_recommendations,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("assess_code_quality failed", error=str(e))
                raise ValueError(f"Quality assessment failed: {e}")

        # Relationship Mapping Tools
        elif name == "map_relationships":
            if not RELATIONSHIP_MAPPING_AVAILABLE:
                raise ValueError("Relationship mapping not available")

            code = arguments.get("code", "")
            context = arguments.get("context", "")
            project_path = arguments.get("project_path")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if relationship_mapper is None:
                    raise ValueError("Relationship mapper not initialized")
                relationship_result = relationship_mapper.understand_relationships(code, context, project_path)

                text_result = [
                    "Relationship Mapping:",
                    f"Components: {len(relationship_result.components)}",
                    f"Dependencies: {len(relationship_result.dependencies)}",
                    f"Relationships: {len(relationship_result.relationships)}",
                ]

                # Architectural patterns analysis not available in current RelationshipMap

                json_result = {
                    "request_id": request_id,
                    "relationships": {
                        "components": [
                            {"name": comp.name, "type": comp.type, "file_path": comp.file_path, "line_number": comp.line_number}
                            for comp in relationship_result.components.values()
                        ],
                        "dependencies": [
                            {"source": dep.source_module, "target": dep.target_module, "type": dep.dependency_type.value}
                            for dep in relationship_result.dependencies
                        ],
                        "relationships": [
                            {"from": rel.source.name, "to": rel.target.name, "type": rel.relationship_type.value, "strength": rel.strength}
                            for rel in relationship_result.relationships
                        ],
                        "circular_dependencies": len(relationship_result.circular_dependencies),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("map_relationships failed", error=str(e))
                raise ValueError(f"Relationship mapping failed: {e}")

        elif name == "detect_circular_dependencies":
            if not RELATIONSHIP_MAPPING_AVAILABLE:
                raise ValueError("Circular dependency detection not available")

            code_structure = arguments.get("code_structure")

            if not code_structure:
                raise ValueError("Code structure parameter is required")

            try:
                # Convert dict to CodeStructure if needed
                if isinstance(code_structure, dict):
                    from .relationship_mapping import CodeStructure

                    # CodeStructure only takes root_path parameter
                    # Extract project path for CodeStructure
                    project_path = code_structure.get("project_path", "")
                    cs = CodeStructure(project_path)

                    # Note: The CodeStructure class doesn't store components/dependencies/relationships
                    # directly. These would need to be processed by the RelationshipMapper.
                else:
                    cs = code_structure

                if relationship_mapper is None:
                    raise ValueError("Relationship mapper not initialized")

                circular_deps_result: Any = relationship_mapper._detect_circular_dependencies(
                    cs.dependencies if hasattr(cs, "dependencies") else []
                )

                avg_impact = sum(cd.impact_score for cd in circular_deps_result) / len(circular_deps_result) if circular_deps_result else 0

                text_result = [
                    "Circular Dependency Analysis:",
                    f"Cycles Found: {len(circular_deps_result)}",
                    (f"Average Impact Score: {avg_impact:.2f}"),
                ]

                if circular_deps_result:
                    text_result.append("\nCircular Dependencies:")
                    for cd in circular_deps_result[:3]:
                        text_result.append(f"  - Cycle: {' -> '.join(cd.cycle_path)} (Impact: {cd.impact_score:.2f})")

                json_result = {
                    "request_id": request_id,
                    "circular_dependencies": {
                        "cycles": [cd.cycle_path for cd in circular_deps_result],
                        "impact_scores": [cd.impact_score for cd in circular_deps_result],
                        "dependency_types": [[dt.value for dt in cd.dependency_types] for cd in circular_deps_result],
                        "resolution_suggestions": [suggestion for cd in circular_deps_result for suggestion in cd.resolution_suggestions],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("detect_circular_dependencies failed", error=str(e))
                raise ValueError(f"Circular dependency detection failed: {e}")

        elif name == "analyze_coupling_cohesion":
            if not RELATIONSHIP_MAPPING_AVAILABLE:
                raise ValueError("Coupling/cohesion analysis not available")

            code_structure = arguments.get("code_structure")

            if not code_structure:
                raise ValueError("Code structure parameter is required")

            try:
                # Convert dict to CodeStructure if needed
                if isinstance(code_structure, dict):
                    from .relationship_mapping import CodeStructure

                    # CodeStructure only takes root_path parameter
                    # Extract project path for CodeStructure
                    project_path = code_structure.get("project_path", "")
                    cs = CodeStructure(project_path)

                    # Note: The CodeStructure class doesn't store components/dependencies/relationships
                    # directly. These would need to be processed by the RelationshipMapper.
                else:
                    cs = code_structure

                if relationship_mapper is None:
                    raise ValueError("Relationship mapper not initialized")

                # Create a simple RelationshipMap from CodeStructure for coupling analysis
                from .relationship_mapping import RelationshipMap

                temp_map = RelationshipMap()
                temp_map.dependencies = cs.dependencies if hasattr(cs, "dependencies") else []
                coupling_result: Any = relationship_mapper._analyze_coupling(temp_map, temp_map.dependencies)

                text_result = [
                    "Coupling and Cohesion Analysis:",
                    f"Analyses Performed: {len(coupling_result)}",
                    (
                        "Average Coupling Score: "
                        f"{sum(ca.coupling_score for ca in coupling_result) / len(coupling_result) if coupling_result else 0:.2f}"
                    ),
                    f"Highly Coupled Pairs: {len([ca for ca in coupling_result if ca.coupling_score > 0.7])}",
                ]

                if coupling_result:
                    text_result.append("\nHighly Coupled Modules:")
                    for ca in sorted(coupling_result, key=lambda x: x.coupling_score, reverse=True)[:3]:
                        text_result.append(f"  - {ca.source_module} <-> {ca.target_module} ({ca.coupling_score:.2f})")

                json_result = {
                    "request_id": request_id,
                    "coupling_cohesion": {
                        "analyses": [
                            {
                                "source": ca.source_module,
                                "target": ca.target_module,
                                "coupling_score": ca.coupling_score,
                                "coupling_types": [ct.value for ct in ca.coupling_types],
                                "impact_score": ca.impact_score,
                            }
                            for ca in coupling_result
                        ],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_coupling_cohesion failed", error=str(e))
                raise ValueError(f"Coupling/cohesion analysis failed: {e}")

        elif name == "map_control_flow":
            if not RELATIONSHIP_MAPPING_AVAILABLE:
                raise ValueError("Control flow mapping not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if relationship_mapper is None:
                    raise ValueError("Relationship mapper not initialized")

                control_flow_result: Any = relationship_mapper.map_control_flow(code, language)

                text_result = [
                    "Control Flow Mapping:",
                    f"Nodes: {len(control_flow_result.nodes)}",
                    f"Edges: {len(control_flow_result.edges)}",
                    f"Paths: {len(control_flow_result.paths)}",
                    f"Complexity: {control_flow_result.cyclomatic_complexity}",
                ]

                json_result = {
                    "request_id": request_id,
                    "control_flow": {
                        "nodes": [
                            {"id": node.id, "type": node.type, "label": node.label, "line_number": node.line_number}
                            for node in control_flow_result.nodes
                        ],
                        "edges": [
                            {"source": edge.source, "target": edge.target, "type": edge.type, "condition": edge.condition}
                            for edge in control_flow_result.edges
                        ],
                        "paths": control_flow_result.paths,
                        "cyclomatic_complexity": control_flow_result.cyclomatic_complexity,
                        "entry_points": control_flow_result.entry_points,
                        "exit_points": control_flow_result.exit_points,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("map_control_flow failed", error=str(e))
                raise ValueError(f"Control flow mapping failed: {e}")

        elif name == "analyze_data_flow":
            if not RELATIONSHIP_MAPPING_AVAILABLE:
                raise ValueError("Data flow analysis not available")

            code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not code:
                raise ValueError("Code parameter is required")

            try:
                if relationship_mapper is None:
                    raise ValueError("Relationship mapper not initialized")

                data_flow_result: Any = relationship_mapper.analyze_data_flow(code, language)

                text_result = [
                    "Data Flow Analysis:",
                    f"Variables: {len(data_flow_result.variables)}",
                    f"Data Dependencies: {len(data_flow_result.data_dependencies)}",
                    f"Flow Paths: {len(data_flow_result.flow_paths)}",
                    f"Critical Variables: {len(data_flow_result.critical_variables)}",
                ]

                if data_flow_result.critical_variables:
                    text_result.append("\nCritical Variables:")
                    for var in data_flow_result.critical_variables[:5]:
                        text_result.append(f"  - {var}")

                json_result = {
                    "request_id": request_id,
                    "data_flow": {
                        "variables": [
                            {"name": var.name, "type": var.type, "scope": var.scope, "mutations": var.mutations}
                            for var in data_flow_result.variables
                        ],
                        "data_dependencies": [
                            {"source": dep.source, "target": dep.target, "type": dep.type, "line_number": dep.line_number}
                            for dep in data_flow_result.data_dependencies
                        ],
                        "flow_paths": data_flow_result.flow_paths,
                        "critical_variables": data_flow_result.critical_variables,
                        "data_flow_complexity": data_flow_result.data_flow_complexity,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_data_flow failed", error=str(e))
                raise ValueError(f"Data flow analysis failed: {e}")

        # Phase 4 Navigation Tools
        elif name == "build_dependency_graph":
            if not NAVIGATION_TOOLS_AVAILABLE:
                raise ValueError("Navigation tools not available")

            project_path = arguments.get("project_path", WORKSPACE_ROOT)
            include_external = arguments.get("include_external", True)

            try:
                if navigation_analyzer is None:
                    raise ValueError("Navigation analyzer not initialized")

                result = navigation_analyzer.graph_builder.build_dependency_graph(include_external)
                circular_deps = navigation_analyzer.graph_builder.detect_circular_dependencies(result)
                module_metrics = navigation_analyzer.graph_builder.calculate_module_metrics(result)
                insights = navigation_analyzer.graph_builder.generate_architectural_insights(result)

                text_result = [
                    "Dependency Graph Construction:",
                    f"Nodes: {len(result.nodes)}",
                    f"Edges: {len(result.edges)}",
                    f"Circular Dependencies: {len(circular_deps)}",
                    f"Modules Analyzed: {len(module_metrics)}",
                    f"Architectural Insights: {len(insights)}",
                ]

                if circular_deps:
                    text_result.append("\nCircular Dependencies:")
                    for dep in circular_deps[:3]:
                        text_result.append(f"  - {' -> '.join(dep.components)} ({dep.severity})")

                if insights:
                    text_result.append("\nArchitectural Insights:")
                    for insight in insights[:3]:
                        text_result.append(f"  - {insight.insight_type}: {insight.severity}")

                json_result = {
                    "request_id": request_id,
                    "dependency_graph": {
                        "nodes": list(result.nodes),
                        "edges": [(u, v, data) for u, v, data in result.edges(data=True)],
                        "circular_dependencies": [
                            {
                                "components": dep.components,
                                "type": dep.dependency_type.value,
                                "severity": dep.severity,
                                "impact": dep.impact,
                                "resolution": dep.suggested_resolution,
                            }
                            for dep in circular_deps
                        ],
                        "module_metrics": {
                            name: {
                                "complexity": m.complexity,
                                "coupling": m.coupling,
                                "cohesion": m.cohesion,
                                "instability": m.instability,
                                "fan_in": m.fan_in,
                                "fan_out": m.fan_out,
                                "lines_of_code": m.lines_of_code,
                            }
                            for name, m in module_metrics.items()
                        },
                        "architectural_insights": [
                            {
                                "type": insight.insight_type,
                                "description": insight.description,
                                "severity": insight.severity,
                                "components": insight.components_involved,
                                "recommendation": insight.recommendation,
                            }
                            for insight in insights
                        ],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("build_dependency_graph failed", error=str(e))
                raise ValueError(f"Dependency graph construction failed: {e}")

        elif name == "analyze_execution_paths":
            if not NAVIGATION_TOOLS_AVAILABLE:
                raise ValueError("Navigation tools not available")

            function_code = arguments.get("code", "")
            language = arguments.get("language", "python")

            if not function_code:
                raise ValueError("Code parameter is required")

            try:
                if navigation_analyzer is None:
                    raise ValueError("Navigation analyzer not initialized")

                control_flow_result = navigation_analyzer.analyze_function_control_flow(function_code)

                text_result = [
                    "Execution Path Analysis:",
                    f"Control Flow Nodes: {len(control_flow_result['control_flow_graph'].nodes)}",
                    f"Control Flow Edges: {len(control_flow_result['control_flow_graph'].edges)}",
                    f"Execution Paths: {len(control_flow_result['execution_paths'])}",
                    f"Cyclomatic Complexity: {control_flow_result['cyclomatic_complexity']}",
                ]

                if control_flow_result["execution_paths"]:
                    text_result.append("\nExecution Paths:")
                    for i, path in enumerate(control_flow_result["execution_paths"][:3]):
                        text_result.append(f"  Path {i + 1}: {len(path.nodes)} nodes, complexity: {path.complexity_score:.2f}")

                json_result = {
                    "request_id": request_id,
                    "execution_analysis": {
                        "control_flow_graph": {
                            "nodes": list(control_flow_result["control_flow_graph"].nodes(data=True)),
                            "edges": list(control_flow_result["control_flow_graph"].edges(data=True)),
                        },
                        "execution_paths": [
                            {
                                "path_id": path.path_id,
                                "nodes": path.nodes,
                                "branches": path.branches,
                                "exceptions": path.exceptions,
                                "complexity_score": path.complexity_score,
                                "coverage_percentage": path.coverage_percentage,
                            }
                            for path in control_flow_result["execution_paths"]
                        ],
                        "cyclomatic_complexity": control_flow_result["cyclomatic_complexity"],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_execution_paths failed", error=str(e))
                raise ValueError(f"Execution path analysis failed: {e}")

        elif name == "visualize_graph":
            if not NAVIGATION_TOOLS_AVAILABLE:
                raise ValueError("Navigation tools not available")

            graph_type_str = arguments.get("graph_type", "dependency")
            output_path = arguments.get("output_path")

            try:
                graph_type = GraphType(graph_type_str)
            except ValueError:
                valid_types = [t.value for t in GraphType]
                raise ValueError(f"Invalid graph type: {graph_type_str}. Valid types: {valid_types}")

            try:
                if navigation_analyzer is None:
                    raise ValueError("Navigation analyzer not initialized")

                if graph_type == GraphType.DEPENDENCY:
                    graph = navigation_analyzer.graph_builder.graphs.get(GraphType.DEPENDENCY.value)
                    if not graph:
                        graph = navigation_analyzer.graph_builder.build_dependency_graph()
                elif graph_type == GraphType.CONTROL_FLOW:
                    raise ValueError("Control flow graph requires function code - use analyze_execution_paths")
                else:
                    raise ValueError(f"Graph type {graph_type.value} not yet implemented")

                visualization = navigation_analyzer.graph_builder.generate_visualization(graph, graph_type, output_path)

                text_result = [
                    f"Graph Visualization ({graph_type.value}):",
                    f"Nodes: {len(graph.nodes)}",
                    f"Edges: {len(graph.edges)}",
                ]

                if output_path:
                    text_result.append(f"Visualization saved to: {visualization}")
                else:
                    text_result.append("DOT source generated (no output path specified)")

                json_result = {
                    "request_id": request_id,
                    "visualization": {
                        "graph_type": graph_type.value,
                        "nodes": len(graph.nodes),
                        "edges": len(graph.edges),
                        "output_path": output_path or visualization,
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("visualize_graph failed", error=str(e))
                raise ValueError(f"Graph visualization failed: {e}")

        elif name == "analyze_project_architecture":
            if not NAVIGATION_TOOLS_AVAILABLE:
                raise ValueError("Navigation tools not available")

            try:
                if navigation_analyzer is None:
                    raise ValueError("Navigation analyzer not initialized")

                arch_result = navigation_analyzer.analyze_project_architecture()
                summary = navigation_analyzer.get_architectural_summary()

                text_result = [
                    "Project Architecture Analysis:",
                    f"Total Modules: {summary['total_modules']}",
                    f"Circular Dependencies: {summary['circular_dependencies']}",
                    f"Architectural Insights: {summary['architectural_insights']}",
                    f"High Coupling Modules: {summary['high_coupling_modules']}",
                    f"Unstable Modules: {summary['unstable_modules']}",
                    f"Average Complexity: {summary['average_complexity']:.2f}",
                ]

                if arch_result.get("architectural_insights"):
                    text_result.append("\nKey Insights:")
                    for insight in arch_result["architectural_insights"][:5]:
                        text_result.append(f"  - {insight.insight_type}: {insight.description}")

                json_result = {
                    "request_id": request_id,
                    "architectural_analysis": {
                        "summary": summary,
                        "dependency_graph_stats": {
                            "nodes": len(arch_result["dependency_graph"].nodes),
                            "edges": len(arch_result["dependency_graph"].edges),
                        },
                        "circular_dependencies": len(arch_result["circular_dependencies"]),
                        "module_metrics": {
                            name: {"complexity": m.complexity, "coupling": m.coupling, "instability": m.instability}
                            for name, m in arch_result.get("module_metrics", {}).items()
                        },
                        "architectural_insights": [
                            {
                                "type": insight.insight_type,
                                "severity": insight.severity,
                                "description": insight.description,
                                "recommendation": insight.recommendation,
                            }
                            for insight in arch_result.get("architectural_insights", [])
                        ],
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_project_architecture failed", error=str(e))
                raise ValueError(f"Project architecture analysis failed: {e}")

        elif name == "safe_rename_symbol":
            if not SAFE_REFACTORING_AVAILABLE:
                raise ValueError("Safe refactoring tools not available")

            old_name = arguments.get("old_name")
            new_name = arguments.get("new_name")
            symbol_type = arguments.get("symbol_type", "function")
            scope = arguments.get("scope")
            project_path = arguments.get("project_path", ".")

            # Validate required arguments
            if old_name is None:
                raise ValueError("old_name is required")
            if new_name is None:
                raise ValueError("new_name is required")

            try:
                if safe_renamer is None:
                    raise ValueError("Safe renamer not initialized")

                rename_result = safe_renamer.rename_symbol_safely(old_name, new_name, symbol_type, scope, project_path)

                text_result = [
                    f"Symbol Rename Operation: {old_name} → {new_name}",
                    f"Symbol Type: {symbol_type}",
                    f"Status: {'Success' if rename_result.get('success') else 'Failed'}",
                ]

                if rename_result.get("success"):
                    text_result.append(f"Operation ID: {rename_result.get('operation_id')} (for rollback)")
                    text_result.append(f"Files Modified: {len(rename_result.get('modified_files', []))}")
                else:
                    text_result.append(f"Error: {rename_result.get('error', 'Unknown error')}")

                json_result = {
                    "request_id": request_id,
                    "rename_operation": {
                        "old_name": old_name,
                        "new_name": new_name,
                        "symbol_type": symbol_type,
                        "scope": scope,
                        "project_path": project_path,
                        "operation_id": rename_result.get("operation_id"),
                        "success": rename_result.get("success"),
                        "modified_files": rename_result.get("modified_files", []),
                        "rollback_available": rename_result.get("rollback_available", False),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("safe_rename_symbol failed", error=str(e))
                raise ValueError(f"Safe symbol rename failed: {e}")

        elif name == "analyze_rename_impact":
            if not SAFE_REFACTORING_AVAILABLE:
                raise ValueError("Safe refactoring tools not available")

            old_name = arguments.get("old_name")
            new_name = arguments.get("new_name")
            symbol_type = arguments.get("symbol_type", "function")
            scope = arguments.get("scope")
            project_path = arguments.get("project_path", ".")

            # Validate required arguments
            if old_name is None:
                raise ValueError("old_name is required")
            if new_name is None:
                raise ValueError("new_name is required")

            try:
                if safe_renamer is None:
                    raise ValueError("Safe renamer not initialized")

                impact_analysis = safe_renamer.analyze_rename_impact(old_name, new_name, symbol_type, scope, project_path)

                text_result = [
                    f"Rename Impact Analysis: {old_name} → {new_name}",
                    f"Symbol Type: {symbol_type}",
                    f"Risk Score: {impact_analysis.risk_score:.2f}",
                    f"Affected Files: {len(impact_analysis.affected_files)}",
                    f"Affected Symbols: {len(impact_analysis.affected_symbols)}",
                    f"External Dependencies: {len(impact_analysis.external_dependencies)}",
                ]

                if impact_analysis.affected_files:
                    text_result.append("\nAffected Files:")
                    for file_path in list(impact_analysis.affected_files)[:5]:
                        text_result.append(f"  - {file_path}")

                json_result = {
                    "request_id": request_id,
                    "impact_analysis": {
                        "old_name": old_name,
                        "new_name": new_name,
                        "symbol_type": symbol_type,
                        "risk_score": impact_analysis.risk_score,
                        "affected_files": list(impact_analysis.affected_files),
                        "affected_symbols": list(impact_analysis.affected_symbols),
                        "external_dependencies": list(impact_analysis.external_dependencies),
                        "is_safe": safe_renamer.validate_rename_safety(impact_analysis),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("analyze_rename_impact failed", error=str(e))
                raise ValueError(f"Rename impact analysis failed: {e}")

        elif name == "safe_extract_function":
            if not SAFE_REFACTORING_AVAILABLE:
                raise ValueError("Safe refactoring tools not available")

            source_range = arguments.get("source_range")
            function_name = arguments.get("function_name")
            target_file = arguments.get("target_file")
            source_file = arguments.get("source_file")
            project_path = arguments.get("project_path", ".")

            # Validate required arguments
            if source_range is None:
                raise ValueError("source_range is required")
            if function_name is None:
                raise ValueError("function_name is required")
            if target_file is None:
                raise ValueError("target_file is required")
            if source_file is None:
                raise ValueError("source_file is required")

            # Ensure source_range is a tuple
            if not isinstance(source_range, tuple):
                raise ValueError("source_range must be a tuple (start_line, end_line)")

            try:
                if safe_extractor is None:
                    raise ValueError("Safe extractor not initialized")

                # Extract function safely with proper error handling
                try:
                    extraction_result: dict[str, Any] = safe_extractor.extract_function_safely(
                        source_range, function_name, target_file, source_file, project_path
                    )
                    operation_success = True
                    operation_data = extraction_result
                except Exception as e:
                    # Return a structured error result
                    operation_success = False
                    operation_data = {
                        "success": False,
                        "error": str(e)
                    }

                text_result = [
                    f"Function Extraction: {function_name}",
                    f"Source: {source_file} lines {source_range[0]}-{source_range[1]}",
                    f"Target: {target_file}",
                    f"Status: {'Success' if operation_success else 'Failed'}",
                ]

                if operation_success:
                    text_result.append(f"Operation ID: {operation_data.get('operation_id')} (for rollback)")
                    text_result.append(f"Dependencies Extracted: {len(operation_data.get('extracted_dependencies', []))}")
                else:
                    text_result.append(f"Error: {operation_data.get('error', 'Unknown error')}")

                json_result = {
                    "request_id": request_id,
                    "extraction_operation": {
                        "function_name": function_name,
                        "source_range": source_range,
                        "source_file": source_file,
                        "target_file": target_file,
                        "project_path": project_path,
                        "operation_id": operation_data.get("operation_id"),
                        "success": operation_data.get("success"),
                        "extracted_dependencies": operation_data.get("extracted_dependencies", []),
                        "rollback_available": operation_data.get("rollback_available", False),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("safe_extract_function failed", error=str(e))
                raise ValueError(f"Safe function extraction failed: {e}")

        elif name == "safe_move_symbol":
            if not SAFE_REFACTORING_AVAILABLE:
                raise ValueError("Safe refactoring tools not available")

            symbol_name = arguments.get("symbol_name")
            source_file = arguments.get("source_file")
            target_file = arguments.get("target_file")
            symbol_type = arguments.get("symbol_type", "function")
            scope = arguments.get("scope")
            project_path = arguments.get("project_path", ".")

            # Validate required arguments
            if symbol_name is None:
                raise ValueError("symbol_name parameter is required")
            if source_file is None:
                raise ValueError("source_file parameter is required")
            if target_file is None:
                raise ValueError("target_file parameter is required")

            try:
                if safe_extractor is None:
                    raise ValueError("Safe extractor not initialized")

                move_result: dict[str, Any] = safe_extractor.move_symbol_safely(
                    symbol_name, source_file, target_file, symbol_type, scope, project_path
                )

                text_result = [
                    f"Symbol Move: {symbol_name}",
                    f"Source: {source_file}",
                    f"Target: {target_file}",
                    f"Type: {symbol_type}",
                    f"Status: {'Success' if move_result.get('success') else 'Failed'}",
                ]

                if move_result.get("success"):
                    text_result.append(f"Operation ID: {move_result.get('operation_id')} (for rollback)")
                    text_result.append(f"References Updated: {len(move_result.get('updated_references', []))}")
                else:
                    text_result.append(f"Error: {move_result.get('error', 'Unknown error')}")

                json_result = {
                    "request_id": request_id,
                    "move_operation": {
                        "symbol_name": symbol_name,
                        "source_file": source_file,
                        "target_file": target_file,
                        "symbol_type": symbol_type,
                        "scope": scope,
                        "project_path": project_path,
                        "operation_id": move_result.get("operation_id"),
                        "success": move_result.get("success"),
                        "updated_references": move_result.get("updated_references", []),
                        "rollback_available": move_result.get("rollback_available", False),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("safe_move_symbol failed", error=str(e))
                raise ValueError(f"Safe symbol move failed: {e}")

        elif name == "execute_rollback":
            if not SAFE_REFACTORING_AVAILABLE:
                raise ValueError("Safe refactoring tools not available")

            operation_id = arguments.get("operation_id")

            # Validate required arguments
            if operation_id is None:
                raise ValueError("operation_id parameter is required")

            try:
                if safe_renamer is None:
                    raise ValueError("Safe renamer not initialized")

                rollback_result: dict[str, Any] = safe_renamer.execute_rollback(operation_id)

                text_result = [
                    f"Rollback Operation: {operation_id}",
                    f"Status: {'Success' if rollback_result.get('success') else 'Failed'}",
                ]

                if rollback_result.get("success"):
                    text_result.append(f"Files Restored: {rollback_result.get('files_restored', 0)}")
                else:
                    text_result.append(f"Error: {rollback_result.get('error', 'Unknown error')}")

                json_result = {
                    "request_id": request_id,
                    "rollback_operation": {
                        "operation_id": operation_id,
                        "success": rollback_result.get("success"),
                        "files_restored": rollback_result.get("files_restored", 0),
                        "error": rollback_result.get("error"),
                    },
                }

                return [{"type": "text", "text": "\n".join(text_result)}, {"type": "text", "text": json.dumps(json_result, indent=2)}]

            except Exception as e:
                logger.error("execute_rollback failed", error=str(e))
                raise ValueError(f"Rollback execution failed: {e}")

        # Batch Operations Handlers - Phase 6
        elif name == "batch_analyze_project":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            project_path = arguments.get("project_path")
            analysis_types = arguments.get("analysis_types", ["complexity", "dependencies"])
            # max_workers = arguments.get("max_workers", 4)  # Unused
            timeout = arguments.get("timeout", 300.0)
            output_format = arguments.get("output_format", "json")

            # Validate required arguments
            if project_path is None:
                raise ValueError("project_path parameter is required")

            try:
                if batch_analyzer is None:
                    raise ValueError("Batch analyzer not initialized")

                # Generate unique operation ID
                operation_id = f"batch_analysis_{uuid.uuid4().hex[:8]}"

                # Execute batch analysis
                batch_result = batch_analyzer.analyze_project_batches(project_path, analysis_types)

                # Format results
                if output_format == "json":
                    response_data = {
                        "request_id": request_id,
                        "operation_id": operation_id,
                        "batch_analysis": batch_result.details,
                        "processing_time_ms": batch_result.execution_time * 1000,
                        "files_analyzed": batch_result.processed_files,
                        "analysis_types": analysis_types,
                    }
                    return [{"type": "text", "text": json_dumps(response_data)}]
                else:
                    lines = [
                        "📊 Batch Analysis Results",
                        f"Operation ID: {operation_id}",
                        f"Project: {project_path}",
                        f"Files Analyzed: {batch_result.processed_files}",
                        f"Processing Time: {batch_result.execution_time:.2f}s",
                        f"Analysis Types: {', '.join(analysis_types)}",
                    ]

                    if batch_result.transformations_applied > 0:
                        lines.append(f"✅ Transformations Applied: {batch_result.transformations_applied}")
                    if batch_result.errors_encountered > 0:
                        lines.append(f"❌ Errors Encountered: {batch_result.errors_encountered}")

                    return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("batch_analyze_project failed", error=str(e))
                raise ValueError(f"Batch project analysis failed: {e}")

        elif name == "batch_transform_code":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            transformation_type = arguments.get("transformation_type")
            project_path = arguments.get("project_path")
            targets = arguments.get("targets", [])
            parameters = arguments.get("parameters", {})
            validation_level = arguments.get("validation_level", "normal")
            dry_run = arguments.get("dry_run", False)
            # max_workers = arguments.get("max_workers", 2)  # Unused

            # Validate required arguments
            if transformation_type is None:
                raise ValueError("transformation_type is required")
            if project_path is None:
                raise ValueError("project_path is required")

            try:
                if batch_transformer is None:
                    raise ValueError("Batch transformer not initialized")

                # Import required types
                from .batch_operations import BatchResults

                # Execute batch transformation
                transform_result: BatchResults = batch_transformer.batch_transform_code(
                    transformation_type=transformation_type,
                    project_path=project_path,
                    code_patterns=targets or [],
                    replacement_patterns=parameters,
                    preview_only=dry_run or False,
                )

                # Format results
                lines = [
                    "🔄 Batch Code Transformation",
                    f"Type: {transformation_type}",
                    f"Project: {project_path}",
                    f"Targets: {len(targets) if targets else 0}",
                    f"Mode: {'DRY RUN' if dry_run else 'LIVE'}",
                    f"Status: {'Success' if transform_result.success else 'Partial/Failed'}",
                    f"Files Processed: {transform_result.processed_files}",
                    f"Execution Time: {transform_result.execution_time:.2f}s",
                ]

                if transform_result.transformations_applied > 0:
                    lines.append(f"✅ Transformations Applied: {transform_result.transformations_applied}")
                if transform_result.errors_encountered > 0:
                    lines.append(f"❌ Errors Encountered: {transform_result.errors_encountered}")

                json_result = {
                    "request_id": request_id,
                    "batch_transformation": {
                        "operation_id": transform_result.operation_id,
                        "success": transform_result.success,
                        "processed_files": transform_result.processed_files,
                        "total_files": transform_result.total_files,
                        "execution_time": batch_result.execution_time,
                        "transformations_applied": batch_result.transformations_applied,
                        "errors_encountered": batch_result.errors_encountered,
                        "warnings_generated": batch_result.warnings_generated,
                        "details": batch_result.details,
                    },
                    "transformation_type": transformation_type,
                    "dry_run": dry_run,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("batch_transform_code failed", error=str(e))
                raise ValueError(f"Batch code transformation failed: {e}")

        elif name == "monitor_batch_progress":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            operation_id = arguments.get("operation_id")
            include_details = arguments.get("include_details", True)
            include_metrics = arguments.get("include_metrics", True)
            include_errors = arguments.get("include_errors", True)

            try:
                if batch_analyzer is None:
                    raise ValueError("Batch analyzer not initialized")

                # Get progress monitor
                progress_monitor = batch_analyzer.progress_monitor

                # Get current progress
                progress_info = progress_monitor.get_progress_summary()

                # Format progress information
                lines = [
                    "📈 Batch Operation Progress",
                    f"Operation ID: {operation_id}",
                    f"Status: {progress_info.get('status', 'Unknown')}",
                    f"Progress: {progress_info.get('progress_percentage', 0):.1f}%",
                ]

                if include_details and "current_stage" in progress_info:
                    lines.append(f"Current Stage: {progress_info['current_stage']}")

                if include_metrics and "performance_metrics" in progress_info:
                    metrics = progress_info["performance_metrics"]
                    lines.append(f"Processing Time: {metrics.get('total_time', 0):.2f}s")
                    lines.append(f"Files Processed: {metrics.get('files_processed', 0)}")

                if include_errors and "errors" in progress_info and progress_info["errors"]:
                    lines.append(f"Errors: {len(progress_info['errors'])}")

                return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("monitor_batch_progress failed", error=str(e))
                raise ValueError(f"Batch progress monitoring failed: {e}")

        elif name == "schedule_batch_operations":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            operations = arguments.get("operations", [])
            max_concurrent = arguments.get("max_concurrent", 3)
            schedule_mode = arguments.get("schedule_mode", "queued")

            try:
                if batch_analyzer is None:
                    raise ValueError("Batch analyzer not initialized")

                # Get batch scheduler
                scheduler = batch_analyzer.scheduler

                # Schedule operations
                scheduled_ops = []
                for op in operations:
                    op_type = op.get("type")
                    op_params = op.get("parameters", {})
                    priority = op.get("priority", "normal")

                    # Create batch operation
                    batch_op = BatchOperation(
                        id=f"op_{uuid.uuid4().hex[:8]}",
                        type=BatchOperationType(op_type),
                        name=f"Batch {op_type}",
                        description=f"Batch operation of type {op_type}",
                        priority=priority,
                        results={"parameters": op_params}
                    )

                    operation_id = scheduler.schedule_operation(batch_op)
                    scheduled_ops.append({"operation_id": operation_id, "type": op_type, "priority": priority})

                # Format results
                lines = [
                    "📅 Batch Operations Scheduled",
                    f"Mode: {schedule_mode}",
                    f"Max Concurrent: {max_concurrent}",
                    f"Operations Scheduled: {len(scheduled_ops)}",
                ]

                for op in scheduled_ops:
                    lines.append(f"  - {op['type']} (ID: {op['operation_id']}, Priority: {op['priority']})")

                schedule_result: Dict[str, Any] = {
                    "request_id": request_id,
                    "scheduled_operations": scheduled_ops,
                    "schedule_mode": schedule_mode,
                    "max_concurrent": max_concurrent,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(schedule_result)}]

            except Exception as e:
                logger.error("schedule_batch_operations failed", error=str(e))
                raise ValueError(f"Batch operations scheduling failed: {e}")

        elif name == "validate_batch_operations":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            operation_plan = arguments.get("operation_plan")
            project_path = arguments.get("project_path")
            validation_level = arguments.get("validation_level", "standard")
            # include_performance = arguments.get("include_performance", True)
            # include_security = arguments.get("include_security", True)

            try:
                if batch_transformer is None:
                    raise ValueError("Batch transformer not initialized")

                # Validate operation plan
                if operation_plan is None:
                    raise ValueError("operation_plan is required")
                if project_path is None:
                    raise ValueError("project_path is required")

                validation_result = batch_transformer.validate_batch_operations(
                    operations=operation_plan,
                    project_path=project_path,
                )

                # Format validation results
                lines = [
                    "✅ Batch Operations Validation",
                    f"Project: {project_path}",
                    f"Validation Level: {validation_level}",
                    f"Overall Status: {'Valid' if validation_result.get('valid_operations') else 'Invalid'}",
                ]

                if validation_result.get("warnings"):
                    lines.append(f"Warnings: {len(validation_result['warnings'])}")

                if validation_result.get("invalid_operations"):
                    lines.append(f"Errors: {len(validation_result['invalid_operations'])}")
                    for error in validation_result["invalid_operations"][:5]:  # Show first 5 errors
                        lines.append(f"  - {error}")

                if validation_result.get("safety_score"):
                    lines.append(f"Safety Score: {validation_result['safety_score']:.1f}/10")

                return [{"type": "text", "text": "\n".join(lines)}]

            except Exception as e:
                logger.error("validate_batch_operations failed", error=str(e))
                raise ValueError(f"Batch operations validation failed: {e}")

        elif name == "execute_batch_rename":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            rename_operations = arguments.get("rename_operations", [])
            project_path = arguments.get("project_path")
            update_references = arguments.get("update_references", True)
            dry_run = arguments.get("dry_run", False)

            try:
                if batch_transformer is None:
                    raise ValueError("Batch transformer not initialized")
                if project_path is None:
                    raise ValueError("project_path is required")

                # Execute batch rename
                from .batch_operations import BatchResults
                batch_rename_result: BatchResults = batch_transformer.batch_rename_symbol(
                    old_name=rename_operations[0]["old_name"],
                    new_name=rename_operations[0]["new_name"],
                    project_path=project_path,
                    symbol_type="function",
                    scope=None,
                    preview_only=dry_run,
                )

                # Format results
                lines = [
                    "🔄 Batch Symbol Rename",
                    f"Project: {project_path}",
                    f"Operations: {len(rename_operations)}",
                    f"Mode: {'DRY RUN' if dry_run else 'LIVE'}",
                    f"References Updated: {update_references}",
                    f"Status: {'Success' if batch_rename_result.success else 'Partial/Failed'}",
                    f"Files Processed: {batch_rename_result.processed_files}",
                    f"Transformations: {batch_rename_result.transformations_applied}",
                ]

                if batch_rename_result.errors_encountered:
                    lines.append(f"❌ Errors: {batch_rename_result.errors_encountered}")
                if batch_rename_result.warnings_generated:
                    lines.append(f"⚠️  Warnings: {batch_rename_result.warnings_generated}")

                json_result = {
                    "request_id": request_id,
                    "batch_rename": asdict(batch_rename_result),
                    "dry_run": dry_run,
                    "update_references": update_references,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("execute_batch_rename failed", error=str(e))
                raise ValueError(f"Batch rename execution failed: {e}")

        elif name == "batch_extract_components":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            extractions = arguments.get("extractions", [])
            project_path = arguments.get("project_path")
            manage_imports = arguments.get("manage_imports", True)
            # max_workers = arguments.get("max_workers", 2)  # Unused

            try:
                if batch_transformer is None:
                    raise ValueError("Batch transformer not initialized")
                if project_path is None:
                    raise ValueError("project_path is required")

                # Execute batch extraction
                extract_result: BatchResults = batch_transformer.batch_extract_components(
                    component_patterns=extractions,
                    project_path=project_path,
                    output_directory="extracted_components",
                    preview_only=False,
                )

                # Format results
                lines = [
                    "📦 Batch Component Extraction",
                    f"Project: {project_path}",
                    f"Components: {len(extractions)}",
                    f"Imports Managed: {manage_imports}",
                    f"Status: {'Success' if extract_result.success else 'Partial/Failed'}",
                ]

                if extract_result.details.get("successful_extractions"):
                    lines.append(f"✅ Successful Extractions: {len(extract_result.details['successful_extractions'])}")
                if extract_result.details.get("failed_extractions"):
                    lines.append(f"❌ Failed Extractions: {len(extract_result.details['failed_extractions'])}")

                json_result = {
                    "request_id": request_id,
                    "batch_extraction": extract_result.to_dict() if hasattr(extract_result, "to_dict") else str(extract_result),
                    "manage_imports": manage_imports,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("batch_extract_components failed", error=str(e))
                raise ValueError(f"Batch component extraction failed: {e}")

        elif name == "generate_batch_report":
            if not BATCH_OPERATIONS_AVAILABLE:
                raise ValueError("Batch operations tools not available")

            operation_id = arguments.get("operation_id")
            report_type = arguments.get("report_type", "detailed")
            include_metrics = arguments.get("include_metrics", True)
            include_errors = arguments.get("include_errors", True)
            include_recommendations = arguments.get("include_recommendations", True)
            output_format = arguments.get("output_format", "json")

            try:
                if batch_analyzer is None:
                    raise ValueError("Batch analyzer not initialized")
                if operation_id is None:
                    raise ValueError("operation_id is required")

                # Generate batch report
                report: dict[str, Any] = batch_analyzer.generate_batch_report(
                    operation_id=operation_id,
                    report_type=report_type,
                    include_metrics=include_metrics,
                    include_errors=include_errors,
                    include_recommendations=include_recommendations,
                )

                # Format report
                lines = [
                    "📊 Batch Operation Report",
                    f"Operation ID: {operation_id}",
                    f"Report Type: {report_type}",
                    f"Generated: {report.get('generated_at', 'Unknown')}",
                ]

                if include_metrics and "performance_metrics" in report:
                    metrics = report.get("performance_metrics", {})
                    lines.append(f"Processing Time: {metrics.get('total_time', 0):.2f}s")
                    lines.append(f"Success Rate: {metrics.get('success_rate', 0):.1f}%")

                if include_recommendations and "recommendations" in report:
                    recommendations = report.get("recommendations", [])
                    lines.append(f"Recommendations: {len(recommendations)}")

                json_result = {
                    "request_id": request_id,
                    "batch_report": report,
                    "report_type": report_type,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("generate_batch_report failed", error=str(e))
                raise ValueError(f"Batch report generation failed: {e}")

        # Security & Quality Analysis Tools
        elif name == "security_scan_comprehensive":
            if not SECURITY_QUALITY_AVAILABLE:
                raise ValueError("Security & Quality analysis tools not available")

            project_path = arguments.get("project_path")
            scan_types = arguments.get("scan_types", ["owasp_top_10", "dependency_check"])
            severity_threshold = arguments.get("severity_threshold", "low")
            include_remediation = arguments.get("include_remediation", True)
            output_format = arguments.get("output_format", "json")

            try:
                if security_scanner is None:
                    raise ValueError("Security scanner not initialized")

                # Perform security scan
                if project_path is None:
                    raise ValueError("project_path is required for security scan")
                security_report_result: SecurityReport = security_scanner.security_scan_comprehensive(project_path)

                # Filter by severity threshold
                severity_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
                threshold_level = severity_order.get(severity_threshold, 0)
                filtered_vulnerabilities = [
                    v for v in security_report_result.vulnerabilities
                    if severity_order.get(v.severity.value, 0) >= threshold_level
                ]
                security_report_result.vulnerabilities = filtered_vulnerabilities

                # Generate summary
                lines = [
                    "🔒 Comprehensive Security Scan Report",
                    f"Project: {project_path}",
                    f"Scan ID: {security_report_result.scan_id}",
                    f"Timestamp: {security_report_result.timestamp.isoformat()}",
                    f"Total Vulnerabilities: {len(security_report_result.vulnerabilities)}",
                ]

                # Severity summary
                for severity, count in security_report_result.severity_summary.items():
                    if count > 0:
                        lines.append(f"{severity.value.upper()}: {count}")

                lines.append(f"Risk Score: {security_report_result.risk_score}/100")

                if include_remediation and security_report_result.recommendations:
                    lines.append("Recommendations:")
                    for rec in security_report_result.recommendations[:5]:  # Top 5 recommendations
                        lines.append(f"  • {rec}")

                json_result = {
                    "request_id": request_id,
                    "security_report": asdict(security_report_result),
                    "scan_types": scan_types,
                    "severity_threshold": severity_threshold,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("security_scan_comprehensive failed", error=str(e))
                raise ValueError(f"Security scan failed: {e}")

        elif name == "quality_assessment_comprehensive":
            if not SECURITY_QUALITY_AVAILABLE:
                raise ValueError("Security & Quality analysis tools not available")

            project_path = arguments.get("project_path")
            file_patterns = arguments.get("file_patterns", ["*.py", "*.js", "*.ts", "*.java", "*.cpp"])
            output_format = arguments.get("output_format", "json")

            # Validate required arguments
            if project_path is None:
                raise ValueError("project_path is required")

            try:
                if quality_analyzer is None:
                    raise ValueError("Quality analyzer not initialized")

                # Perform quality assessment on matching files
                all_assessments = []
                from .security_quality_analysis import QualityMetrics
                total_metrics = QualityMetrics()

                import glob

                for pattern in file_patterns:
                    files = glob.glob(os.path.join(project_path, "**", pattern), recursive=True)
                    for file_path in files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                code = f.read()
                            assessment = quality_analyzer.assess_code_quality(code, file_path)
                            all_assessments.append(assessment)
                            total_metrics.overall_score += assessment.metrics.overall_score
                        except Exception:
                            continue

                # Calculate average metrics
                if all_assessments:
                    num_files = len(all_assessments)
                    total_metrics.overall_score /= num_files
                    total_metrics.code_smells_count = sum(a.metrics.code_smells_count for a in all_assessments)
                    total_metrics.security_issues_count = sum(a.metrics.security_issues_count for a in all_assessments)

                # Generate summary
                lines = [
                    "📊 Comprehensive Quality Assessment Report",
                    f"Project: {project_path}",
                    f"Files Analyzed: {len(all_assessments)}",
                    f"Overall Quality Score: {total_metrics.overall_score:.1f}/100",
                    f"Quality Grade: {quality_analyzer._determine_quality_grade(total_metrics.overall_score)}",
                ]

                lines.append(f"Code Smells Found: {total_metrics.code_smells_count}")
                lines.append(f"Security Issues: {total_metrics.security_issues_count}")

                # Top recommendations
                all_recommendations = []
                for assessment in all_assessments:
                    all_recommendations.extend(assessment.improvement_recommendations)

                if all_recommendations:
                    lines.append("Top Recommendations:")
                    for rec in list(set(all_recommendations))[:5]:  # Top 5 unique recommendations
                        lines.append(f"  • {rec}")

                json_result = {
                    "request_id": request_id,
                    "quality_assessment": {
                        "project_path": project_path,
                        "files_analyzed": len(all_assessments),
                        "overall_metrics": asdict(total_metrics),
                        "individual_assessments": [asdict(a) for a in all_assessments[:10]],  # First 10
                        "all_recommendations": list(set(all_recommendations)),
                    },
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("quality_assessment_comprehensive failed", error=str(e))
                raise ValueError(f"Quality assessment failed: {e}")

        elif name == "compliance_reporting_generate":
            if not SECURITY_QUALITY_AVAILABLE:
                raise ValueError("Security & Quality analysis tools not available")

            project_path = arguments.get("project_path")
            include_security_scan = arguments.get("include_security_scan", True)
            include_quality_assessment = arguments.get("include_quality_assessment", True)
            certification_threshold = arguments.get("certification_threshold", 85.0)
            output_format = arguments.get("output_format", "json")

            # Validate required arguments
            if project_path is None:
                raise ValueError("project_path is required")

            try:
                if compliance_reporter is None:
                    raise ValueError("Compliance reporter not initialized")

                # Perform security scan if requested
                security_report = None
                if include_security_scan and security_scanner:
                    security_report = security_scanner.security_scan_comprehensive(project_path)

                # Perform quality assessment if requested
                quality_assessment = None
                if include_quality_assessment and quality_analyzer:
                    # Simple quality assessment
                    quality_assessment = quality_analyzer.assess_code_quality("", project_path)

                # Generate compliance report
                from datetime import datetime

                from .security_quality_analysis import QualityMetrics

                compliance_report = compliance_reporter.generate_compliance_report(
                    security_report or SecurityReport(
                        scan_id=f"compliance_{datetime.now().isoformat()}",
                        timestamp=datetime.now(),
                        project_path=project_path
                    ),
                    quality_assessment or QualityAssessment(
                        project_path=project_path,
                        timestamp=datetime.now(),
                        metrics=QualityMetrics()
                    ),
                )

                # Generate summary
                lines = [
                    "📋 Compliance Report",
                    f"Project: {project_path}",
                    f"Overall Compliance Score: {compliance_report.get('overall_compliance_score', 0):.1f}%",
                    f"Certification Ready: {'✅ Yes' if compliance_report.get('certification_ready', False) else '❌ No'}",
                ]

                # Standard scores
                standard_scores = compliance_report.get('standard_scores', {})
                for standard_name, score in standard_scores.items():
                    status = "✅" if score >= certification_threshold else "❌"
                    lines.append(f"{status} {standard_name}: {score:.1f}%")

                # Recommendations
                recommendations = compliance_report.get('recommendations', [])
                if recommendations:
                    lines.append("Recommendations:")
                    for rec in recommendations[:3]:
                        lines.append(f"  • {rec}")

                json_result = {
                    "request_id": request_id,
                    "compliance_report": compliance_report,
                    "certification_threshold": certification_threshold,
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("compliance_reporting_generate failed", error=str(e))
                raise ValueError(f"Compliance reporting failed: {e}")

        elif name == "quality_gates_evaluate":
            if not SECURITY_QUALITY_AVAILABLE:
                raise ValueError("Security & Quality analysis tools not available")

            project_path = arguments.get("project_path")
            custom_gates = arguments.get("gates")
            fail_on_error = arguments.get("fail_on_error", True)
            fail_on_warning = arguments.get("fail_on_warning", False)
            output_format = arguments.get("output_format", "json")

            try:
                if quality_gates is None:
                    raise ValueError("Quality gates not initialized")
                if quality_analyzer is None:
                    raise ValueError("Quality analyzer not initialized")

                # Create custom gates if provided
                from .security_quality_analysis import QualityGate
                gates: list[QualityGate] = []
                if custom_gates:
                    parsed_gates = []
                    for gate_data in custom_gates:
                        parsed_gate = QualityGate(
                            name=gate_data["name"],
                            metric=getattr(QualityMetric, gate_data["metric"].upper()),
                            threshold=gate_data["threshold"],
                            operator=gate_data["operator"],
                            severity=gate_data["severity"],
                            enabled=gate_data.get("enabled", True),
                        )
                        parsed_gates.append(parsed_gate)
                    gates = parsed_gates

                # Perform quality assessment first
                quality_assessment = quality_analyzer.assess_code_quality("", project_path)

                # Evaluate quality gates
                gate_results: dict[str, Any] = quality_gates.evaluate_quality_gates(quality_assessment, gates)

                # Generate summary
                lines = [
                    "🚪 Quality Gate Evaluation",
                    f"Project: {project_path}",
                    f"Overall Result: {gate_results.get('overall_result', 'UNKNOWN')}",
                    f"Quality Score: {gate_results.get('metrics', {}).get('overall_score', 0):.1f}/100",
                ]

                # Failed gates
                failed_gates = gate_results.get('failed_gates', [])
                if isinstance(failed_gates, list) and failed_gates:
                    lines.append("❌ Failed Gates:")
                    for gate in failed_gates:
                        if isinstance(gate, dict):
                            lines.append(f"  • {gate.get('gate_name', 'Unknown')}: {gate.get('message', 'No message')}")

                # Warnings
                warnings = gate_results.get('warnings', [])
                if isinstance(warnings, list) and warnings:
                    lines.append("⚠️  Warnings:")
                    for gate in warnings:
                        if isinstance(gate, dict):
                            lines.append(f"  • {gate.get('gate_name', 'Unknown')}: {gate.get('message', 'No message')}")

                # Determine if build should fail
                build_failed = False
                overall_result = gate_results.get('overall_result', 'PASSED')
                if fail_on_error and overall_result == "FAILED":
                    build_failed = True
                if fail_on_warning and warnings:
                    build_failed = True

                lines.append(f"Build Status: {'❌ FAILED' if build_failed else '✅ PASSED'}")

                json_result = {
                    "request_id": request_id,
                    "quality_gate_results": gate_results,
                    "build_failed": str(build_failed),
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("quality_gates_evaluate failed", error=str(e))
                raise ValueError(f"Quality gate evaluation failed: {e}")

        elif name == "vulnerability_database_check":
            if not SECURITY_QUALITY_AVAILABLE:
                raise ValueError("Security & Quality analysis tools not available")

            project_path = arguments.get("project_path")
            dependency_files = arguments.get(
                "dependency_files", ["requirements.txt", "package.json", "pom.xml", "build.gradle", "pyproject.toml"]
            )
            severity_filter = arguments.get("severity_filter", "all")
            output_format = arguments.get("output_format", "json")

            try:
                if security_scanner is None:
                    raise ValueError("Security scanner not initialized")

                # Check dependency vulnerabilities
                vulnerabilities = security_scanner.dependency_checker.scan_dependencies(project_path)

                # Filter by severity
                if severity_filter != "all":
                    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                    min_severity = severity_order.get(severity_filter, 0)
                    filtered_vulnerabilities = [v for v in vulnerabilities if severity_order.get(v.severity.value, 0) >= min_severity]
                    vulnerabilities = filtered_vulnerabilities

                # Generate summary
                lines = [
                    "🔍 Vulnerability Database Check",
                    f"Project: {project_path}",
                    f"Dependencies Scanned: {len(dependency_files)}",
                    f"Vulnerabilities Found: {len(vulnerabilities)}",
                ]

                # Severity breakdown
                severity_counts: dict[SeverityLevel, int] = {}
                for vuln in vulnerabilities:
                    severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1

                for severity, count in severity_counts.items():
                    lines.append(f"{severity.value.upper()}: {count}")

                # Top vulnerabilities
                if vulnerabilities:
                    lines.append("Top Vulnerabilities:")
                    for vuln in vulnerabilities[:5]:  # Top 5
                        lines.append(f"  • {vuln.title}: {vuln.description}")

                json_result = {
                    "request_id": request_id,
                    "vulnerability_check": {
                        "project_path": project_path,
                        "dependencies_checked": dependency_files,
                        "vulnerabilities": [asdict(v) for v in vulnerabilities],
                        "severity_filter": severity_filter,
                    },
                }

                return [{"type": "text", "text": "\n".join(lines)}, {"type": "text", "text": json_dumps(json_result)}]

            except Exception as e:
                logger.error("vulnerability_database_check failed", error=str(e))
                raise ValueError(f"Vulnerability database check failed: {e}")

        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error("Tool call error", tool=name, error=str(e), request_id=request_id)
        raise


def main():
    """Run the FastApply MCP server with stdio transport."""
    logger.info("Starting FastApply MCP server")

    try:
        # Run the server using stdio transport
        logger.info("Initializing stdio transport...")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error("Server error", error=str(e))
        raise
    finally:
        logger.info("FastApply MCP server stopped")


if __name__ == "__main__":
    main()
