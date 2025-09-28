"""
Ripgrep Integration Layer

Ultra-fast pattern discovery using ripgrep for large codebases.
Provides foundation for multi-stage search pipeline (ripgrep → AST → semantic).
"""

import asyncio
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SearchType(Enum):
    """Types of search operations supported by ripgrep."""

    PATTERN = "pattern"
    LITERAL = "literal"
    WORD = "word"
    REGEX = "regex"


class OutputFormat(Enum):
    """Output formats for ripgrep results."""

    JSON = "json"
    TEXT = "text"
    PATHS_ONLY = "paths"


@dataclass
class SearchOptions:
    """Configuration options for ripgrep searches."""

    search_type: SearchType = SearchType.PATTERN
    case_sensitive: bool = True
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    max_results: Optional[int] = None
    context_lines: int = 0
    file_types: Optional[List[str]] = None
    max_filesize: Optional[str] = None  # e.g., "10M", "1G"
    max_depth: Optional[int] = None
    follow_symlinks: bool = False
    output_format: OutputFormat = OutputFormat.JSON

    def __post_init__(self):
        if self.include_patterns is None:
            self.include_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []


@dataclass
class SearchResult:
    """Individual search result from ripgrep."""

    file_path: str
    line_number: int
    line_text: str
    byte_offset: int
    context_before: Optional[List[str]] = None
    context_after: Optional[List[str]] = None
    submatches: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []
        if self.submatches is None:
            self.submatches = []


@dataclass
class SearchResults:
    """Collection of search results with metadata."""

    results: List[SearchResult]
    total_matches: int
    files_searched: int
    search_time: float
    pattern: str
    search_options: SearchOptions

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "results": [asdict(result) for result in self.results],
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "search_time": self.search_time,
            "pattern": self.pattern,
            "search_options": asdict(self.search_options),
        }


@dataclass
class FileMetrics:
    """Metrics about file analysis."""

    file_path: str
    size_bytes: int
    line_count: int
    language: str
    encoding: str
    last_modified: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolCandidate:
    """Potential symbol found by ripgrep pattern matching."""

    symbol_name: str
    symbol_type: str  # function, class, variable, etc.
    file_path: str
    line_number: int
    confidence_score: float  # 0.0 to 1.0
    context_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RipgrepIntegration:
    """
    High-performance ripgrep integration for pattern discovery.

    Provides ultra-fast search capabilities with intelligent result processing
    and integration with the multi-stage search pipeline.
    """

    def __init__(self, ripgrep_path: str = "rg"):
        """
        Initialize ripgrep integration.

        Args:
            ripgrep_path: Path to ripgrep executable (defaults to 'rg')
        """
        self.ripgrep_path = ripgrep_path
        self._verify_ripgrep_installation()

        # Common programming language file extensions
        self.language_patterns = {
            "python": [".py"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".hpp", ".cc", ".cxx"],
            "go": [".go"],
            "rust": [".rs"],
            "ruby": [".rb"],
            "php": [".php"],
            "swift": [".swift"],
            "kotlin": [".kt"],
            "scala": [".scala"],
        }

        # Common symbol patterns for different languages
        self.symbol_patterns = {
            "function": {
                "python": [r"def\s+(\w+)\s*\(", r"async\s+def\s+(\w+)\s*\("],
                "javascript": [r"function\s+(\w+)\s*\(", r"const\s+(\w+)\s*=\s*(\([^)]*\)\s*=>|\w+\s*\([^)]*\)\s*\{)"],
                "typescript": [r"function\s+(\w+)\s*\(", r"const\s+(\w+)\s*=\s*(\([^)]*\)\s*=>|\w+\s*\([^)]*\)\s*\{)"],
                "java": [r"(public|private|protected)?\s*(static)?\s+\w+\s+(\w+)\s*\("],
                "c": [r"\w+\s+(\w+)\s*\("],
                "cpp": [r"\w+\s+(\w+)\s*\("],
                "go": [r"func\s+(\w+)\s*\("],
                "rust": [r"fn\s+(\w+)\s*\("],
            },
            "class": {
                "python": [r"class\s+(\w+)"],
                "javascript": [r"class\s+(\w+)"],
                "typescript": [r"class\s+(\w+)"],
                "java": [r"(public|private|protected)?\s*class\s+(\w+)"],
                "c": [r"struct\s+(\w+)"],
                "cpp": [r"class\s+(\w+)"],
                "go": [r"type\s+(\w+)\s+struct"],
                "rust": [r"struct\s+(\w+)"],
            },
            "variable": {
                "python": [r"(\w+)\s*=", r"(\w+)\s*:"],
                "javascript": [r"(const|let|var)\s+(\w+)\s*="],
                "typescript": [r"(const|let|var)\s+(\w+)\s*=", r"(\w+):\s*\w+"],
                "java": [r"\w+\s+(\w+)\s*="],
                "c": [r"\w+\s+(\w+)\s*="],
                "cpp": [r"\w+\s+(\w+)\s*="],
                "go": [r"var\s+(\w+)\s*", r"(\w+)\s*:="],
                "rust": [r"let\s+(mut\s+)?(\w+)\s*="],
            },
        }

    def _verify_ripgrep_installation(self) -> None:
        """Verify that ripgrep is installed and accessible."""
        try:
            result = subprocess.run([self.ripgrep_path, "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError(f"ripgrep not found at {self.ripgrep_path}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"ripgrep installation verification failed: {e}")

    def search_files(self, pattern: str, path: str, options: Optional[SearchOptions] = None) -> SearchResults:
        """
        Search for files using ripgrep with the specified pattern.

        Args:
            pattern: Pattern to search for
            path: Directory or file path to search in
            options: Search configuration options

        Returns:
            SearchResults containing matched files and metadata
        """
        if options is None:
            options = SearchOptions()

        start_time = time.time()

        # Build ripgrep command
        cmd = self._build_ripgrep_command(pattern, path, options)

        # Execute search
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for searches
            )

            if result.returncode != 0:
                if result.returncode == 1:  # No matches found
                    return SearchResults([], 0, 0, time.time() - start_time, pattern, options)
                else:
                    raise RuntimeError(f"ripgrep search failed: {result.stderr}")

            # Parse results
            search_results = self._parse_ripgrep_output(result.stdout, options.output_format)

            # Estimate files searched (ripgrep doesn't provide this directly)
            files_searched = self._estimate_files_searched(path, options)

            search_time = time.time() - start_time

            return SearchResults(
                results=search_results,
                total_matches=len(search_results),
                files_searched=files_searched,
                search_time=search_time,
                pattern=pattern,
                search_options=options,
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError("ripgrep search timed out after 30 seconds")

    def search_code_patterns(self, pattern: str, language: str, path: str, options: Optional[SearchOptions] = None) -> SearchResults:
        """
        Search for code patterns specific to a programming language.

        Args:
            pattern: Code pattern to search for
            language: Programming language (python, javascript, etc.)
            path: Directory or file path to search in
            options: Search configuration options

        Returns:
            SearchResults with language-specific optimizations
        """
        if options is None:
            options = SearchOptions()

        # Add language-specific file extensions
        if language in self.language_patterns:
            lang_extensions = self.language_patterns[language]
            file_type_patterns = [f"*{ext}" for ext in lang_extensions]

            # Merge with existing include patterns
            if options.include_patterns:
                options.include_patterns.extend(file_type_patterns)
            else:
                options.include_patterns = file_type_patterns

        # Apply language-specific optimizations
        if language in self.symbol_patterns:
            # For symbol searches, use more targeted patterns
            for symbol_type, patterns in self.symbol_patterns[language].items():
                if any(pattern in p for p in patterns):
                    # This looks like a symbol search, enhance the pattern
                    enhanced_pattern = self._enhance_symbol_pattern(pattern, language, symbol_type)
                    return self.search_files(enhanced_pattern, path, options)

        return self.search_files(pattern, path, options)

    def find_symbol_candidates(
        self, symbol_name: str, path: str, symbol_type: Optional[str] = None, language: Optional[str] = None
    ) -> List[SymbolCandidate]:
        """
        Find potential symbol candidates using ripgrep pattern matching.

        Args:
            symbol_name: Name of the symbol to find
            path: Directory or file path to search in
            symbol_type: Type of symbol (function, class, variable)
            language: Programming language for context

        Returns:
            List of SymbolCandidate objects with confidence scores
        """
        candidates = []

        # Determine search patterns based on symbol type and language
        search_patterns = self._get_symbol_search_patterns(symbol_name, symbol_type, language)

        for pattern, candidate_type, conf_modifier in search_patterns:
            options = SearchOptions(
                search_type=SearchType.REGEX,
                context_lines=2,  # Get context for confidence scoring
                output_format=OutputFormat.JSON,
            )

            try:
                results = self.search_files(pattern, path, options)

                for result in results.results:
                    # Calculate confidence score based on various factors
                    confidence = self._calculate_symbol_confidence(result, symbol_name, candidate_type, language) * conf_modifier

                    candidate = SymbolCandidate(
                        symbol_name=symbol_name,
                        symbol_type=candidate_type,
                        file_path=result.file_path,
                        line_number=result.line_number,
                        confidence_score=min(confidence, 1.0),
                        context_text=result.line_text,
                    )

                    candidates.append(candidate)

            except Exception:
                # Log error but continue with other patterns
                continue

        # Sort by confidence score and remove duplicates
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        unique_candidates = self._remove_duplicate_candidates(candidates)

        return unique_candidates

    def analyze_file_metrics(self, file_path: str) -> FileMetrics:
        """
        Analyze file metrics using ripgrep and system tools.

        Args:
            file_path: Path to the file to analyze

        Returns:
            FileMetrics object with file statistics
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Basic file information
        stat = path_obj.stat()

        # Count lines using ripgrep (very fast)
        line_count = int(
            subprocess.run([self.ripgrep_path, "--count-matches", ".", file_path], capture_output=True, text=True).stdout.strip() or "0"
        )

        # Detect language based on file extension
        language = self._detect_file_language(file_path)

        # Detect encoding (basic detection)
        encoding = self._detect_file_encoding(file_path)

        return FileMetrics(
            file_path=file_path,
            size_bytes=stat.st_size,
            line_count=line_count,
            language=language,
            encoding=encoding,
            last_modified=stat.st_mtime,
        )

    def _build_ripgrep_command(self, pattern: str, path: str, options: SearchOptions) -> List[str]:
        """Build ripgrep command line arguments."""
        cmd = [self.ripgrep_path]

        # Search type options
        if options.search_type == SearchType.LITERAL:
            cmd.append("--fixed-strings")
        elif options.search_type == SearchType.WORD:
            cmd.append("--word-regexp")

        # Case sensitivity
        if not options.case_sensitive:
            cmd.append("--ignore-case")

        # Context lines
        if options.context_lines > 0:
            cmd.extend(["--context", str(options.context_lines)])

        # File type filters
        if options.file_types:
            type_patterns = []
            for file_type in options.file_types:
                if file_type in self.language_patterns:
                    type_patterns.extend(self.language_patterns[file_type])
            if type_patterns:
                cmd.extend(["--type-add", f"custom:*{','.join(type_patterns)}", "--type", "custom"])

        # Include patterns
        if options.include_patterns:
            for pattern in options.include_patterns:
                cmd.extend(["--glob", pattern])

        # Exclude patterns
        if options.exclude_patterns:
            for pattern in options.exclude_patterns:
                cmd.extend(["--glob", f"!{pattern}"])

        # Max file size
        if options.max_filesize:
            cmd.extend(["--max-filesize", options.max_filesize])

        # Max depth
        if options.max_depth:
            cmd.extend(["--max-depth", str(options.max_depth)])

        # Follow symlinks
        if options.follow_symlinks:
            cmd.append("--follow")

        # Output format
        if options.output_format == OutputFormat.JSON:
            cmd.extend(["--json"])

        # Max results
        if options.max_results:
            cmd.extend(["--max-count", str(options.max_results)])

        # Add pattern and path
        cmd.append(pattern)
        cmd.append(path)

        return cmd

    def _parse_ripgrep_output(self, output: str, format_type: OutputFormat) -> List[SearchResult]:
        """Parse ripgrep output into SearchResult objects."""
        results = []

        if format_type == OutputFormat.JSON:
            for line in output.strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        # Only process match entries
                        if data.get("type") == "match":
                            result = SearchResult(
                                file_path=data.get("data", {}).get("path", {}).get("text", ""),
                                line_number=data.get("data", {}).get("line_number", 0),
                                line_text=data.get("data", {}).get("lines", {}).get("text", ""),
                                byte_offset=data.get("data", {}).get("absolute_offset", 0),
                                context_before=[],
                                context_after=[],
                                submatches=data.get("data", {}).get("submatches", []),
                            )
                            results.append(result)
                    except json.JSONDecodeError:
                        continue
        else:
            # Simple text parsing
            for line_num, line in enumerate(output.strip().split("\n"), 1):
                if line.strip():
                    result = SearchResult(
                        file_path="unknown",  # Would need more context for path
                        line_number=line_num,
                        line_text=line,
                        byte_offset=0,
                        context_before=[],
                        context_after=[],
                    )
                    results.append(result)

        return results

    def _estimate_files_searched(self, path: str, options: SearchOptions) -> int:
        """Estimate number of files searched (ripgrep doesn't provide this directly)."""
        try:
            # Use find to count files that would be searched
            cmd = ["find", path, "-type", "f"]

            # Apply similar filtering logic as ripgrep
            if options.include_patterns:
                for pattern in options.include_patterns:
                    cmd.extend(["-name", pattern])

            if options.exclude_patterns:
                for pattern in options.exclude_patterns:
                    cmd.extend(["!", "-name", pattern])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        except Exception:
            pass

        # Fallback estimate
        return 100  # Conservative estimate

    def _enhance_symbol_pattern(self, pattern: str, language: str, symbol_type: str) -> str:
        """Enhance search pattern for better symbol matching."""
        # Add language-specific enhancements
        if language == "python" and symbol_type == "function":
            return f"(def\\s+|async\\s+def\\s+){pattern}\\s*\\("
        elif language in ["javascript", "typescript"] and symbol_type == "function":
            return f"(function\\s+{pattern}\\s*\\(|const\\s+{pattern}\\s*=\\s*\\()"
        elif language == "java" and symbol_type == "class":
            return f"class\\\\s+{pattern}\\\\s*\\{{"

        return pattern

    def _get_symbol_search_patterns(self, symbol_name: str, symbol_type: Optional[str], language: Optional[str]) -> List[tuple]:
        """Get search patterns for symbol detection."""
        patterns = []

        # Escape special regex characters in symbol name
        escaped_name = re.escape(symbol_name)

        if language and language in self.symbol_patterns:
            lang_patterns = self.symbol_patterns[language]

            if symbol_type and symbol_type in lang_patterns:
                # Use specific patterns for this symbol type
                for pattern in lang_patterns[symbol_type]:
                    patterns.append((pattern.replace(r"(\w+)", escaped_name), symbol_type, 1.0))
            else:
                # Try all symbol types for this language
                for sym_type, sym_patterns in lang_patterns.items():
                    for pattern in sym_patterns:
                        patterns.append((pattern.replace(r"(\w+)", escaped_name), sym_type, 0.8))
        else:
            # Generic patterns that work across languages
            generic_patterns = [
                (f"\\b{escaped_name}\\b", "variable", 0.6),
                (f"def\\s+{escaped_name}\\s*\\(", "function", 0.7),
                (f"class\\s+{escaped_name}\\b", "class", 0.7),
                (f"function\\s+{escaped_name}\\s*\\(", "function", 0.7),
            ]
            patterns.extend(generic_patterns)

        return patterns

    def _calculate_symbol_confidence(self, result: SearchResult, symbol_name: str, symbol_type: str, language: Optional[str]) -> float:
        """Calculate confidence score for symbol candidate."""
        confidence = 0.5  # Base confidence

        line_text = result.line_text.lower()

        # Boost confidence for exact matches
        if symbol_name.lower() in line_text:
            confidence += 0.2

        # Boost confidence for symbol-specific patterns
        if symbol_type == "function":
            if any(keyword in line_text for keyword in ["def ", "function ", "fn ", "func "]):
                confidence += 0.2
        elif symbol_type == "class":
            if "class " in line_text:
                confidence += 0.2
        elif symbol_type == "variable":
            if any(op in line_text for op in [" = ", ":=", "const ", "let ", "var "]):
                confidence += 0.2

        # Boost confidence for language-specific patterns
        if language:
            if language == "python" and "def " in line_text:
                confidence += 0.1
            elif language in ["javascript", "typescript"] and "function " in line_text:
                confidence += 0.1
            elif language == "java" and ("class " in line_text or "public " in line_text):
                confidence += 0.1

        return confidence

    def _remove_duplicate_candidates(self, candidates: List[SymbolCandidate]) -> List[SymbolCandidate]:
        """Remove duplicate candidates (same file and line number)."""
        seen = set()
        unique_candidates = []

        for candidate in candidates:
            key = (candidate.file_path, candidate.line_number, candidate.symbol_type)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        return unique_candidates

    def _detect_file_language(self, file_path: str) -> str:
        """Detect programming language based on file extension."""
        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()

        for language, extensions in self.language_patterns.items():
            if suffix in extensions:
                return language

        return "unknown"

    def _detect_file_encoding(self, file_path: str) -> str:
        """Basic file encoding detection."""
        try:
            with open(file_path, "rb") as f:
                # Read first few bytes to check for BOM
                bom = f.read(4)

                if bom.startswith(b"\xef\xbb\xbf"):
                    return "utf-8-sig"
                elif bom.startswith(b"\xff\xfe"):
                    return "utf-16-le"
                elif bom.startswith(b"\xfe\xff"):
                    return "utf-16-be"
                else:
                    # Default to utf-8
                    return "utf-8"
        except Exception:
            return "unknown"

    def batch_search(
        self, patterns: List[str], path: str, options: Optional[SearchOptions] = None, max_concurrent: int = 4
    ) -> Dict[str, SearchResults]:
        """
        Perform multiple searches concurrently.

        Args:
            patterns: List of patterns to search for
            path: Directory or file path to search in
            options: Search configuration options
            max_concurrent: Maximum concurrent searches

        Returns:
            Dictionary mapping patterns to SearchResults
        """

        async def async_search(pattern: str) -> tuple:
            try:
                result = self.search_files(pattern, path, options)
                return (pattern, result)
            except Exception:
                return (pattern, SearchResults([], 0, 0, 0, pattern, options or SearchOptions()))

        # Run searches concurrently with semaphore
        async def run_batch_search():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def search_with_semaphore(pattern: str):
                async with semaphore:
                    return await async_search(pattern)

            tasks = [search_with_semaphore(pattern) for pattern in patterns]
            return await asyncio.gather(*tasks)

        # Run async function from sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(run_batch_search())

        return dict(results)
