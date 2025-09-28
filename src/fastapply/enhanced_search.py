"""
Enhanced Search Infrastructure - Intelligent Search Pipeline

Implements a multi-stage search system that combines ripgrep speed with intelligent
result ranking, filtering, and semantic understanding for ultra-fast code discovery.
"""

import concurrent.futures
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ast_rule_intelligence import LLMAstReasoningEngine
from .ripgrep_integration import RipgrepIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Search strategies for different use cases."""
    EXACT = "exact"           # Exact pattern matching
    FUZZY = "fuzzy"          # Fuzzy pattern matching
    SEMANTIC = "semantic"   # Semantic similarity search
    HYBRID = "hybrid"       # Combination of multiple strategies


class ResultRanking(Enum):
    """Result ranking strategies."""
    RELEVANCE = "relevance"    # By pattern relevance
    FREQUENCY = "frequency"    # By occurrence frequency
    RECENCY = "recency"       # By file modification time
    CONFIDENCE = "confidence"  # By confidence score
    COMBINED = "combined"     # Combined ranking


@dataclass
class SearchContext:
    """Context for search operations."""
    query: str
    path: str = "."
    file_types: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    max_results: int = 100
    strategy: SearchStrategy = SearchStrategy.HYBRID
    ranking: ResultRanking = ResultRanking.COMBINED
    timeout: float = 30.0
    case_sensitive: bool = False
    context_lines: int = 3


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with metadata and ranking."""
    file_path: str
    line_number: int
    line_content: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    match_type: str = "exact"  # exact, fuzzy, semantic
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    frequency_score: float = 0.0
    recency_score: float = 0.0
    combined_score: float = 0.0
    language: str = ""
    symbol_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    total_files_searched: int = 0
    total_matches_found: int = 0
    search_duration: float = 0.0
    ranking_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    strategy_breakdown: Dict[str, float] = field(default_factory=dict)


class SearchCache:
    """Intelligent caching system for search results."""

    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, Tuple[List[EnhancedSearchResult], float]] = {}
        self.max_entries = max_entries
        self.access_times: Dict[str, float] = {}

    def _generate_key(self, context: SearchContext) -> str:
        """Generate cache key from search context."""
        key_data = {
            "query": context.query,
            "path": context.path,
            "file_types": sorted(context.file_types),
            "exclude_patterns": sorted(context.exclude_patterns),
            "include_patterns": sorted(context.include_patterns),
            "strategy": context.strategy.value,
            "case_sensitive": context.case_sensitive,
        }
        return json.dumps(key_data, sort_keys=True)

    def get(self, context: SearchContext) -> Optional[List[EnhancedSearchResult]]:
        """Get cached results."""
        key = self._generate_key(context)
        if key in self.cache:
            results, timestamp = self.cache[key]
            self.access_times[key] = time.time()
            return results
        return None

    def set(self, context: SearchContext, results: List[EnhancedSearchResult]) -> None:
        """Cache search results."""
        key = self._generate_key(context)
        current_time = time.time()

        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.access_times.keys())
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = (results, current_time)
        self.access_times[key] = current_time

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()


class ResultRanker:
    """Intelligent result ranking system."""

    def __init__(self):
        self.language_weights = {
            "python": 1.2,
            "javascript": 1.1,
            "typescript": 1.15,
            "java": 1.0,
            "cpp": 0.9,
            "c": 0.85,
            "go": 1.05,
            "rust": 1.1,
            "php": 0.8,
            "ruby": 0.85
        }

    def calculate_relevance_score(self, result: EnhancedSearchResult, query: str) -> float:
        """Calculate relevance score based on pattern matching."""
        content = result.line_content.lower()
        query_lower = query.lower()

        # Exact match bonus
        if query_lower in content:
            score = 1.0
        else:
            # Fuzzy match score
            score = self._fuzzy_match_score(content, query_lower)

        # Position bonus (matches at beginning of line are more relevant)
        if content.startswith(query_lower):
            score *= 1.2

        # Symbol type bonus
        if result.symbol_type in ["function", "class", "method"]:
            score *= 1.3

        return min(score, 1.0)

    def calculate_frequency_score(self, result: EnhancedSearchResult, all_results: List[EnhancedSearchResult]) -> float:
        """Calculate frequency score based on occurrence patterns."""
        # Count occurrences in same file
        file_matches = [r for r in all_results if r.file_path == result.file_path]

        # Fewer matches per file = higher score (more specific)
        if len(file_matches) == 1:
            return 1.0
        elif len(file_matches) <= 3:
            return 0.8
        elif len(file_matches) <= 10:
            return 0.6
        else:
            return 0.4

    def calculate_recency_score(self, result: EnhancedSearchResult) -> float:
        """Calculate recency score based on file modification time."""
        try:
            file_path = Path(result.file_path)
            if file_path.exists():
                mod_time = file_path.stat().st_mtime
                current_time = time.time()
                age_days = (current_time - mod_time) / (24 * 3600)

                # Recent files get higher scores
                if age_days < 1:
                    return 1.0
                elif age_days < 7:
                    return 0.9
                elif age_days < 30:
                    return 0.8
                elif age_days < 90:
                    return 0.7
                else:
                    return 0.6
        except (OSError, IOError):
            pass

        return 0.5

    def _fuzzy_match_score(self, text: str, pattern: str) -> float:
        """Calculate fuzzy match score."""
        if not pattern:
            return 0.0

        pattern_len = len(pattern)
        text_len = len(text)

        if pattern_len == 0:
            return 0.0
        if text_len == 0:
            return 0.0

        # Simple character-based fuzzy matching
        pattern_chars = set(pattern.lower())
        text_chars = set(text.lower())

        intersection = pattern_chars.intersection(text_chars)
        union = pattern_chars.union(text_chars)

        return len(intersection) / len(union)

    def rank_results(self, results: List[EnhancedSearchResult], query: str,
                    ranking: ResultRanking) -> List[EnhancedSearchResult]:
        """Rank search results using specified strategy."""
        if not results:
            return results

        # Calculate individual scores
        for result in results:
            result.relevance_score = self.calculate_relevance_score(result, query)
            result.frequency_score = self.calculate_frequency_score(result, results)
            result.recency_score = self.calculate_recency_score(result)

            # Language bonus
            lang_weight = self.language_weights.get(result.language, 1.0)
            result.relevance_score *= lang_weight

        # Calculate combined scores
        for result in results:
            if ranking == ResultRanking.COMBINED:
                result.combined_score = (
                    result.relevance_score * 0.4 +
                    result.confidence_score * 0.3 +
                    result.frequency_score * 0.2 +
                    result.recency_score * 0.1
                )
            elif ranking == ResultRanking.RELEVANCE:
                result.combined_score = result.relevance_score
            elif ranking == ResultRanking.CONFIDENCE:
                result.combined_score = result.confidence_score
            elif ranking == ResultRanking.FREQUENCY:
                result.combined_score = result.frequency_score
            elif ranking == ResultRanking.RECENCY:
                result.combined_score = result.recency_score

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results


class EnhancedSearchInfrastructure:
    """Enhanced search infrastructure with intelligent pipeline."""

    def __init__(self, ripgrep_path: str = "rg"):
        self.ripgrep = RipgrepIntegration(ripgrep_path)
        self.ast_intelligence = LLMAstReasoningEngine()
        self.cache = SearchCache()
        self.ranker = ResultRanker()
        self.metrics = SearchMetrics()

        # Language detection patterns
        self.language_patterns = {
            "python": [r"\.py$", r"import\s+", r"from\s+\w+\s+import"],
            "javascript": [r"\.js$", r"\.jsx$", r"const\s+\w+\s*=", r"function\s+\w+"],
            "typescript": [r"\.ts$", r"\.tsx$", r"interface\s+\w+", r"type\s+\w+"],
            "java": [r"\.java$", r"public\s+class", r"private\s+\w+"],
            "cpp": [r"\.(cpp|cc|cxx|h|hpp)$", r"#include\s*<", r"namespace\s+\w+"],
            "go": [r"\.go$", r"package\s+\w+", r"func\s+\w+"],
            "rust": [r"\.rs$", r"fn\s+\w+", r"let\s+mut\s+"],
            "php": [r"\.php$", r"<\?php", r"function\s+\w+"],
            "ruby": [r"\.rb$", r"def\s+\w+", r"module\s+\w+"],
        }

    def search(self, context: SearchContext) -> Tuple[List[EnhancedSearchResult], SearchMetrics]:
        """Execute enhanced search with intelligent pipeline."""
        start_time = time.time()

        # Check cache first
        cached_results = self.cache.get(context)
        if cached_results:
            self.metrics.cache_hits += 1
            return cached_results, self.metrics

        self.metrics.cache_misses += 1

        try:
            # Execute search based on strategy
            if context.strategy == SearchStrategy.EXACT:
                results = self._exact_search(context)
            elif context.strategy == SearchStrategy.FUZZY:
                results = self._fuzzy_search(context)
            elif context.strategy == SearchStrategy.SEMANTIC:
                results = self._semantic_search(context)
            elif context.strategy == SearchStrategy.HYBRID:
                results = self._hybrid_search(context)
            else:
                results = self._exact_search(context)

            # Rank results
            ranking_start = time.time()
            results = self.ranker.rank_results(results, context.query, context.ranking)
            self.metrics.ranking_duration = time.time() - ranking_start

            # Update metrics
            self.metrics.total_matches_found = len(results)
            self.metrics.search_duration = time.time() - start_time

            # Cache results
            self.cache.set(context, results)

            return results, self.metrics

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], self.metrics

    def _exact_search(self, context: SearchContext) -> List[EnhancedSearchResult]:
        """Execute exact pattern search using ripgrep."""
        results = []

        # Build search options for ripgrep
        from .ripgrep_integration import SearchOptions, SearchType

        options = SearchOptions(
            search_type=SearchType.PATTERN,
            case_sensitive=context.case_sensitive,
            include_patterns=context.include_patterns,
            exclude_patterns=context.exclude_patterns,
            max_results=context.max_results,
            context_lines=context.context_lines,
            file_types=[ft.lower() for ft in context.file_types] if context.file_types else None
        )

        # Use ripgrep for fast exact matching
        ripgrep_results = self.ripgrep.search_files(
            pattern=context.query,
            path=context.path or ".",
            options=options
        )

        # Convert to enhanced results
        for rg_result in ripgrep_results.results:
            enhanced_result = EnhancedSearchResult(
                file_path=rg_result.file_path,
                line_number=rg_result.line_number,
                line_content=rg_result.line_text,
                context_before=rg_result.context_before or [],
                context_after=rg_result.context_after or [],
                match_type="exact",
                confidence_score=1.0,
                language=self._detect_language(rg_result.file_path),
                metadata={"submatches": rg_result.submatches}
            )
            results.append(enhanced_result)

        self.metrics.total_files_searched = ripgrep_results.files_searched
        return results

    def _fuzzy_search(self, context: SearchContext) -> List[EnhancedSearchResult]:
        """Execute fuzzy pattern search."""
        results = []

        # Generate fuzzy patterns
        fuzzy_patterns = self._generate_fuzzy_patterns(context.query)

        for pattern in fuzzy_patterns:
            # Create new context with fuzzy pattern
            fuzzy_context = SearchContext(
                query=pattern,
                file_types=context.file_types,
                exclude_patterns=context.exclude_patterns,
                include_patterns=context.include_patterns,
                max_results=context.max_results,
                case_sensitive=context.case_sensitive,
                context_lines=context.context_lines,
                path=context.path
            )
            pattern_results = self._exact_search(fuzzy_context)
            for result in pattern_results:
                result.match_type = "fuzzy"
                result.confidence_score = self._calculate_fuzzy_confidence(
                    result.line_content, context.query
                )
                results.append(result)

        # Remove duplicates and limit results
        results = self._deduplicate_results(results)
        return results[:context.max_results]

    def _semantic_search(self, context: SearchContext) -> List[EnhancedSearchResult]:
        """Execute semantic similarity search."""
        results = []

        # Use AST intelligence for semantic understanding if available
        try:
            # For now, fall back to enhanced exact search for semantic results
            # TODO: Integrate with actual semantic analysis when available
            logger.info("Semantic search using enhanced pattern matching")

            # Generate semantic-like patterns from the query
            semantic_patterns = self._generate_semantic_patterns(context.query)

            for pattern in semantic_patterns:
                pattern_context = SearchContext(
                    query=pattern,
                    file_types=context.file_types,
                    exclude_patterns=context.exclude_patterns,
                    include_patterns=context.include_patterns,
                    max_results=context.max_results // len(semantic_patterns),
                    strategy=SearchStrategy.EXACT,
                    ranking=ResultRanking.RELEVANCE,
                    timeout=context.timeout / len(semantic_patterns),
                    case_sensitive=context.case_sensitive,
                    context_lines=context.context_lines
                )

                pattern_results = self._exact_search(pattern_context)
                for result in pattern_results:
                    result.match_type = "semantic"
                    result.confidence_score *= 0.8  # Lower confidence for semantic inference
                    results.append(result)

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        return results

    def _generate_semantic_patterns(self, query: str) -> List[str]:
        """Generate semantic search patterns from query."""
        patterns = [query]  # Include original query

        query_lower = query.lower()

        # Add conceptual variations based on common programming patterns
        if "function" in query_lower or "def" in query_lower:
            patterns.extend(["def ", "function ", "=> ", "lambda "])

        if "class" in query_lower:
            patterns.extend(["class ", "struct ", "interface "])

        if "import" in query_lower:
            patterns.extend(["import ", "from ", "require ", "#include "])

        if "variable" in query_lower or "var" in query_lower:
            patterns.extend(["var ", "let ", "const ", "="])

        return list(set(patterns))

    def _hybrid_search(self, context: SearchContext) -> List[EnhancedSearchResult]:
        """Execute hybrid search combining multiple strategies."""
        all_results = []

        # Execute different search strategies in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit search tasks
            exact_future = executor.submit(self._exact_search, context)
            fuzzy_future = executor.submit(self._fuzzy_search, context)
            semantic_future = executor.submit(self._semantic_search, context)

            # Collect results
            exact_results = exact_future.result(timeout=context.timeout)
            fuzzy_results = fuzzy_future.result(timeout=context.timeout)
            semantic_results = semantic_future.result(timeout=context.timeout)

            all_results.extend(exact_results)
            all_results.extend(fuzzy_results)
            all_results.extend(semantic_results)

        # Remove duplicates and limit results
        results = self._deduplicate_results(all_results)
        return results[:context.max_results]

    def _generate_fuzzy_patterns(self, query: str) -> List[str]:
        """Generate fuzzy search patterns."""
        patterns = [query]  # Original pattern

        # Add character variations
        if len(query) > 3:
            # Remove one character at each position
            for i in range(len(query)):
                pattern = query[:i] + query[i+1:]
                patterns.append(pattern)

            # Add common typos
            common_subs = {
                'a': ['s', 'e'],
                'e': ['a', 'i'],
                'i': ['e', 'o'],
                'o': ['i', 'u'],
                's': ['z'],
                'z': ['s']
            }

            for old_char, new_chars in common_subs.items():
                if old_char in query:
                    for new_char in new_chars:
                        patterns.append(query.replace(old_char, new_char))

        return list(set(patterns))

    def _calculate_fuzzy_confidence(self, text: str, pattern: str) -> float:
        """Calculate fuzzy match confidence score."""
        text_lower = text.lower()
        pattern_lower = pattern.lower()

        # Direct substring match
        if pattern_lower in text_lower:
            return 0.9

        # Character overlap
        pattern_chars = set(pattern_lower)
        text_chars = set(text_lower)
        overlap = len(pattern_chars.intersection(text_chars))
        overlap_ratio = overlap / len(pattern_chars) if pattern_chars else 0

        return min(overlap_ratio, 0.8)

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path and content."""
        # First check file extension
        for language, patterns in self.language_patterns.items():
            if re.search(patterns[0], file_path, re.IGNORECASE):
                return language

        # Try to detect from content if file exists
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1KB

                for language, patterns in self.language_patterns.items():
                    for pattern in patterns[1:]:  # Skip extension pattern
                        if re.search(pattern, content, re.IGNORECASE):
                            return language
        except (OSError, IOError):
            pass

        return "unknown"

    def _deduplicate_results(self, results: List[EnhancedSearchResult]) -> List[EnhancedSearchResult]:
        """Remove duplicate search results."""
        seen = set()
        unique_results = []

        for result in results:
            key = (result.file_path, result.line_number, result.line_content)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        return unique_results

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "cache_hit_rate": (
                self.metrics.cache_hits /
                (self.metrics.cache_hits + self.metrics.cache_misses)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            ),
            "average_search_time": (
                self.metrics.search_duration / max(self.metrics.cache_misses, 1)
            ),
            "total_searches": self.metrics.cache_hits + self.metrics.cache_misses,
            "strategy_breakdown": self.metrics.strategy_breakdown
        }

    def clear_cache(self) -> None:
        """Clear search cache."""
        self.cache.clear()
        logger.info("Search cache cleared")

    def optimize_for_patterns(self, common_patterns: List[str]) -> None:
        """Optimize search infrastructure for common patterns."""
        # Pre-warm cache with common patterns
        for pattern in common_patterns:
            context = SearchContext(query=pattern, max_results=10)
            self.search(context)

        logger.info(f"Optimized for {len(common_patterns)} common patterns")

    def _detect_optimal_strategy(self, query: str, context: str, language: str) -> SearchStrategy:
        """Detect optimal search strategy based on query and context."""
        query_lower = query.lower()

        # Detect intent from query keywords
        if any(keyword in query_lower for keyword in ["find", "search", "locate"]):
            if any(keyword in query_lower for keyword in ["exact", "precise", "match"]):
                return SearchStrategy.EXACT
            elif any(keyword in query_lower for keyword in ["similar", "like", "related"]):
                return SearchStrategy.FUZZY
            elif any(keyword in query_lower for keyword in ["understand", "analyze", "meaning"]):
                return SearchStrategy.SEMANTIC
        elif any(keyword in query_lower for keyword in ["understand", "analyze", "meaning"]):
            return SearchStrategy.SEMANTIC

        # Use language-specific optimizations
        if language and language.lower() in ["python", "javascript", "typescript"]:
            return SearchStrategy.HYBRID

        # Default to hybrid for comprehensive results
        return SearchStrategy.HYBRID

    def get_context(self, context_id: str) -> SearchContext:
        """Get or create search context for context_id."""
        # For now, create a new context each time
        # In a real implementation, this would maintain context state
        return SearchContext(
            query="",
            max_results=100,
            strategy=SearchStrategy.HYBRID,
            ranking=ResultRanking.COMBINED
        )

    def refine_search_results(self, query: str, previous_results: List[Any],
                              context: SearchContext) -> List[EnhancedSearchResult]:
        """Refine existing search results based on new query."""
        # For now, re-execute search with refined context
        context.query = query
        results, _ = self.search(context)
        return results
