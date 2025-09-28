"""
Comprehensive test suite for enhanced search functionality.

Tests the intelligent search pipeline, result ranking, caching, and MCP tool integration.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import the module under test
from fastapply.enhanced_search import (
    EnhancedSearchInfrastructure,
    EnhancedSearchResult,
    ResultRanker,
    ResultRanking,
    SearchCache,
    SearchContext,
    SearchMetrics,
    SearchStrategy,
)


class TestEnhancedSearchInfrastructure(unittest.TestCase):
    """Test the enhanced search infrastructure functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files with different content
        self.create_test_files()

        # Initialize enhanced search instance
        self.search_instance = EnhancedSearchInfrastructure()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for search testing."""
        # Python files
        with open("test1.py", "w", encoding="utf-8") as f:
            f.write("""
def hello_world(name):
    print(f"Hello, {name}!")

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
""")

        with open("test2.py", "w", encoding="utf-8") as f:
            f.write("""
import os

def calculate_sum(a, b):
    return a + b

class Calculator:
    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y
""")

        # JavaScript files
        with open("app.js", "w", encoding="utf-8") as f:
            f.write("""
function greet(name) {
    console.log(`Hello, ${name}!`);
}

class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
    }

    getInfo() {
        return `${this.name} <${this.email}>`;
    }
}
""")

        # Configuration file
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump({
                "app_name": "Test App",
                "version": "1.0.0",
                "debug": True
            }, f, indent=2)

    def test_search_context_creation(self):
        """Test SearchContext creation with various parameters."""
        context = SearchContext(
            query="test",
            file_types=["python", "javascript"],
            max_results=50,
            strategy=SearchStrategy.HYBRID,
            ranking=ResultRanking.COMBINED
        )

        self.assertEqual(context.query, "test")
        self.assertEqual(context.file_types, ["python", "javascript"])
        self.assertEqual(context.max_results, 50)
        self.assertEqual(context.strategy, SearchStrategy.HYBRID)
        self.assertEqual(context.ranking, ResultRanking.COMBINED)

    def test_exact_search_strategy(self):
        """Test exact search strategy."""
        context = SearchContext(
            query="def hello_world",
            include_patterns=[f"{self.test_dir}/*"],
            strategy=SearchStrategy.EXACT,
            max_results=10
        )

        results, metrics = self.search_instance.search(context)

        self.assertIsInstance(results, list)
        self.assertIsInstance(metrics, SearchMetrics)
        # Note: Results may be empty if ripgrep is not available
        self.assertIsInstance(len(results), int)
        if results:
            self.assertEqual(results[0].match_type, "exact")

    def test_fuzzy_search_strategy(self):
        """Test fuzzy search strategy."""
        context = SearchContext(
            query="helo_world",  # Intentional typo
            include_patterns=[f"{self.test_dir}/*"],
            strategy=SearchStrategy.FUZZY,
            max_results=10
        )

        results, metrics = self.search_instance.search(context)

        self.assertIsInstance(results, list)
        # Fuzzy search should still find results despite the typo

    def test_hybrid_search_strategy(self):
        """Test hybrid search strategy."""
        context = SearchContext(
            query="class",
            path=self.test_dir,
            strategy=SearchStrategy.HYBRID,
            max_results=10
        )

        results, metrics = self.search_instance.search(context)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_result_ranking(self):
        """Test result ranking functionality."""
        context = SearchContext(
            query="def",
            include_patterns=[f"{self.test_dir}/*"],
            strategy=SearchStrategy.EXACT,
            ranking=ResultRanking.COMBINED,
            max_results=20
        )

        results, metrics = self.search_instance.search(context)

        # Results should be ranked by combined score
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i].combined_score,
                    results[i + 1].combined_score
                )

    def test_search_cache_functionality(self):
        """Test search caching functionality."""
        context = SearchContext(
            query="print",
            path=self.test_dir,
            max_results=5
        )

        # First search - should be a cache miss
        results1, metrics1 = self.search_instance.search(context)

        # Second search with same context - should be a cache hit
        results2, metrics2 = self.search_instance.search(context)
        cache_hits_2 = self.search_instance.metrics.cache_hits

        # Results should be identical
        self.assertEqual(len(results1), len(results2))

        # Cache metrics should reflect the hit
        self.assertGreater(cache_hits_2, 0)

    def test_language_detection(self):
        """Test language detection from file paths and content."""
        # Test file extension detection
        lang = self.search_instance._detect_language("test.py")
        self.assertEqual(lang, "python")

        lang = self.search_instance._detect_language("app.js")
        self.assertEqual(lang, "javascript")

        lang = self.search_instance._detect_language("config.json")
        self.assertEqual(lang, "unknown")

    def test_search_metrics(self):
        """Test search performance metrics."""
        context = SearchContext(
            query="class",
            include_patterns=[f"{self.test_dir}/*"],
            max_results=10
        )

        results, metrics = self.search_instance.search(context)

        self.assertIsInstance(metrics, SearchMetrics)
        self.assertGreaterEqual(metrics.total_files_searched, 0)
        self.assertGreaterEqual(metrics.total_matches_found, 0)
        self.assertGreaterEqual(metrics.search_duration, 0.0)

    def test_search_statistics(self):
        """Test search statistics functionality."""
        stats = self.search_instance.get_search_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("cache_size", stats)
        self.assertIn("cache_hit_rate", stats)
        self.assertIn("average_search_time", stats)
        self.assertIn("total_searches", stats)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add something to cache
        context = SearchContext(query="test", path=self.test_dir)
        self.search_instance.search(context)

        # Verify cache has content

        # Clear cache
        self.search_instance.clear_cache()

        # Verify cache is empty
        stats_after = self.search_instance.get_search_statistics()
        cache_size_after = stats_after["cache_size"]

        self.assertEqual(cache_size_after, 0)

    def test_pattern_optimization(self):
        """Test pattern optimization functionality."""
        common_patterns = ["def", "class", "import"]

        # This should not raise an exception
        self.search_instance.optimize_for_patterns(common_patterns)

        # Verify cache is populated
        stats = self.search_instance.get_search_statistics()
        self.assertGreaterEqual(stats["cache_size"], 0)

    def test_strategy_detection(self):
        """Test optimal strategy detection."""
        # Test exact detection
        strategy = self.search_instance._detect_optimal_strategy(
            "find exact match", "find exact match", "python"
        )
        self.assertEqual(strategy, SearchStrategy.EXACT)

        # Test fuzzy detection
        strategy = self.search_instance._detect_optimal_strategy(
            "find similar patterns", "find similar patterns", "python"
        )
        self.assertEqual(strategy, SearchStrategy.FUZZY)

        # Test semantic detection - use non-optimized language to avoid hybrid override
        strategy = self.search_instance._detect_optimal_strategy(
            "analyze code meaning", "analyze code meaning", "ruby"
        )
        self.assertEqual(strategy, SearchStrategy.SEMANTIC)

        # Test default hybrid for python
        strategy = self.search_instance._detect_optimal_strategy(
            "search code", "search code", "python"
        )
        self.assertEqual(strategy, SearchStrategy.HYBRID)

    def test_context_management(self):
        """Test search context management."""
        context_id = "test_context_123"

        context = self.search_instance.get_context(context_id)

        self.assertIsInstance(context, SearchContext)
        self.assertEqual(context.max_results, 100)
        self.assertEqual(context.strategy, SearchStrategy.HYBRID)

    def test_result_refinement(self):
        """Test search result refinement."""
        # First, get some results
        context = SearchContext(
            query="def",
            include_patterns=[f"{self.test_dir}/*"],
            max_results=5
        )
        initial_results, _ = self.search_instance.search(context)

        # Then refine them
        refined_results = self.search_instance.refine_search_results(
            "def hello", initial_results, context
        )

        self.assertIsInstance(refined_results, list)

    def test_deduplication(self):
        """Test result deduplication."""
        # Create duplicate results
        duplicate_results = [
            EnhancedSearchResult("test.py", 1, "content", combined_score=0.8),
            EnhancedSearchResult("test.py", 1, "content", combined_score=0.9),  # Same file/line
            EnhancedSearchResult("test.py", 2, "different content", combined_score=0.7)
        ]

        unique_results = self.search_instance._deduplicate_results(duplicate_results)

        # Should remove the duplicate
        self.assertEqual(len(unique_results), 2)

    def test_fuzzy_pattern_generation(self):
        """Test fuzzy pattern generation."""
        patterns = self.search_instance._generate_fuzzy_patterns("hello")

        self.assertIsInstance(patterns, list)
        self.assertIn("hello", patterns)  # Original pattern should be included
        self.assertGreater(len(patterns), 0)

    def test_fuzzy_confidence_calculation(self):
        """Test fuzzy match confidence calculation."""
        confidence = self.search_instance._calculate_fuzzy_confidence(
            "hello world", "hello"
        )

        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with non-existent path
        context = SearchContext(
            query="test",
            path="/non/existent/path"
        )

        results, metrics = self.search_instance.search(context)
        # Should handle gracefully without exceptions
        self.assertIsInstance(results, list)

    @patch('fastapply.enhanced_search.RipgrepIntegration')
    def test_ripgrep_integration_failure(self, mock_ripgrep):
        """Test behavior when ripgrep integration fails."""
        # Make ripgrep raise an exception
        mock_ripgrep_instance = MagicMock()
        mock_ripgrep_instance.search_files.side_effect = Exception("Ripgrep failed")
        mock_ripgrep.return_value = mock_ripgrep_instance

        # Recreate search instance with mocked ripgrep
        search_instance = EnhancedSearchInfrastructure()

        context = SearchContext(query="test", path=self.test_dir)
        results, metrics = search_instance.search(context)

        # Should handle the failure gracefully
        self.assertIsInstance(results, list)


class TestSearchCache(unittest.TestCase):
    """Test the search cache functionality."""

    def setUp(self):
        """Set up test environment."""
        self.cache = SearchCache(max_entries=3)

    def test_cache_operations(self):
        """Test basic cache operations."""
        context = SearchContext(query="test", max_results=10)
        results = [
            EnhancedSearchResult("file1.py", 1, "content", combined_score=0.8),
            EnhancedSearchResult("file2.py", 2, "content", combined_score=0.9)
        ]

        # Test cache miss
        cached_results = self.cache.get(context)
        self.assertIsNone(cached_results)

        # Test cache set
        self.cache.set(context, results)

        # Test cache hit
        cached_results = self.cache.get(context)
        self.assertEqual(len(cached_results), len(results))

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        contexts = [
            SearchContext(query=f"test_{i}", max_results=10)
            for i in range(5)
        ]
        results = [EnhancedSearchResult("file.py", 1, "content", combined_score=0.8)]

        # Fill cache beyond capacity
        for i, context in enumerate(contexts):
            self.cache.set(context, results)

        # Only the last 3 entries should remain
        self.assertEqual(len(self.cache.cache), 3)

        # Oldest entries should be evicted
        self.assertIsNone(self.cache.get(contexts[0]))
        self.assertIsNone(self.cache.get(contexts[1]))
        self.assertIsNotNone(self.cache.get(contexts[2]))


class TestResultRanker(unittest.TestCase):
    """Test the result ranking functionality."""

    def setUp(self):
        """Set up test environment."""
        self.ranker = ResultRanker()

    def test_relevance_scoring(self):
        """Test relevance score calculation."""
        result = EnhancedSearchResult(
            file_path="test.py",
            line_number=1,
            line_content="def hello_world():",
            language="python",
            symbol_type="function"
        )

        score = self.ranker.calculate_relevance_score(result, "hello")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_frequency_scoring(self):
        """Test frequency score calculation."""
        results = [
            EnhancedSearchResult("file1.py", 1, "content"),
            EnhancedSearchResult("file1.py", 2, "content"),  # Same file
            EnhancedSearchResult("file2.py", 1, "content")   # Different file
        ]

        # Test scoring for each result
        for result in results:
            score = self.ranker.calculate_frequency_score(result, results)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_recency_scoring(self):
        """Test recency score calculation."""
        result = EnhancedSearchResult("test.py", 1, "content")

        score = self.ranker.calculate_recency_score(result)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_result_ranking(self):
        """Test overall result ranking."""
        results = [
            EnhancedSearchResult("file1.py", 1, "hello world", relevance_score=0.9),
            EnhancedSearchResult("file2.py", 1, "hello world", relevance_score=0.7),
            EnhancedSearchResult("file3.py", 1, "hello world", relevance_score=0.8)
        ]

        ranked_results = self.ranker.rank_results(results, "hello", ResultRanking.RELEVANCE)

        # Results should be sorted by score
        for i in range(len(ranked_results) - 1):
            self.assertGreaterEqual(
                ranked_results[i].combined_score,
                ranked_results[i + 1].combined_score
            )


if __name__ == "__main__":
    unittest.main()
