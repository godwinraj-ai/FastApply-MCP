"""
Comprehensive test suite for advanced symbol operations.

Tests symbol finding, scope resolution, pattern matching, and relationship analysis
with coverage for all major functionality.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import the module under test
from fastapply.symbol_operations import (
    AdvancedSymbolOperations,
    ReferenceAnalysis,
    ReferenceInfo,
    ReferenceType,
    ResolvedScope,
    SymbolInfo,
    SymbolScope,
    SymbolSearchContext,
    SymbolType,
)


class TestAdvancedSymbolOperations(unittest.TestCase):
    """Test the advanced symbol operations functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files with different symbol types
        self.create_test_files()

        # Initialize symbol operations instance
        self.symbol_ops = AdvancedSymbolOperations()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for symbol operations testing."""
        # Python file with various symbols
        with open("test_symbols.py", "w", encoding="utf-8") as f:
            f.write('''
import os
import sys
from typing import List, Dict

GLOBAL_CONSTANT = "test_value"

class TestClass:
    """A test class for symbol operations."""

    def __init__(self, value: int):
        self.value = value
        self._private_var = "private"

    def public_method(self, param: str) -> str:
        return f"Hello {param}"

    def _private_method(self) -> None:
        """Private method for testing."""
        pass

    @property
    def computed_property(self) -> int:
        return self.value * 2

def standalone_function(name: str) -> str:
    return f"Function result: {name}"

async def async_function(data: List[str]) -> Dict[str, str]:
    return {"result": "async"}

def helper_function():
    pass
''')

        # JavaScript file
        with open("app.js", "w", encoding="utf-8") as f:
            f.write("""
const CONFIG = {
    apiUrl: "https://api.example.com",
    timeout: 5000
};

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.users = [];
    }

    async getUsers() {
        const response = await fetch(this.apiUrl + '/users');
        return response.json();
    }

    findUserById(id) {
        return this.users.find(user => user.id === id);
    }
}

function formatDate(date) {
    return date.toISOString();
}

let globalVariable = "test";
""")

        # TypeScript file
        with open("types.ts", "w", encoding="utf-8") as f:
            f.write("""
interface User {
    id: number;
    name: string;
    email: string;
}

type UserRole = 'admin' | 'user' | 'guest';

class AuthService {
    private token: string;

    constructor(token: string) {
        this.token = token;
    }

    public authenticate(user: User): boolean {
        return user.role === 'admin';
    }

    private validateToken(): boolean {
        return this.token.length > 0;
    }
}

const API_ENDPOINT = 'https://api.example.com';
""")

    def test_symbol_info_creation(self):
        """Test SymbolInfo dataclass creation and attributes."""
        symbol = SymbolInfo(
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=10,
            scope=SymbolScope.GLOBAL,
            documentation="Test function",
            signature="test_function(param: str) -> str",
            parameters=["param"],
            return_type="str",
            confidence_score=0.95,
            language="python",
        )

        self.assertEqual(symbol.name, "test_function")
        self.assertEqual(symbol.symbol_type, SymbolType.FUNCTION)
        self.assertEqual(symbol.file_path, "test.py")
        self.assertEqual(symbol.line_number, 10)
        self.assertEqual(symbol.scope, SymbolScope.GLOBAL)
        self.assertEqual(symbol.documentation, "Test function")
        self.assertEqual(symbol.signature, "test_function(param: str) -> str")
        self.assertEqual(symbol.parameters, ["param"])
        self.assertEqual(symbol.return_type, "str")
        self.assertEqual(symbol.confidence_score, 0.95)
        self.assertEqual(symbol.language, "python")

    def test_symbol_search_context_creation(self):
        """Test SymbolSearchContext creation with various parameters."""
        context = SymbolSearchContext(
            symbol_name="TestClass",
            symbol_type=SymbolType.CLASS,
            scope="global",
            file_path="test.py",
            language="python",
            include_imports=True,
            include_definitions=True,
            max_results=50,
            case_sensitive=True,
            use_semantic_search=True,
        )

        self.assertEqual(context.symbol_name, "TestClass")
        self.assertEqual(context.symbol_type, SymbolType.CLASS)
        self.assertEqual(context.scope, "global")
        self.assertEqual(context.file_path, "test.py")
        self.assertEqual(context.language, "python")
        self.assertTrue(context.include_imports)
        self.assertTrue(context.include_definitions)
        self.assertEqual(context.max_results, 50)
        self.assertTrue(context.case_sensitive)
        self.assertTrue(context.use_semantic_search)

    def test_find_symbol_basic(self):
        """Test basic symbol finding functionality."""
        # Test finding a class
        symbol = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)

        self.assertIsInstance(symbol, SymbolInfo)
        self.assertEqual(symbol.name, "TestClass")
        self.assertEqual(symbol.symbol_type, SymbolType.CLASS)
        self.assertIn("test_symbols.py", symbol.file_path)

    def test_find_symbol_with_scope(self):
        """Test symbol finding with scope constraints."""
        # Test finding a method within class scope
        symbol = self.symbol_ops.find_symbol("public_method", SymbolType.METHOD, "TestClass")

        self.assertIsInstance(symbol, SymbolInfo)
        self.assertEqual(symbol.name, "public_method")
        self.assertIn("test_symbols.py", symbol.file_path)

    def test_find_symbols_by_pattern(self):
        """Test finding symbols using regex patterns."""
        # Test pattern matching for functions
        symbols = self.symbol_ops.find_symbols_by_pattern("def.*function", "python")

        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)

        # All found symbols should be functions (including async functions)
        for symbol in symbols:
            self.assertIn(symbol.symbol_type, [SymbolType.FUNCTION, SymbolType.ASYNC_FUNCTION])

    def test_resolve_symbol_scope(self):
        """Test symbol scope resolution."""
        context = "def test_function():\n    value = GLOBAL_CONSTANT"

        resolved_scope = self.symbol_ops.resolve_symbol_scope("GLOBAL_CONSTANT", context)

        self.assertIsInstance(resolved_scope, ResolvedScope)
        self.assertEqual(resolved_scope.symbol_name, "GLOBAL_CONSTANT")
        self.assertIn("test_symbols.py", resolved_scope.resolved_path)
        self.assertEqual(resolved_scope.scope_level, SymbolScope.GLOBAL)

    def test_get_symbol_references(self):
        """Test finding symbol references."""
        # First, find a symbol
        symbol = self.symbol_ops.find_symbol("GLOBAL_CONSTANT", SymbolType.CONSTANT)

        if symbol.file_path:  # Only test if symbol was found
            references = self.symbol_ops.get_symbol_references(symbol)

            self.assertIsInstance(references, list)
            # References should not include the definition itself
            for ref in references:
                self.assertNotEqual(ref.line_number, symbol.line_number)

    def test_analyze_symbol_relationships(self):
        """Test symbol relationship analysis."""
        # Find a symbol to analyze
        symbol = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)

        if symbol.file_path:  # Only test if symbol was found
            relationships = self.symbol_ops.analyze_symbol_relationships(symbol)

            self.assertIsInstance(relationships, dict)
            self.assertIn("dependencies", relationships)
            self.assertIn("dependents", relationships)
            self.assertIn("related", relationships)

            # All relationship values should be lists
            for key, value in relationships.items():
                self.assertIsInstance(value, list)

    def test_multi_language_support(self):
        """Test symbol operations across different languages."""
        # Test JavaScript
        js_symbols = self.symbol_ops.find_symbols_by_pattern("class", "javascript")
        self.assertGreater(len(js_symbols), 0)

        # Test TypeScript
        ts_symbols = self.symbol_ops.find_symbols_by_pattern("interface", "typescript")
        self.assertGreater(len(ts_symbols), 0)

        # Test Python
        py_symbols = self.symbol_ops.find_symbols_by_pattern("def", "python")
        self.assertGreater(len(py_symbols), 0)

    def test_symbol_type_detection(self):
        """Test automatic symbol type detection."""
        # Test function detection
        symbol = self.symbol_ops.find_symbol("standalone_function", SymbolType.FUNCTION)
        self.assertEqual(symbol.symbol_type, SymbolType.FUNCTION)

        # Test class detection
        symbol = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)
        self.assertEqual(symbol.symbol_type, SymbolType.CLASS)

        # Test constant detection
        symbol = self.symbol_ops.find_symbol("GLOBAL_CONSTANT", SymbolType.CONSTANT)
        self.assertIn(symbol.symbol_type, [SymbolType.CONSTANT, SymbolType.VARIABLE])

    def test_async_function_detection(self):
        """Test async function detection."""
        symbol = self.symbol_ops.find_symbol("async_function", SymbolType.ASYNC_FUNCTION)
        self.assertIn(symbol.symbol_type, [SymbolType.ASYNC_FUNCTION, SymbolType.FUNCTION])

    def test_private_method_detection(self):
        """Test private method detection."""
        symbol = self.symbol_ops.find_symbol("_private_method", SymbolType.METHOD)
        if symbol.file_path:  # Only check if found
            self.assertTrue(symbol.is_private)

    def test_cache_functionality(self):
        """Test symbol caching functionality."""
        # First search - should populate cache
        symbol1 = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)
        cache_size_before = len(self.symbol_ops._symbol_cache)

        # Second search - should use cache
        symbol2 = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)
        cache_size_after = len(self.symbol_ops._symbol_cache)

        # Results should be identical
        self.assertEqual(symbol1.name, symbol2.name)
        self.assertEqual(symbol1.file_path, symbol2.file_path)

        # Cache should have been populated
        self.assertGreaterEqual(cache_size_after, cache_size_before)

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        metrics = self.symbol_ops.get_performance_metrics()

        self.assertIsInstance(metrics, dict)
        self.assertIn("cache_size", metrics)
        self.assertIn("scope_cache_size", metrics)
        self.assertIn("cache_hit_rate", metrics)
        self.assertIn("average_search_time", metrics)
        self.assertIn("supported_languages", metrics)

        # Check supported languages
        supported_langs = metrics["supported_languages"]
        self.assertIn("python", supported_langs)
        self.assertIn("javascript", supported_langs)
        self.assertIn("typescript", supported_langs)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Populate cache
        self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)
        self.symbol_ops.resolve_symbol_scope("GLOBAL_CONSTANT", "test context")

        # Verify cache has content
        self.assertGreater(len(self.symbol_ops._symbol_cache), 0)
        self.assertGreater(len(self.symbol_ops._scope_cache), 0)

        # Clear cache
        self.symbol_ops.clear_cache()

        # Verify cache is empty
        self.assertEqual(len(self.symbol_ops._symbol_cache), 0)
        self.assertEqual(len(self.symbol_ops._scope_cache), 0)

    def test_symbol_with_no_matches(self):
        """Test handling of symbols with no matches."""
        symbol = self.symbol_ops.find_symbol("nonexistent_symbol", SymbolType.FUNCTION)

        self.assertIsInstance(symbol, SymbolInfo)
        self.assertEqual(symbol.name, "nonexistent_symbol")
        self.assertEqual(symbol.confidence_score, 0.0)
        self.assertEqual(symbol.file_path, "")

    def test_case_sensitive_search(self):
        """Test case sensitive vs insensitive search."""
        # Case sensitive search
        context_sensitive = SymbolSearchContext(symbol_name="TestClass", case_sensitive=True)

        # Case insensitive search
        context_insensitive = SymbolSearchContext(symbol_name="testclass", case_sensitive=False)

        # Both should find the same symbol due to case insensitivity handling
        symbol1 = self.symbol_ops.find_symbol(context_sensitive.symbol_name, context_sensitive.symbol_type)
        symbol2 = self.symbol_ops.find_symbol(context_insensitive.symbol_name, context_insensitive.symbol_type)

        if symbol1.file_path and symbol2.file_path:
            self.assertEqual(symbol1.file_path, symbol2.file_path)

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with empty symbol name
        symbol = self.symbol_ops.find_symbol("", SymbolType.FUNCTION)
        self.assertIsInstance(symbol, SymbolInfo)

        # Test with invalid file path
        symbol = self.symbol_ops.find_symbol("test", SymbolType.FUNCTION, scope="invalid_scope")
        self.assertIsInstance(symbol, SymbolInfo)

        # Test with non-existent language
        symbols = self.symbol_ops.find_symbols_by_pattern("test", "nonexistent_lang")
        self.assertIsInstance(symbols, list)

    @patch("fastapply.symbol_operations.RipgrepIntegration")
    def test_ripgrep_integration_failure(self, mock_ripgrep):
        """Test behavior when ripgrep integration fails."""
        # Make ripgrep raise an exception
        mock_ripgrep_instance = MagicMock()
        mock_ripgrep_instance.search_files.side_effect = Exception("Ripgrep failed")
        mock_ripgrep.return_value = mock_ripgrep_instance

        # Recreate symbol ops with mocked ripgrep
        symbol_ops = AdvancedSymbolOperations()

        # Should handle the failure gracefully
        symbol = symbol_ops.find_symbol("test", SymbolType.FUNCTION)
        self.assertIsInstance(symbol, SymbolInfo)

    def test_symbol_metadata_enrichment(self):
        """Test that symbol metadata is properly enriched."""
        symbol = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)

        if symbol.file_path:  # Only test if found
            self.assertIsInstance(symbol.metadata, dict)
            self.assertIn("language", symbol.metadata)
            self.assertEqual(symbol.language, "python")

    def test_symbol_confidence_scoring(self):
        """Test confidence scoring for different symbol matches."""
        # Test exact match
        exact_symbol = self.symbol_ops.find_symbol("TestClass", SymbolType.CLASS)

        # Test semantic match
        semantic_symbol = self.symbol_ops.find_symbol("test_class", SymbolType.CLASS)

        # Exact matches should have higher confidence
        if exact_symbol.file_path and semantic_symbol.file_path:
            self.assertGreaterEqual(exact_symbol.confidence_score, semantic_symbol.confidence_score)


class TestSymbolInfo(unittest.TestCase):
    """Test SymbolInfo dataclass functionality."""

    def test_default_values(self):
        """Test SymbolInfo default values."""
        symbol = SymbolInfo(name="test", symbol_type=SymbolType.FUNCTION, file_path="test.py", line_number=1)

        self.assertEqual(symbol.column_number, 0)
        self.assertEqual(symbol.scope, SymbolScope.GLOBAL)
        self.assertIsNone(symbol.parent_symbol)
        self.assertIsNone(symbol.documentation)
        self.assertEqual(symbol.parameters, [])
        self.assertEqual(symbol.decorators, [])
        self.assertFalse(symbol.is_async)
        self.assertFalse(symbol.is_private)
        self.assertFalse(symbol.is_protected)
        self.assertFalse(symbol.is_static)
        self.assertFalse(symbol.is_abstract)
        self.assertEqual(symbol.confidence_score, 1.0)
        self.assertEqual(symbol.language, "python")
        self.assertEqual(symbol.metadata, {})

    def test_comprehensive_symbol_info(self):
        """Test SymbolInfo with all fields populated."""
        symbol = SymbolInfo(
            name="complex_function",
            symbol_type=SymbolType.ASYNC_FUNCTION,
            file_path="complex.py",
            line_number=42,
            column_number=8,
            scope=SymbolScope.CLASS,
            parent_symbol="ComplexClass",
            documentation="A complex async function",
            signature="async complex_function(param: List[str]) -> Dict[str, Any]",
            return_type="Dict[str, Any]",
            parameters=["param"],
            decorators=["@staticmethod", "@timer"],
            is_async=True,
            is_private=False,
            is_protected=False,
            is_static=True,
            is_abstract=False,
            confidence_score=0.95,
            language="python",
            metadata={"complexity": "high", "lines_of_code": 150},
        )

        self.assertEqual(symbol.name, "complex_function")
        self.assertEqual(symbol.symbol_type, SymbolType.ASYNC_FUNCTION)
        self.assertTrue(symbol.is_async)
        self.assertTrue(symbol.is_static)
        self.assertEqual(symbol.decorators, ["@staticmethod", "@timer"])
        self.assertEqual(symbol.metadata["complexity"], "high")


class TestResolvedScope(unittest.TestCase):
    """Test ResolvedScope functionality."""

    def test_resolved_scope_creation(self):
        """Test ResolvedScope creation with all fields."""
        context = {"file": "test.py", "line": 10}
        alternatives = [SymbolInfo("alt1", SymbolType.VARIABLE, "alt.py", 1), SymbolInfo("alt2", SymbolType.VARIABLE, "alt2.py", 2)]

        resolved = ResolvedScope(
            symbol_name="test_symbol",
            resolved_path="test.py:10",
            scope_level=SymbolScope.FUNCTION,
            context=context,
            confidence=0.85,
            alternative_matches=alternatives,
        )

        self.assertEqual(resolved.symbol_name, "test_symbol")
        self.assertEqual(resolved.resolved_path, "test.py:10")
        self.assertEqual(resolved.scope_level, SymbolScope.FUNCTION)
        self.assertEqual(resolved.context, context)
        self.assertEqual(resolved.confidence, 0.85)
        self.assertEqual(len(resolved.alternative_matches), 2)


class TestReferenceAnalysis(unittest.TestCase):
    """Test the reference analysis functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files with reference patterns
        self.create_test_files()

        # Initialize reference analysis instance
        self.ref_analysis = ReferenceAnalysis()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for reference analysis testing."""
        # Python file with various references
        with open("test_references.py", "w", encoding="utf-8") as f:
            f.write('''
import os
import sys
from typing import List, Dict

GLOBAL_CONSTANT = "test_value"

class TestClass:
    """A test class for reference analysis."""

    def __init__(self, value: int):
        self.value = value
        self._private_var = "private"

    def public_method(self, param: str) -> str:
        return f"Hello {param}"

    def _private_method(self) -> None:
        """Private method for testing."""
        pass

    @property
    def computed_property(self) -> int:
        return self.value * 2

def standalone_function(name: str) -> str:
    return f"Function result: {name}"

def function_with_references():
    # Reference to global constant
    value = GLOBAL_CONSTANT

    # Reference to class
    obj = TestClass(42)

    # Reference to method
    result = obj.public_method("test")

    # Reference to standalone function
    output = standalone_function("reference")

    return result + output

class SubClass(TestClass):
    """Subclass for inheritance testing."""

    def overridden_method(self):
        # Reference to parent method
        return super().public_method("subclass")

    def new_method(self):
        # Reference to global function
        return standalone_function("new")
''')

        # JavaScript file
        with open("app.js", "w", encoding="utf-8") as f:
            f.write("""
const CONFIG = {
    apiUrl: "https://api.example.com",
    timeout: 5000
};

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.users = [];
    }

    async getUsers() {
        const response = await fetch(this.apiUrl + '/users');
        return response.json();
    }

    findUserById(id) {
        return this.users.find(user => user.id === id);
    }
}

function formatDate(date) {
    return date.toISOString();
}

function processData() {
    // Reference to global constant
    const url = CONFIG.apiUrl;

    // Reference to class
    const service = new UserService(url);

    // Reference to method
    const users = service.getUsers();

    // Reference to function
    const formatted = formatDate(new Date());

    return { users, formatted };
}
""")

    def test_reference_info_creation(self):
        """Test ReferenceInfo dataclass creation."""
        ref_info = ReferenceInfo(
            symbol_name="test_function",
            reference_type=ReferenceType.CALL,
            file_path="test.py",
            line_number=15,
            context="function call",
            confidence_score=0.95,
            language="python",
            scope=SymbolScope.GLOBAL
        )

        self.assertEqual(ref_info.symbol_name, "test_function")
        self.assertEqual(ref_info.file_path, "test.py")
        self.assertEqual(ref_info.line_number, 15)
        self.assertEqual(ref_info.reference_type, ReferenceType.CALL)
        self.assertEqual(ref_info.scope, SymbolScope.GLOBAL)
        self.assertEqual(ref_info.confidence_score, 0.95)
        self.assertEqual(ref_info.language, "python")
        self.assertEqual(ref_info.context, "function call")

    def test_analyze_symbol_references_basic(self):
        """Test basic symbol reference analysis."""
        # Find a symbol first
        symbol_ops = AdvancedSymbolOperations()
        symbol = symbol_ops.find_symbol("GLOBAL_CONSTANT", SymbolType.CONSTANT)

        if symbol.file_path:  # Only test if symbol was found
            references = self.ref_analysis.analyze_symbol_references(symbol)

            self.assertIsInstance(references, list)
            # Should find at least the definition and some references
            self.assertGreaterEqual(len(references), 1)

            # Check reference structure
            for ref in references:
                self.assertIsInstance(ref, ReferenceInfo)
                self.assertIsInstance(ref.reference_type, ReferenceType)
                self.assertIsInstance(ref.scope, SymbolScope)

    def test_reference_type_classification(self):
        """Test reference type classification."""
        symbol = SymbolInfo("TestClass", SymbolType.CLASS, "test_references.py", 8)
        references = self.ref_analysis.analyze_symbol_references(symbol)

        if references:
            # Should have different reference types
            reference_types = {ref.reference_type for ref in references}
            self.assertGreater(len(reference_types), 0)

            # Check that types are valid ReferenceType enum values
            for ref_type in reference_types:
                self.assertIn(ref_type, ReferenceType)

    def test_get_symbol_dependencies(self):
        """Test symbol dependency analysis."""
        symbol = SymbolInfo("function_with_references", SymbolType.FUNCTION, "test_references.py", 40)

        dependencies = self.ref_analysis.get_symbol_dependencies(symbol)

        self.assertIsInstance(dependencies, dict)
        self.assertIn("direct", dependencies)
        self.assertIn("transitive", dependencies)

        # Should have direct dependencies
        direct_deps = dependencies["direct"]
        self.assertIsInstance(direct_deps, list)

    def test_analyze_refactoring_safety(self):
        """Test refactoring safety analysis."""
        symbol = SymbolInfo("GLOBAL_CONSTANT", SymbolType.CONSTANT, "test_references.py", 6)

        safety_analysis = self.ref_analysis.analyze_refactoring_safety(symbol)

        self.assertIsInstance(safety_analysis, dict)
        self.assertIn("is_safe_to_rename", safety_analysis)
        self.assertIn("impact_score", safety_analysis)
        self.assertIn("risk_factors", safety_analysis)
        self.assertIn("affected_files", safety_analysis)
        self.assertIn("reference_count", safety_analysis)

        # Impact score should be between 0 and 1
        self.assertGreaterEqual(safety_analysis["impact_score"], 0.0)
        self.assertLessEqual(safety_analysis["impact_score"], 1.0)

    def test_get_reference_statistics(self):
        """Test reference statistics."""
        symbol = SymbolInfo("TestClass", SymbolType.CLASS, "test_references.py", 8)

        stats = self.ref_analysis.get_reference_statistics(symbol)

        self.assertIsInstance(stats, dict)
        self.assertIn("total_references", stats)
        self.assertIn("references_by_type", stats)
        self.assertIn("references_by_file", stats)
        self.assertIn("references_by_scope", stats)
        self.assertIn("average_confidence", stats)
        self.assertIn("reference_density", stats)

        # Total references should be non-negative
        self.assertGreaterEqual(stats["total_references"], 0)

        # Average confidence should be between 0 and 1
        self.assertGreaterEqual(stats["average_confidence"], 0.0)
        self.assertLessEqual(stats["average_confidence"], 1.0)

    def test_find_unused_symbols(self):
        """Test unused symbol detection."""
        unused_symbols = self.ref_analysis.find_unused_symbols("test_references.py", "python")

        self.assertIsInstance(unused_symbols, list)
        # All returned items should be SymbolInfo objects
        for symbol in unused_symbols:
            self.assertIsInstance(symbol, SymbolInfo)

    def test_cross_language_reference_analysis(self):
        """Test reference analysis across different languages."""
        # Test Python
        python_symbol = SymbolInfo("formatDate", SymbolType.FUNCTION, "app.js", 25)
        python_refs = self.ref_analysis.analyze_symbol_references(python_symbol)

        # Test JavaScript
        js_symbol = SymbolInfo("TestClass", SymbolType.CLASS, "test_references.py", 8)
        js_refs = self.ref_analysis.analyze_symbol_references(js_symbol)

        # Should work for both languages
        self.assertIsInstance(python_refs, list)
        self.assertIsInstance(js_refs, list)

    def test_reference_pattern_extraction(self):
        """Test that reference patterns are correctly extracted."""
        # This tests the internal pattern matching logic
        symbol = SymbolInfo("UserService", SymbolType.CLASS, "app.js", 10)
        references = self.ref_analysis.analyze_symbol_references(symbol)

        if references:
            # Check that references have proper context
            for ref in references:
                self.assertIsInstance(ref.line_content, str)
                self.assertIsInstance(ref.context, str)
                self.assertGreater(ref.confidence_score, 0.0)

    def test_reference_scope_detection(self):
        """Test that reference scopes are correctly detected."""
        symbol = SymbolInfo("computed_property", SymbolType.PROPERTY, "test_references.py", 30)
        references = self.ref_analysis.analyze_symbol_references(symbol)

        if references:
            # Should have different scopes
            scopes = {ref.scope for ref in references}
            self.assertGreater(len(scopes), 0)

            # All scopes should be valid SymbolScope enum values
            for scope in scopes:
                self.assertIn(scope, SymbolScope)

    def test_reference_confidence_scoring(self):
        """Test that reference confidence scores are reasonable."""
        symbol = SymbolInfo("public_method", SymbolType.METHOD, "test_references.py", 18)
        references = self.ref_analysis.analyze_symbol_references(symbol)

        if references:
            for ref in references:
                # Confidence should be between 0 and 1
                self.assertGreaterEqual(ref.confidence_score, 0.0)
                self.assertLessEqual(ref.confidence_score, 1.0)

    def test_error_handling_invalid_symbol(self):
        """Test error handling for invalid symbols."""
        symbol = SymbolInfo("nonexistent_symbol", SymbolType.FUNCTION, "nonexistent.py", 1)

        # Should handle gracefully without exceptions
        references = self.ref_analysis.analyze_symbol_references(symbol)
        self.assertIsInstance(references, list)

        dependencies = self.ref_analysis.get_symbol_dependencies(symbol)
        self.assertIsInstance(dependencies, dict)

        safety = self.ref_analysis.analyze_refactoring_safety(symbol)
        self.assertIsInstance(safety, dict)

        stats = self.ref_analysis.get_reference_statistics(symbol)
        self.assertIsInstance(stats, dict)

    def test_large_file_performance(self):
        """Test performance with larger files."""
        # Create a larger test file
        with open("large_test.py", "w", encoding="utf-8") as f:
            f.write("""
# Large test file for performance testing
import os
import sys
import json
from typing import List, Dict, Any

LARGE_CONSTANT = "large_value"

class LargeClass:
    def __init__(self):
        self.data = []

    def method1(self): pass
    def method2(self): pass
    def method3(self): pass

    def complex_method(self):
        result = LARGE_CONSTANT
        for i in range(100):
            result += str(i)
        return result

def large_function():
    return LARGE_CONSTANT + " processed"

# Multiple references to test performance
def test_function1():
    return LARGE_CONSTANT

def test_function2():
    return LARGE_CONSTANT

def test_function3():
    return LARGE_CONSTANT

def test_function4():
    return LARGE_CONSTANT

def test_function5():
    return LARGE_CONSTANT
""")

        # Test with a symbol from the large file
        symbol = SymbolInfo("LARGE_CONSTANT", SymbolType.CONSTANT, "large_test.py", 8)

        # Should still perform reasonably
        references = self.ref_analysis.analyze_symbol_references(symbol)
        self.assertIsInstance(references, list)

        # Should find multiple references
        self.assertGreater(len(references), 3)

    def test_reference_metadata_enrichment(self):
        """Test that reference metadata is properly enriched."""
        symbol = SymbolInfo("UserService", SymbolType.CLASS, "app.js", 10)
        references = self.ref_analysis.analyze_symbol_references(symbol)

        if references:
            for ref in references:
                self.assertIsInstance(ref.metadata, dict)
                self.assertIn("language", ref.metadata)
                self.assertEqual(ref.language, "javascript")


if __name__ == "__main__":
    unittest.main()
