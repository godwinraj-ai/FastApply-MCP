"""
Simplified test suite for reference analysis functionality.

Tests the core functionality with actual implementation details.
"""

import os
import shutil
import tempfile
import unittest

# Import the module under test
from fastapply.symbol_operations import ReferenceAnalysis, ReferenceInfo, ReferenceType, SymbolInfo, SymbolScope, SymbolType


class TestReferenceAnalysisSimple(unittest.TestCase):
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
        """Create test files with various reference patterns."""
        # Python files with complex references
        with open("module1.py", "w", encoding="utf-8") as f:
            f.write("""
class TestClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

def process_data(data):
    return TestClass(data).get_value()
""")

        with open("module2.py", "w", encoding="utf-8") as f:
            f.write("""
from module1 import TestClass, process_data

def use_class():
    instance = TestClass(10)
    result = instance.get_value()
    return process_data(result)
""")

    def test_reference_type_enum(self):
        """Test ReferenceType enum values."""
        self.assertEqual(ReferenceType.READ.value, "read")
        self.assertEqual(ReferenceType.WRITE.value, "write")
        self.assertEqual(ReferenceType.CALL.value, "call")
        self.assertEqual(ReferenceType.IMPORT.value, "import")
        self.assertEqual(ReferenceType.INHERIT.value, "inherit")
        self.assertEqual(ReferenceType.IMPLEMENT.value, "implement")
        self.assertEqual(ReferenceType.OVERRIDE.value, "override")
        self.assertEqual(ReferenceType.REFERENCE.value, "reference")
        self.assertEqual(ReferenceType.TYPE_ANNOTATION.value, "type_annotation")
        self.assertEqual(ReferenceType.DECORATE.value, "decorate")

    def test_symbol_scope_enum(self):
        """Test SymbolScope enum values."""
        self.assertEqual(SymbolScope.GLOBAL.value, "global")
        self.assertEqual(SymbolScope.MODULE.value, "module")
        self.assertEqual(SymbolScope.CLASS.value, "class")
        self.assertEqual(SymbolScope.FUNCTION.value, "function")
        self.assertEqual(SymbolScope.LOCAL.value, "local")
        self.assertEqual(SymbolScope.BLOCK.value, "block")

    def test_reference_info_creation(self):
        """Test ReferenceInfo dataclass creation."""
        ref_info = ReferenceInfo(
            symbol_name="TestClass",
            reference_type=ReferenceType.IMPORT,
            file_path="module2.py",
            line_number=3,
            context="from module1 import TestClass",
            confidence_score=0.95,
            scope=SymbolScope.MODULE
        )

        self.assertEqual(ref_info.symbol_name, "TestClass")
        self.assertEqual(ref_info.reference_type, ReferenceType.IMPORT)
        self.assertEqual(ref_info.file_path, "module2.py")
        self.assertEqual(ref_info.line_number, 3)
        self.assertEqual(ref_info.context, "from module1 import TestClass")
        self.assertEqual(ref_info.confidence_score, 0.95)
        self.assertEqual(ref_info.scope, SymbolScope.MODULE)

    def test_analyze_symbol_references_basic(self):
        """Test basic symbol reference analysis."""
        # Create a test symbol info
        symbol_info = SymbolInfo(
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="module1.py",
            line_number=2,
            scope=SymbolScope.MODULE
        )

        # Test reference analysis with actual files
        references = self.ref_analysis.analyze_symbol_references(
            symbol_info,
            include_definitions=False,
            max_results=10
        )

        self.assertIsInstance(references, list)
        # Should find references in the test files
        self.assertGreater(len(references), 0)

        # Verify reference properties
        for ref in references:
            self.assertIsInstance(ref, ReferenceInfo)
            self.assertIsInstance(ref.reference_type, ReferenceType)
            self.assertIsInstance(ref.scope, SymbolScope)
            self.assertGreaterEqual(ref.confidence_score, 0.0)
            self.assertLessEqual(ref.confidence_score, 1.0)
            self.assertEqual(ref.symbol_name, "TestClass")

    def test_dependency_analysis(self):
        """Test dependency analysis functionality."""
        symbol_info = SymbolInfo(
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="module1.py",
            line_number=2,
            scope=SymbolScope.MODULE
        )

        # Get references first
        references = self.ref_analysis.analyze_symbol_references(symbol_info, max_results=10)

        # Then analyze dependencies
        dependencies = self.ref_analysis.get_symbol_dependencies(symbol_info, references)

        self.assertIsInstance(dependencies, dict)
        self.assertIn("direct", dependencies)
        self.assertIn("transitive", dependencies)
        self.assertIsInstance(dependencies["direct"], list)
        self.assertIsInstance(dependencies["transitive"], list)

    def test_reference_statistics(self):
        """Test reference statistics calculation."""
        symbol_info = SymbolInfo(
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="module1.py",
            line_number=2,
            scope=SymbolScope.MODULE
        )

        # Calculate statistics
        stats = self.ref_analysis.get_reference_statistics(symbol_info)

        self.assertIsInstance(stats, dict)
        self.assertIn("total_references", stats)
        self.assertIn("references_by_type", stats)
        self.assertIn("references_by_file", stats)
        self.assertIn("average_confidence", stats)

        # Verify statistics are calculated correctly
        self.assertIsInstance(stats["references_by_type"], dict)
        self.assertIsInstance(stats["references_by_file"], dict)

    def test_unused_symbol_detection(self):
        """Test unused symbol detection."""
        # This test might not find unused symbols in our simple test case
        # but it should run without errors
        unused_symbols = self.ref_analysis.find_unused_symbols(
            file_path=self.test_dir,
            language="python"
        )

        self.assertIsInstance(unused_symbols, list)

    def test_error_handling(self):
        """Test error handling with non-existent paths."""
        symbol_info = SymbolInfo(
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="nonexistent.py",
            line_number=1,
            scope=SymbolScope.MODULE
        )

        # Should handle gracefully without exceptions
        references = self.ref_analysis.analyze_symbol_references(
            symbol_info,
            include_definitions=False,
            max_results=10
        )

        self.assertIsInstance(references, list)



if __name__ == "__main__":
    unittest.main()
