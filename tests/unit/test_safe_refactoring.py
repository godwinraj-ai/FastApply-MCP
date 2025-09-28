#!/usr/bin/env python3
"""
Comprehensive test suite for safe refactoring tools.

Tests SafeSymbolRenaming and CodeExtractionAndMovement classes with coverage for
all major functionality including safety validation, rollback capabilities, and
error handling scenarios.

Phase 5 Implementation Tests - Safe Refactoring Execution Tools
"""

import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime
from unittest.mock import Mock

from fastapply.safe_refactoring import (
    CodeExtractionAndMovement,
    ExtractionOperation,
    ImpactAnalysis,
    MoveOperation,
    RefactoringResult,
    RenameOperation,
    RollbackPlan,
    SafeSymbolRenaming,
    SymbolInfo,
    SymbolScope,
    SymbolType,
    create_safe_refactoring_tools,
)
from fastapply.symbol_operations import ReferenceInfo, ReferenceType


class TestSafeRefactoringDataClasses(unittest.TestCase):
    """Test safe refactoring dataclasses and enums."""

    def test_refactoring_result_enum(self):
        """Test RefactoringResult enum values."""
        self.assertEqual(RefactoringResult.SUCCESS.value, "success")
        self.assertEqual(RefactoringResult.PARTIAL.value, "partial")
        self.assertEqual(RefactoringResult.FAILED.value, "failed")
        self.assertEqual(RefactoringResult.CANCELLED.value, "cancelled")

    def test_rename_operation_creation(self):
        """Test RenameOperation dataclass creation."""
        ref_info = Mock()
        ref_info.symbol_name = "test_function"

        rename_op = RenameOperation(
            old_name="test_function",
            new_name="new_function_name",
            symbol_type="function",
            file_path="test.py",
            scope="global",
            references=[ref_info]
        )

        self.assertEqual(rename_op.old_name, "test_function")
        self.assertEqual(rename_op.new_name, "new_function_name")
        self.assertEqual(rename_op.symbol_type, "function")
        self.assertEqual(rename_op.file_path, "test.py")
        self.assertEqual(rename_op.scope, "global")
        self.assertEqual(len(rename_op.references), 1)

    def test_extraction_operation_creation(self):
        """Test ExtractionOperation dataclass creation."""
        extraction_op = ExtractionOperation(
            source_range=(10, 20),
            target_name="extracted_function",
            target_file="utils.py",
            extraction_type="function",
            dependencies=["os", "sys"]
        )

        self.assertEqual(extraction_op.source_range, (10, 20))
        self.assertEqual(extraction_op.target_name, "extracted_function")
        self.assertEqual(extraction_op.target_file, "utils.py")
        self.assertEqual(extraction_op.extraction_type, "function")
        self.assertEqual(extraction_op.dependencies, ["os", "sys"])

    def test_move_operation_creation(self):
        """Test MoveOperation dataclass creation."""
        move_op = MoveOperation(
            symbol_name="TestClass",
            source_file="old_module.py",
            target_file="new_module.py",
            symbol_type="class",
            scope="global"
        )

        self.assertEqual(move_op.symbol_name, "TestClass")
        self.assertEqual(move_op.source_file, "old_module.py")
        self.assertEqual(move_op.target_file, "new_module.py")
        self.assertEqual(move_op.symbol_type, "class")
        self.assertEqual(move_op.scope, "global")

    def test_rollback_plan_creation(self):
        """Test RollbackPlan dataclass creation."""
        original_files = {"test.py": "original content"}
        backup_files = {"test.py": "/tmp/test.py.backup"}
        operation_log = ["Backup created", "Operation started"]

        rollback_plan = RollbackPlan(
            original_files=original_files,
            backup_files=backup_files,
            operation_log=operation_log,
            timestamp=datetime(2025, 1, 1, 12, 0, 0)
        )

        self.assertEqual(rollback_plan.original_files, original_files)
        self.assertEqual(rollback_plan.backup_files, backup_files)
        self.assertEqual(rollback_plan.operation_log, operation_log)
        self.assertEqual(rollback_plan.timestamp, datetime(2025, 1, 1, 12, 0, 0))

    def test_impact_analysis_creation(self):
        """Test ImpactAnalysis dataclass creation."""
        impact = ImpactAnalysis(
            affected_files={"test.py", "utils.py"},
            affected_symbols={"function1", "function2"},
            external_dependencies={"external_module"},
            test_impact=True,
            breaking_changes=False,
            risk_score=0.3
        )

        self.assertEqual(impact.affected_files, {"test.py", "utils.py"})
        self.assertEqual(impact.affected_symbols, {"function1", "function2"})
        self.assertEqual(impact.external_dependencies, {"external_module"})
        self.assertTrue(impact.test_impact)
        self.assertFalse(impact.breaking_changes)
        self.assertEqual(impact.risk_score, 0.3)


class TestSafeSymbolRenaming(unittest.TestCase):
    """Test SafeSymbolRenaming functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files
        self.create_test_files()

        # Mock search engine
        self.mock_search_engine = Mock()
        self.safe_rename = SafeSymbolRenaming(self.mock_search_engine)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for renaming tests."""
        # Python file with function to rename
        with open("test_rename.py", "w", encoding="utf-8") as f:
            f.write('''
def old_function_name():
    """Function to be renamed."""
    return "Hello, World!"

def caller_function():
    # Reference to old_function_name
    result = old_function_name()
    return result + " processed"

class TestClass:
    def method_with_reference(self):
        # Reference to old_function_name
        return old_function_name()
''')

        # Another file with references
        with open("other_file.py", "w", encoding="utf-8") as f:
            f.write('''
from test_rename import old_function_name

def external_function():
    # External reference
    return old_function_name()
''')

    def test_initialization(self):
        """Test SafeSymbolRenaming initialization."""
        self.assertIsNotNone(self.safe_rename.search_engine)
        self.assertIsInstance(self.safe_rename.rollback_plans, dict)
        self.assertIsInstance(self.safe_rename.operation_lock, threading.Lock)

    def test_rename_symbol_safely_success(self):
        """Test successful symbol renaming."""
        # Mock symbol finding
        symbol_info = SymbolInfo(
            name="old_function_name",
            symbol_type=SymbolType.FUNCTION,
            file_path="test_rename.py",
            line_number=3,
            scope=SymbolScope.GLOBAL
        )
        self.safe_rename._find_symbol_definition = Mock(return_value=symbol_info)

        # Mock reference finding
        references = [
            ReferenceInfo(
                symbol_name="old_function_name",
                file_path="test_rename.py",
                line_number=8,
                context="function call",
                reference_type=ReferenceType.CALL
            ),
            ReferenceInfo(
                symbol_name="old_function_name",
                file_path="test_rename.py",
                line_number=13,
                context="function call",
                reference_type=ReferenceType.CALL
            ),
            ReferenceInfo(
                symbol_name="old_function_name",
                file_path="other_file.py",
                line_number=3,
                context="function call",
                reference_type=ReferenceType.CALL
            )
        ]
        self.safe_rename._find_all_references = Mock(return_value=references)

        # Mock impact analysis
        impact_analysis = ImpactAnalysis(
            affected_files={"test_rename.py", "other_file.py"},
            affected_symbols={"old_function_name"},
            external_dependencies=set(),
            test_impact=False,
            breaking_changes=False,
            risk_score=0.1
        )
        self.safe_rename.analyze_rename_impact = Mock(return_value=impact_analysis)

        # Mock the rename operations to succeed
        self.safe_rename._rename_symbol_in_file = Mock()
        self.safe_rename._rename_reference_in_file = Mock(return_value=True)

        result = self.safe_rename.rename_symbol_safely(
            "old_function_name", "new_function_name", "function"
        )

        self.assertEqual(result["status"], RefactoringResult.SUCCESS.value)
        self.assertEqual(result["symbol_renamed"], "old_function_name")
        self.assertEqual(result["new_name"], "new_function_name")
        self.assertEqual(result["references_updated"], 3)
        self.assertEqual(result["files_affected"], 2)  # test_rename.py + other_file.py = 2 unique files
        self.assertIn("operation_id", result)

    def test_rename_symbol_safely_symbol_not_found(self):
        """Test renaming when symbol is not found."""
        self.safe_rename._find_symbol_definition = Mock(return_value=None)

        # Mock impact analysis to pass safety validation
        safe_impact_analysis = ImpactAnalysis(
            affected_files=set(),
            affected_symbols=set(),
            external_dependencies=set(),
            test_impact=False,
            breaking_changes=False,
            risk_score=0.1
        )
        self.safe_rename.analyze_rename_impact = Mock(return_value=safe_impact_analysis)

        result = self.safe_rename.rename_symbol_safely(
            "nonexistent_function", "new_name", "function"
        )

        self.assertEqual(result["status"], RefactoringResult.FAILED.value)
        self.assertIn("not found", result["error"])

    def test_rename_symbol_safely_unsafe_operation(self):
        """Test renaming when operation is deemed unsafe."""
        symbol_info = SymbolInfo(
            name="risky_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )
        self.safe_rename._find_symbol_definition = Mock(return_value=symbol_info)

        # Mock unsafe impact analysis
        impact_analysis = ImpactAnalysis(
            affected_files={"test.py"},
            affected_symbols={"risky_function"},
            external_dependencies={"external_api"},
            test_impact=True,
            breaking_changes=True,
            risk_score=0.9  # High risk
        )
        self.safe_rename.analyze_rename_impact = Mock(return_value=impact_analysis)

        result = self.safe_rename.rename_symbol_safely(
            "risky_function", "new_name", "function"
        )

        self.assertEqual(result["status"], RefactoringResult.FAILED.value)
        self.assertIn("deemed unsafe", result["error"])

    def test_analyze_rename_impact_success(self):
        """Test rename impact analysis."""
        symbol_info = SymbolInfo(
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )
        self.safe_rename._find_symbol_definition = Mock(return_value=symbol_info)

        references = [
            Mock(file_path="test.py", line_number=10),
            Mock(file_path="test.py", line_number=20)
        ]
        self.safe_rename._find_all_references = Mock(return_value=references)

        self.safe_rename._check_name_conflicts = Mock(return_value=[])
        self.safe_rename._identify_external_apis = Mock(return_value=set())

        impact = self.safe_rename.analyze_rename_impact(
            "test_function", "new_function", "function"
        )

        self.assertIsInstance(impact, ImpactAnalysis)
        self.assertEqual(impact.affected_files, {"test.py"})
        self.assertEqual(impact.risk_score, 0.02)  # Based on 2 references

    def test_analyze_rename_impact_symbol_not_found(self):
        """Test impact analysis when symbol is not found."""
        self.safe_rename._find_symbol_definition = Mock(return_value=None)

        impact = self.safe_rename.analyze_rename_impact(
            "nonexistent_function", "new_name", "function"
        )

        self.assertIsInstance(impact, ImpactAnalysis)
        self.assertEqual(impact.risk_score, 1.0)  # Maximum risk
        self.assertTrue(impact.breaking_changes)

    def test_generate_rollback_plan(self):
        """Test rollback plan generation."""
        symbol_info = SymbolInfo(
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )

        references = [
            Mock(file_path="test.py", line_number=10),
            Mock(file_path="other.py", line_number=5)
        ]

        # Create test files
        with open("test.py", "w") as f:
            f.write("def test_function():\n    pass")
        with open("other.py", "w") as f:
            f.write("print(test_function())")

        rollback_plan = self.safe_rename.generate_rollback_plan(
            symbol_info, "new_name", references
        )

        self.assertIsInstance(rollback_plan, RollbackPlan)
        self.assertIn("test.py", rollback_plan.original_files)
        self.assertIn("other.py", rollback_plan.original_files)
        self.assertEqual(len(rollback_plan.backup_files), 2)
        self.assertGreater(len(rollback_plan.operation_log), 0)

    def test_validate_rename_safety(self):
        """Test rename safety validation."""
        # Safe case
        safe_impact = ImpactAnalysis(
            affected_files={"test.py"},
            affected_symbols={"test_func"},
            external_dependencies=set(),
            test_impact=False,
            breaking_changes=False,
            risk_score=0.3
        )
        self.assertTrue(self.safe_rename.validate_rename_safety(safe_impact))

        # Unsafe case - high risk
        unsafe_impact = ImpactAnalysis(
            affected_files={"test.py"},
            affected_symbols={"test_func"},
            external_dependencies=set(),
            test_impact=False,
            breaking_changes=False,
            risk_score=0.9
        )
        self.assertFalse(self.safe_rename.validate_rename_safety(unsafe_impact))

        # Unsafe case - too many affected files
        many_files_impact = ImpactAnalysis(
            affected_files={f"file_{i}.py" for i in range(100)},
            affected_symbols={"test_func"},
            external_dependencies=set(),
            test_impact=False,
            breaking_changes=False,
            risk_score=0.5
        )
        self.assertFalse(self.safe_rename.validate_rename_safety(many_files_impact))

    def test_execute_rollback_success(self):
        """Test successful rollback execution."""
        # Create test files
        with open("test.py", "w") as f:
            f.write("modified content")

        rollback_plan = RollbackPlan(
            original_files={"test.py": "original content"},
            backup_files={"test.py": "/tmp/test.py.backup"},
            operation_log=["Backup created"]
        )

        operation_id = "test_operation"
        self.safe_rename.rollback_plans[operation_id] = rollback_plan

        result = self.safe_rename.execute_rollback(operation_id)

        self.assertEqual(result["status"], RefactoringResult.SUCCESS.value)
        self.assertEqual(result["files_restored"], 1)
        self.assertIn("Backup created", result["operation_log"])
        self.assertNotIn(operation_id, self.safe_rename.rollback_plans)

    def test_execute_rollback_not_found(self):
        """Test rollback when operation ID is not found."""
        result = self.safe_rename.execute_rollback("nonexistent_operation")

        self.assertEqual(result["status"], RefactoringResult.FAILED.value)
        self.assertIn("No rollback plan found", result["error"])

    def test_find_symbol_definition_success(self):
        """Test successful symbol definition finding."""
        from fastapply.enhanced_search import EnhancedSearchResult

        # Mock search results
        search_result = EnhancedSearchResult(
            file_path="test.py",
            line_number=5,
            line_content="def test_function():",
            context_before=["function definition"],
            confidence_score=0.9,
            symbol_type="function"
        )
        self.mock_search_engine.search.return_value = [search_result]

        symbol = self.safe_rename._find_symbol_definition(
            "test_function", "function", "global", "."
        )

        self.assertIsInstance(symbol, SymbolInfo)
        self.assertEqual(symbol.name, "test_function")
        self.assertEqual(symbol.file_path, "test.py")
        self.assertEqual(symbol.line_number, 5)

    def test_find_symbol_definition_not_found(self):
        """Test symbol definition finding when no matches."""
        self.mock_search_engine.search.return_value = []

        symbol = self.safe_rename._find_symbol_definition(
            "nonexistent_function", "function", "global", "."
        )

        self.assertIsNone(symbol)

    def test_find_all_references(self):
        """Test finding all references to a symbol."""
        from fastapply.enhanced_search import EnhancedSearchResult

        symbol_info = SymbolInfo(
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )

        # Mock search results
        search_results = [
            EnhancedSearchResult(
                file_path="test.py",
                line_number=10,
                line_content="result = test_function()",
                context_before=["function call"],
                confidence_score=0.9,
                symbol_type="function"
            ),
            EnhancedSearchResult(
                file_path="other.py",
                line_number=5,
                line_content="print(test_function())",
                context_before=["function call"],
                confidence_score=0.9,
                symbol_type="function"
            )
        ]
        self.mock_search_engine.search.return_value = search_results

        references = self.safe_rename._find_all_references(symbol_info, ".")

        self.assertEqual(len(references), 2)
        self.assertEqual(references[0].file_path, "test.py")
        self.assertEqual(references[0].line_number, 10)
        self.assertEqual(references[1].file_path, "other.py")
        self.assertEqual(references[1].line_number, 5)

    def test_check_name_conflicts(self):
        """Test name conflict checking."""
        from fastapply.enhanced_search import EnhancedSearchResult

        # Mock conflicting results
        conflict_result = EnhancedSearchResult(
            file_path="conflict.py",
            line_number=3,
            line_content="def existing_function():",
            context_before=["function definition"],
            confidence_score=0.9,
            symbol_type="function"
        )
        self.mock_search_engine.search.return_value = [conflict_result]

        conflicts = self.safe_rename._check_name_conflicts(
            "existing_function", "function", "."
        )

        self.assertEqual(len(conflicts), 1)
        self.assertIn("conflict.py", conflicts[0])

    def test_calculate_rename_risk_score(self):
        """Test risk score calculation."""
        # Low risk case
        low_risk = self.safe_rename._calculate_rename_risk_score(
            reference_count=5,
            external_api_count=0,
            test_file_count=1,
            has_conflicts=False
        )
        self.assertLess(low_risk, 0.3)

        # High risk case
        high_risk = self.safe_rename._calculate_rename_risk_score(
            reference_count=100,
            external_api_count=5,
            test_file_count=10,
            has_conflicts=True
        )
        self.assertGreater(high_risk, 0.8)

    def test_create_backup_file(self):
        """Test backup file creation."""
        # Create a test file
        test_file = "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        backup_path = self.safe_rename._create_backup_file(test_file)

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith(".backup"))
        self.assertIn("test_file", backup_path)

        # Verify backup content
        with open(backup_path, "r") as f:
            backup_content = f.read()
        self.assertEqual(backup_content, "test content")

    def test_thread_safety(self):
        """Test thread safety of operations."""
        results = []
        errors = []

        def worker(op_id):
            try:
                result = self.safe_rename.rename_symbol_safely(
                    f"test_func_{op_id}", f"new_func_{op_id}", "function"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have results for all operations
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)


class TestCodeExtractionAndMovement(unittest.TestCase):
    """Test CodeExtractionAndMovement functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files
        self.create_test_files()

        # Mock search engine
        self.mock_search_engine = Mock()
        self.code_ops = CodeExtractionAndMovement(self.mock_search_engine)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        """Create test files for extraction and movement tests."""
        # Source file with extractable function
        with open("source.py", "w", encoding="utf-8") as f:
            f.write('''
import os
import sys

def main_function():
    """Main function with extractable logic."""
    # Extractable code starts here
    data = process_data()
    result = format_output(data)
    return result

    # More extractable code
    extra_data = get_extra_data()
    return result + extra_data

def helper_function():
    """Helper function."""
    return "helper result"

class TestClass:
    def extractable_method(self):
        """Method with extractable logic."""
        value = self.calculate_value()
        return value * 2
''')

        # Target file for extraction
        with open("target.py", "w", encoding="utf-8") as f:
            f.write('''
# Target file for extracted functions
import utils

def existing_function():
    return "existing"
''')

    def test_initialization(self):
        """Test CodeExtractionAndMovement initialization."""
        self.assertIsNotNone(self.code_ops.search_engine)
        self.assertIsInstance(self.code_ops.rollback_plans, dict)
        self.assertIsInstance(self.code_ops.operation_lock, threading.Lock)

    def test_extract_function_safely_success(self):
        """Test successful function extraction."""
        # Mock safety analysis
        safety_analysis = {
            "is_safe": True,
            "safety_score": 0.9,
            "dependencies": [],
            "external_references": [],
            "risk_factors": []
        }
        self.code_ops.analyze_extraction_safety = Mock(return_value=safety_analysis)

        # Mock rollback plan creation
        rollback_plan = RollbackPlan(
            original_files={},
            backup_files={},
            operation_log=[]
        )
        self.code_ops._create_extraction_rollback_plan = Mock(return_value=rollback_plan)

        result = self.code_ops.extract_function_safely(
            source_range=(10, 15),
            function_name="extracted_function",
            target_file="target.py",
            source_file="source.py"
        )

        self.assertEqual(result["status"], RefactoringResult.SUCCESS.value)
        self.assertEqual(result["function_name"], "extracted_function")
        self.assertEqual(result["source_file"], "source.py")
        self.assertEqual(result["target_file"], "target.py")
        self.assertIn("operation_id", result)

    def test_extract_function_safely_unsafe(self):
        """Test function extraction when deemed unsafe."""
        # Mock unsafe safety analysis
        safety_analysis = {
            "is_safe": False,
            "safety_score": 0.3,
            "dependencies": ["complex_module"],
            "external_references": ["global_var"],
            "risk_factors": ["Uses global variables"]
        }
        self.code_ops.analyze_extraction_safety = Mock(return_value=safety_analysis)

        result = self.code_ops.extract_function_safely(
            source_range=(10, 15),
            function_name="extracted_function",
            target_file="target.py",
            source_file="source.py"
        )

        self.assertEqual(result["status"], RefactoringResult.FAILED.value)
        self.assertIn("deemed unsafe", result["error"])
        self.assertIn("safety_analysis", result)

    def test_move_symbol_safely_success(self):
        """Test successful symbol movement."""
        # Create test files
        with open("move_source.py", "w") as f:
            f.write("def move_function():\n    return 'moved'")
        with open("move_target.py", "w") as f:
            f.write("# Target file")

        # Mock safety analysis
        safety_analysis = {
            "is_safe": True,
            "safety_score": 0.9,
            "dependencies": [],
            "circular_dependencies": [],
            "risk_factors": []
        }
        self.code_ops.analyze_movement_safety = Mock(return_value=safety_analysis)

        # Mock symbol finding
        symbol_info = SymbolInfo(
            name="move_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="move_source.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )
        self.code_ops._find_symbol_definition = Mock(return_value=symbol_info)

        # Mock rollback plan
        rollback_plan = RollbackPlan(
            original_files={},
            backup_files={},
            operation_log=[]
        )
        self.code_ops._create_movement_rollback_plan = Mock(return_value=rollback_plan)

        result = self.code_ops.move_symbol_safely(
            symbol_name="move_function",
            source_file="move_source.py",
            target_file="move_target.py",
            symbol_type="function"
        )

        self.assertEqual(result["status"], RefactoringResult.SUCCESS.value)
        self.assertEqual(result["symbol_name"], "move_function")
        self.assertEqual(result["source_file"], "move_source.py")
        self.assertEqual(result["target_file"], "move_target.py")
        self.assertIn("operation_id", result)

    def test_move_symbol_safely_symbol_not_found(self):
        """Test symbol movement when symbol is not found."""
        # Mock the analyze_movement_safety method to return safe so we get to the symbol check
        self.code_ops.analyze_movement_safety = Mock(return_value={"is_safe": True})
        self.code_ops._find_symbol_definition = Mock(return_value=None)

        result = self.code_ops.move_symbol_safely(
            symbol_name="nonexistent_function",
            source_file="source.py",
            target_file="target.py",
            symbol_type="function"
        )

        self.assertEqual(result["status"], RefactoringResult.FAILED.value)
        self.assertIn("not found", result["error"])

    def test_analyze_extraction_safety_success(self):
        """Test successful extraction safety analysis."""
        with open("test_source.py", "w") as f:
            f.write("def function():\n    return 'test'\n    # More lines\n    return 'done'")

        result = self.code_ops.analyze_extraction_safety(
            source_range=(2, 4),
            source_file="test_source.py"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("is_safe", result)
        self.assertIn("safety_score", result)
        self.assertIn("dependencies", result)
        self.assertIn("external_references", result)
        self.assertIn("risk_factors", result)

    def test_analyze_extraction_safety_file_not_found(self):
        """Test extraction safety analysis when file doesn't exist."""
        result = self.code_ops.analyze_extraction_safety(
            source_range=(1, 5),
            source_file="nonexistent.py"
        )

        self.assertFalse(result["is_safe"])
        self.assertIn("does not exist", result["error"])

    def test_analyze_extraction_safety_invalid_range(self):
        """Test extraction safety analysis with invalid range."""
        with open("test_source.py", "w") as f:
            f.write("def function():\n    return 'test'")

        result = self.code_ops.analyze_extraction_safety(
            source_range=(10, 20),  # Beyond file length
            source_file="test_source.py"
        )

        self.assertFalse(result["is_safe"])
        self.assertIn("Invalid source range", result["error"])

    def test_analyze_movement_safety_success(self):
        """Test successful movement safety analysis."""
        # Create test files
        with open("move_source.py", "w") as f:
            f.write("def test_function():\n    return 'test'")
        os.makedirs("target_dir", exist_ok=True)
        with open("target_dir/move_target.py", "w") as f:
            f.write("# Target file")

        # Mock symbol finding
        symbol_info = SymbolInfo(
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="move_source.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )
        self.code_ops._find_symbol_definition = Mock(return_value=symbol_info)

        result = self.code_ops.analyze_movement_safety(
            symbol_name="test_function",
            source_file="move_source.py",
            target_file="target_dir/move_target.py"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("is_safe", result)
        self.assertIn("safety_score", result)

    def test_analyze_movement_safety_source_not_found(self):
        """Test movement safety analysis when source file doesn't exist."""
        result = self.code_ops.analyze_movement_safety(
            symbol_name="test_function",
            source_file="nonexistent.py",
            target_file="target.py"
        )

        self.assertFalse(result["is_safe"])
        self.assertIn("Source file does not exist", result["error"])

    def test_analyze_movement_safety_target_dir_not_found(self):
        """Test movement safety analysis when target directory doesn't exist."""
        with open("move_source.py", "w") as f:
            f.write("def test_function():\n    return 'test'")

        result = self.code_ops.analyze_movement_safety(
            symbol_name="test_function",
            source_file="move_source.py",
            target_file="nonexistent_dir/target.py"
        )

        self.assertFalse(result["is_safe"])
        self.assertIn("Target directory does not exist", result["error"])

    def test_manage_import_dependencies(self):
        """Test import dependency management."""
        move_op = MoveOperation(
            symbol_name="test_function",
            source_file="source.py",
            target_file="target.py",
            symbol_type="function"
        )

        # Mock import analysis
        self.code_ops._analyze_required_imports = Mock(return_value=["import os", "import sys"])
        self.code_ops._analyze_unused_imports = Mock(return_value=["import unused_module"])
        self.code_ops._add_import_to_file = Mock(return_value=True)
        self.code_ops._remove_import_from_file = Mock(return_value=True)

        result = self.code_ops.manage_import_dependencies(move_op)

        self.assertIsInstance(result, dict)
        self.assertIn("imports_added", result)
        self.assertIn("imports_removed", result)
        self.assertIn("exports_added", result)
        self.assertIn("exports_removed", result)
        self.assertIn("errors", result)

    def test_analyze_code_dependencies(self):
        """Test code dependency analysis."""
        code = '''
import os
import sys
from typing import List
import external_module

def function():
    os.path.join()
    sys.path.append()
    List[str]()
    external_module.function()
'''

        dependencies = self.code_ops._analyze_code_dependencies(code, "test.py")

        self.assertIsInstance(dependencies, list)
        self.assertIn("os", dependencies)
        self.assertIn("sys", dependencies)
        self.assertIn("typing", dependencies)
        self.assertIn("external_module", dependencies)

    def test_analyze_external_references(self):
        """Test external reference analysis."""
        code = '''
def function():
    global_var = value
    another_function()
    return result + global_constant
'''

        external_refs = self.code_ops._analyze_external_references(code, "test.py")

        self.assertIsInstance(external_refs, list)
        # Should find variable references (filtering out keywords)
        self.assertIn("global_var", external_refs)
        self.assertIn("another_function", external_refs)
        self.assertIn("result", external_refs)
        self.assertIn("global_constant", external_refs)

    def test_calculate_extraction_safety_score(self):
        """Test extraction safety score calculation."""
        # High safety case
        high_safety = self.code_ops._calculate_extraction_safety_score(
            dependencies=[],
            external_refs=[]
        )
        self.assertEqual(high_safety, 1.0)

        # Low safety case
        low_safety = self.code_ops._calculate_extraction_safety_score(
            dependencies=["complex_module1", "complex_module2", "complex_module3"],
            external_refs=["external_var1", "external_var2", "external_var3", "external_var4"]
        )
        self.assertLess(low_safety, 0.5)

    def test_identify_extraction_risks(self):
        """Test extraction risk identification."""
        # Safe code
        safe_code = '''
def function():
    result = calculate()
    return result
'''
        safe_risks = self.code_ops._identify_extraction_risks(safe_code)
        self.assertEqual(len(safe_risks), 0)

        # Risky code
        risky_code = '''
def function():
    global counter
    nonlocal value
    yield result
    async def nested():
        pass
'''
        risky_risks = self.code_ops._identify_extraction_risks(risky_code)
        self.assertGreater(len(risky_risks), 0)
        self.assertTrue(any("global" in risk for risk in risky_risks))
        self.assertTrue(any("nonlocal" in risk for risk in risky_risks))
        self.assertTrue(any("generator" in risk for risk in risky_risks))
        self.assertTrue(any("async" in risk for risk in risky_risks))

    def test_find_symbol_definition_in_file(self):
        """Test finding symbol definition in a file."""
        with open("test_symbols.py", "w") as f:
            f.write('''
def test_function():
    return "test"

class TestClass:
    def method(self):
        return "method"
''')

        # Find function
        symbol = self.code_ops._find_symbol_definition("test_function", "function", "test_symbols.py")
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.name, "test_function")
        self.assertEqual(symbol.symbol_type, SymbolType.FUNCTION)

        # Find class
        symbol = self.code_ops._find_symbol_definition("TestClass", "class", "test_symbols.py")
        self.assertIsNotNone(symbol)
        self.assertEqual(symbol.name, "TestClass")
        self.assertEqual(symbol.symbol_type, SymbolType.CLASS)

    def test_check_circular_dependencies(self):
        """Test circular dependency checking."""
        # Create files with circular imports
        with open("module_a.py", "w") as f:
            f.write('''
import module_b

def function_a():
    return module_b.function_b()
''')
        with open("module_b.py", "w") as f:
            f.write('''
import module_a

def function_b():
    return module_a.function_a()
''')

        circular_deps = self.code_ops._check_circular_dependencies("module_a.py", "module_b.py")
        self.assertGreater(len(circular_deps), 0)

    def test_get_file_imports(self):
        """Test getting imports from a file."""
        with open("test_imports.py", "w") as f:
            f.write('''
import os
import sys
from typing import List, Dict
from collections import defaultdict

def function():
    pass
''')

        imports = self.code_ops._get_file_imports("test_imports.py")

        self.assertIsInstance(imports, list)
        self.assertIn("os", imports)
        self.assertIn("sys", imports)
        self.assertIn("typing", imports)
        self.assertIn("collections", imports)

    def test_calculate_movement_safety_score(self):
        """Test movement safety score calculation."""
        # High safety case
        high_safety = self.code_ops._calculate_movement_safety_score(
            dependencies=[],
            circular_deps=[]
        )
        self.assertEqual(high_safety, 1.0)

        # Low safety case
        low_safety = self.code_ops._calculate_movement_safety_score(
            dependencies=["dep1", "dep2", "dep3"],
            circular_deps=["circular1", "circular2"]
        )
        self.assertLess(low_safety, 0.5)

    def test_identify_movement_risks(self):
        """Test movement risk identification."""
        # Safe symbol
        safe_symbol = SymbolInfo(
            name="normal_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.GLOBAL
        )

        with open("test.py", "w") as f:
            f.write('def normal_function():\n    return "safe"')

        safe_risks = self.code_ops._identify_movement_risks(safe_symbol, "target.py")
        self.assertEqual(len(safe_risks), 0)

        # Risky symbol
        risky_symbol = SymbolInfo(
            name="__private_method__",
            symbol_type=SymbolType.METHOD,
            file_path="test.py",
            line_number=1,
            scope=SymbolScope.CLASS
        )

        with open("test.py", "w") as f:
            f.write('''
class TestClass:
    @property
    def __private_method__(self):
        return "risky"
''')

        risky_risks = self.code_ops._identify_movement_risks(risky_symbol, "target.py")
        self.assertGreater(len(risky_risks), 0)

    def test_add_import_to_file(self):
        """Test adding import to file."""
        with open("test_imports.py", "w") as f:
            f.write('# Test file\ndef function():\n    pass')

        result = self.code_ops._add_import_to_file("test_imports.py", "import os")
        self.assertTrue(result)

        # Verify import was added
        with open("test_imports.py", "r") as f:
            content = f.read()
        self.assertIn("import os", content)

    def test_remove_import_from_file(self):
        """Test removing import from file."""
        with open("test_imports.py", "w") as f:
            f.write('import os\nimport sys\ndef function():\n    pass')

        result = self.code_ops._remove_import_from_file("test_imports.py", "import os")
        self.assertTrue(result)

        # Verify import was removed
        with open("test_imports.py", "r") as f:
            content = f.read()
        self.assertNotIn("import os", content)
        self.assertIn("import sys", content)

    def test_create_backup_file(self):
        """Test backup file creation."""
        # Create a test file
        test_file = "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        backup_path = self.code_ops._create_backup_file(test_file)

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.endswith(".backup"))

        # Verify backup content
        with open(backup_path, "r") as f:
            backup_content = f.read()
        self.assertEqual(backup_content, "test content")

    def test_execute_partial_rollback(self):
        """Test partial rollback execution."""
        # Create original file content
        with open("test_file.py", "w") as f:
            f.write("original content")

        rollback_plan = RollbackPlan(
            original_files={"test_file.py": "original content"},
            backup_files={},
            operation_log=[]
        )

        # Modify the file
        with open("test_file.py", "w") as f:
            f.write("modified content")

        # Execute partial rollback
        self.code_ops._execute_partial_rollback(rollback_plan)

        # Verify file was restored
        with open("test_file.py", "r") as f:
            content = f.read()
        self.assertEqual(content, "original content")


class TestSafeRefactoringIntegration(unittest.TestCase):
    """Test integration scenarios for safe refactoring tools."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_create_safe_refactoring_tools(self):
        """Test factory function for creating safe refactoring tools."""
        rename_tool, extraction_tool = create_safe_refactoring_tools()

        self.assertIsInstance(rename_tool, SafeSymbolRenaming)
        self.assertIsInstance(extraction_tool, CodeExtractionAndMovement)
        self.assertIsNotNone(rename_tool.search_engine)
        self.assertIsNotNone(extraction_tool.search_engine)

    def test_concurrent_operations(self):
        """Test concurrent refactoring operations."""
        # Create test files
        for i in range(3):
            with open(f"test_file_{i}.py", "w") as f:
                f.write(f'def function_{i}():\n    return {i}')

        rename_tool, extraction_tool = create_safe_refactoring_tools()

        results = []
        errors = []

        def rename_worker(i):
            try:
                # This should fail gracefully without proper mocking
                result = rename_tool.rename_symbol_safely(
                    f"function_{i}", f"renamed_function_{i}", "function"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=rename_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should handle concurrent operations gracefully
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)

    def test_rollback_plan_cleanup(self):
        """Test rollback plan cleanup functionality."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Add some rollback plans
        for i in range(5):
            rollback_plan = RollbackPlan(
                original_files={},
                backup_files={},
                operation_log=[f"Operation {i}"]
            )
            rename_tool.rollback_plans[f"operation_{i}"] = rollback_plan

        self.assertEqual(len(rename_tool.rollback_plans), 5)

        # Execute rollbacks
        for i in range(3):
            result = rename_tool.execute_rollback(f"operation_{i}")
            self.assertEqual(result["status"], RefactoringResult.SUCCESS.value)

        # Should have removed executed rollbacks
        self.assertEqual(len(rename_tool.rollback_plans), 2)

    def test_error_scenarios(self):
        """Test various error scenarios."""
        rename_tool, extraction_tool = create_safe_refactoring_tools()

        # Test invalid file paths
        result = rename_tool.rename_symbol_safely(
            "test", "new_test", "function", None, "/invalid/path"
        )
        self.assertEqual(result["status"], RefactoringResult.FAILED.value)

        # Test empty symbol names
        result = rename_tool.rename_symbol_safely(
            "", "new_name", "function"
        )
        self.assertEqual(result["status"], RefactoringResult.FAILED.value)

        # Test invalid extraction ranges
        result = extraction_tool.extract_function_safely(
            source_range=(-1, 10),
            function_name="test",
            target_file="target.py",
            source_file="source.py"
        )
        self.assertEqual(result["status"], RefactoringResult.FAILED.value)

    def test_memory_cleanup(self):
        """Test memory cleanup for rollback plans."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Add many rollback plans
        for i in range(100):
            rollback_plan = RollbackPlan(
                original_files={},
                backup_files={},
                operation_log=[]
            )
            rename_tool.rollback_plans[f"op_{i}"] = rollback_plan

        self.assertEqual(len(rename_tool.rollback_plans), 100)

        # Clean up some plans
        for i in range(0, 100, 2):  # Remove every other plan
            if f"op_{i}" in rename_tool.rollback_plans:
                del rename_tool.rollback_plans[f"op_{i}"]

        self.assertEqual(len(rename_tool.rollback_plans), 50)


class TestSafeRefactoringEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for safe refactoring."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def test_empty_files(self):
        """Test refactoring operations on empty files."""
        # Create empty file
        with open("empty.py", "w") as f:
            f.write("")

        rename_tool, extraction_tool = create_safe_refactoring_tools()

        # Should handle empty files gracefully
        result = rename_tool.rename_symbol_safely(
            "nonexistent", "new_name", "function"
        )
        self.assertEqual(result["status"], RefactoringResult.FAILED.value)

    def test_large_files(self):
        """Test refactoring operations on large files."""
        # Create a large file
        with open("large_file.py", "w") as f:
            f.write("# Large file\n")
            for i in range(1000):
                f.write(f'def function_{i}():\n    return {i}\n\n')

        rename_tool, _ = create_safe_refactoring_tools()

        # Should handle large files (operation will fail but gracefully)
        result = rename_tool.rename_symbol_safely(
            "function_500", "renamed_function_500", "function"
        )
        # Should not crash, even if it fails
        self.assertIn("status", result)

    def test_unicode_content(self):
        """Test refactoring operations with unicode content."""
        with open("unicode_file.py", "w", encoding="utf-8") as f:
            f.write('''
def 测试函数():
    """Test function with unicode name."""
    return "你好，世界！ 🌍"

def hello_function():
    """Regular function."""
    return 测试函数()
''')

        rename_tool, _ = create_safe_refactoring_tools()

        # Should handle unicode content
        result = rename_tool.rename_symbol_safely(
            "测试函数", "new_test_function", "function"
        )
        # Should not crash
        self.assertIn("status", result)

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # Create read-only file
        with open("readonly.py", "w") as f:
            f.write("def test_function():\n    return 'test'")

        # Make file read-only
        os.chmod("readonly.py", 0o444)

        rename_tool, _ = create_safe_refactoring_tools()

        try:
            # Should handle permission errors gracefully
            result = rename_tool.rename_symbol_safely(
                "test_function", "new_function", "function"
            )
            self.assertIn("status", result)
        finally:
            # Restore permissions for cleanup
            os.chmod("readonly.py", 0o666)

    def test_concurrent_file_access(self):
        """Test concurrent file access scenarios."""
        # Create test file
        with open("concurrent_test.py", "w") as f:
            f.write("def concurrent_function():\n    return 'concurrent'")

        rename_tool, extraction_tool = create_safe_refactoring_tools()

        results = []

        def file_worker(worker_id):
            try:
                result = rename_tool.rename_symbol_safely(
                    "concurrent_function", f"renamed_function_{worker_id}", "function"
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        # Start multiple threads accessing the same file
        threads = []
        for i in range(5):
            thread = threading.Thread(target=file_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should handle concurrent access
        self.assertEqual(len(results), 5)

    def test_rollback_with_missing_backups(self):
        """Test rollback scenarios with missing backup files."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Create rollback plan with missing backup files
        rollback_plan = RollbackPlan(
            original_files={"test.py": "original content"},
            backup_files={"test.py": "/tmp/nonexistent_backup.backup"},
            operation_log=["Backup failed"]
        )

        operation_id = "test_operation"
        rename_tool.rollback_plans[operation_id] = rollback_plan

        # Should handle missing backup files gracefully
        result = rename_tool.execute_rollback(operation_id)
        self.assertIn("status", result)

    def test_extremely_long_symbol_names(self):
        """Test handling of extremely long symbol names."""
        long_name = "a" * 1000  # 1000 character name

        rename_tool, _ = create_safe_refactoring_tools()

        # Should handle extremely long names
        result = rename_tool.rename_symbol_safely(
            long_name, "new_name", "function"
        )
        self.assertIn("status", result)

    def test_nested_directory_structures(self):
        """Test refactoring in nested directory structures."""
        # Create nested directory structure
        nested_path = "deep/nested/directory/structure"
        os.makedirs(nested_path, exist_ok=True)

        # Create file in nested directory
        with open(f"{nested_path}/nested_file.py", "w") as f:
            f.write("def nested_function():\n    return 'nested'")

        rename_tool, extraction_tool = create_safe_refactoring_tools()

        # Should handle nested paths
        result = rename_tool.rename_symbol_safely(
            "nested_function", "renamed_nested_function", "function",
            None, nested_path
        )
        self.assertIn("status", result)

    def test_backup_file_cleanup(self):
        """Test cleanup of backup files."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Create test file
        with open("cleanup_test.py", "w") as f:
            f.write("def cleanup_function():\n    return 'cleanup'")

        # Create backup through the normal flow
        backup_path = rename_tool._create_backup_file("cleanup_test.py")
        self.assertTrue(os.path.exists(backup_path))

        # Simulate successful operation and cleanup
        if os.path.exists(backup_path):
            os.remove(backup_path)

        # Backup should be cleaned up
        self.assertFalse(os.path.exists(backup_path))


if __name__ == "__main__":
    unittest.main()
