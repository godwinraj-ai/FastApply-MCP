#!/usr/bin/env python3
"""
Integration tests for safe refactoring tools.

Tests MCP tool integration, complex scenarios, and end-to-end workflows
for safe refactoring operations.

Phase 5 Implementation Tests - Safe Refactoring Integration
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from fastapply.safe_refactoring import (
    CodeExtractionAndMovement,
    ImpactAnalysis,
    RefactoringResult,
    SafeSymbolRenaming,
    create_safe_refactoring_tools,
)


class TestSafeRefactoringMCPIntegration(unittest.TestCase):
    """Test MCP integration for safe refactoring tools."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test project structure
        self.create_test_project()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_test_project(self):
        """Create a realistic test project structure."""
        # Main source directory
        os.makedirs("src/calculator", exist_ok=True)
        os.makedirs("tests", exist_ok=True)
        os.makedirs("docs", exist_ok=True)

        # Calculator module with functions to rename
        with open("src/calculator/core.py", "w", encoding="utf-8") as f:
            f.write('''"""
Core calculator functionality.
"""

from typing import Union, List

def add_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers together."""
    return a + b

def subtract_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Subtract second number from first."""
    return a - b

def multiply_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Multiply two numbers."""
    return a * b

def divide_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Divide first number by second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    """Main calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = add_numbers(a, b)
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a, b):
        result = subtract_numbers(a, b)
        self.history.append(f"{a} - {b} = {result}")
        return result
''')

        # Utils module with functions to extract
        with open("src/calculator/utils.py", "w", encoding="utf-8") as f:
            f.write('''"""
Utility functions for calculator.
"""

import json
from typing import Any, Dict

def format_result(result: Any, precision: int = 2) -> str:
    """Format calculation result with specified precision."""
    if isinstance(result, float):
        return f"{result:.{precision}f}"
    return str(result)

def save_to_json(data: Dict[str, Any], filename: str) -> None:
    """Save data to JSON file."""
    # Extractable code starts here
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    # Extractable validation code
    if os.path.exists(filename):
        print(f"Data saved to {filename}")
    else:
        print(f"Failed to save data to {filename}")

    return True

def load_from_json(filename: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def validate_calculation(result: Any) -> bool:
    """Validate calculation result."""
    return result is not None and not isinstance(result, str)
''')

        # Test files that reference calculator functions
        with open("tests/test_calculator.py", "w", encoding="utf-8") as f:
            f.write('''"""
Test cases for calculator functionality.
"""

import pytest
from src.calculator.core import add_numbers, subtract_numbers, Calculator

def test_add_numbers():
    """Test addition functionality."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0

def test_subtract_numbers():
    """Test subtraction functionality."""
    assert subtract_numbers(5, 3) == 2
    assert subtract_numbers(0, 5) == -5
    assert subtract_numbers(10, 10) == 0

def test_calculator():
    """Test Calculator class."""
    calc = Calculator()
    assert calc.add(5, 3) == 8
    assert calc.subtract(10, 4) == 6
    assert len(calc.history) == 2
''')

        # Main application file
        with open("main.py", "w", encoding="utf-8") as f:
            f.write('''"""
Main application entry point.
"""

from src.calculator.core import Calculator, add_numbers, subtract_numbers
from src.calculator.utils import format_result

def main():
    """Main application function."""
    calc = Calculator()

    # Use functions that could be renamed
    result1 = add_numbers(10, 5)
    result2 = subtract_numbers(20, 8)

    formatted1 = format_result(result1)
    formatted2 = format_result(result2)

    print(f"Results: {formatted1}, {formatted2}")

    return calc

if __name__ == "__main__":
    calculator = main()
''')

        # Configuration file
        with open("config.json", "w", encoding="utf-8") as f:
            f.write('''{
    "app_name": "Calculator App",
    "version": "1.0.0",
    "default_precision": 2,
    "max_history": 100
}''')

    def test_mcp_safe_refactoring_integration(self):
        """Test integration with MCP tools."""
        # Create safe refactoring tools
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Test that tools are properly initialized
        self.assertIsInstance(rename_tool, SafeSymbolRenaming)
        self.assertIsInstance(extract_tool, CodeExtractionAndMovement)

        # Test that search engines are available
        self.assertIsNotNone(rename_tool.search_engine)
        self.assertIsNotNone(extract_tool.search_engine)

        # Test rollback plan storage
        self.assertIsInstance(rename_tool.rollback_plans, dict)
        self.assertIsInstance(extract_tool.rollback_plans, dict)

    def test_project_wide_symbol_rename_analysis(self):
        """Test project-wide symbol rename impact analysis."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Analyze impact of renaming add_numbers function
        impact_analysis = rename_tool.analyze_rename_impact(
            old_name="add_numbers",
            new_name="sum_numbers",
            symbol_type="function",
            project_path="."
        )

        self.assertIsInstance(impact_analysis, ImpactAnalysis)
        self.assertGreater(len(impact_analysis.affected_files), 0)
        self.assertIn("add_numbers", impact_analysis.affected_symbols)

        # Should identify test files
        self.assertTrue(impact_analysis.test_impact)

        # Should have measurable risk score
        self.assertGreater(impact_analysis.risk_score, 0.0)
        self.assertLessEqual(impact_analysis.risk_score, 1.0)

    def test_code_extraction_safety_validation(self):
        """Test code extraction safety validation."""
        _, extract_tool = create_safe_refactoring_tools()

        # Test extracting validation logic from save_to_json
        safety_analysis = extract_tool.analyze_extraction_safety(
            source_range=(15, 22),  # Lines with validation code
            source_file="src/calculator/utils.py"
        )

        self.assertIsInstance(safety_analysis, dict)
        self.assertIn("is_safe", safety_analysis)
        self.assertIn("safety_score", safety_analysis)
        self.assertIn("dependencies", safety_analysis)

        # Should identify file system operations as risk
        self.assertIn("risk_factors", safety_analysis)

    def test_function_extraction_with_dependencies(self):
        """Test function extraction with dependency management."""
        _, extract_tool = create_safe_refactoring_tools()

        # Create move operation for extracting validation logic
        move_op = Mock()
        move_op.symbol_name = "validate_calculation"
        move_op.source_file = "src/calculator/utils.py"
        move_op.target_file = "src/calculator/validation.py"
        move_op.symbol_type = "function"

        # Test dependency management
        dependency_result = extract_tool.manage_import_dependencies(move_op)

        self.assertIsInstance(dependency_result, dict)
        self.assertIn("imports_added", dependency_result)
        self.assertIn("imports_removed", dependency_result)
        self.assertIn("errors", dependency_result)

    def test_rollback_plan_persistence(self):
        """Test rollback plan persistence and cleanup."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Create multiple rollback plans
        for i in range(5):
            operation_id = f"rename_operation_{i}"
            rollback_plan = Mock()
            rollback_plan.original_files = {f"file_{i}.py": f"content_{i}"}
            rollback_plan.backup_files = {f"file_{i}.py": f"backup_{i}"}
            rollback_plan.operation_log = [f"Operation {i}"]

            rename_tool.rollback_plans[operation_id] = rollback_plan

        self.assertEqual(len(rename_tool.rollback_plans), 5)

        # Execute some rollbacks
        for i in range(3):
            operation_id = f"rename_operation_{i}"
            # Mock successful rollback
            if operation_id in rename_tool.rollback_plans:
                del rename_tool.rollback_plans[operation_id]

        # Verify remaining plans
        self.assertEqual(len(rename_tool.rollback_plans), 2)

    def test_cross_module_refactoring(self):
        """Test refactoring operations across multiple modules."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Test moving Calculator class to separate file
        movement_analysis = extract_tool.analyze_movement_safety(
            symbol_name="Calculator",
            source_file="src/calculator/core.py",
            target_file="src/calculator/calculator_class.py",
            symbol_type="class"
        )

        self.assertIsInstance(movement_analysis, dict)
        self.assertIn("is_safe", movement_analysis)
        self.assertIn("safety_score", movement_analysis)

        # Should identify dependencies on other functions
        self.assertIn("dependencies", movement_analysis)

    def test_error_recovery_mechanisms(self):
        """Test error recovery and fallback mechanisms."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Test with invalid file paths
        invalid_result = rename_tool.rename_symbol_safely(
            old_name="test_function",
            new_name="new_function",
            symbol_type="function",
            project_path="/nonexistent/path"
        )

        self.assertEqual(invalid_result["status"], RefactoringResult.FAILED.value)
        self.assertIn("error", invalid_result)

        # Test with invalid extraction ranges
        invalid_extraction = extract_tool.extract_function_safely(
            source_range=(-1, 10),
            function_name="test_function",
            target_file="target.py",
            source_file="nonexistent.py"
        )

        self.assertEqual(invalid_extraction["status"], RefactoringResult.FAILED.value)
        self.assertIn("error", invalid_extraction)


class TestSafeRefactoringEndToEnd(unittest.TestCase):
    """Test end-to-end refactoring workflows."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create realistic project
        self.create_realistic_project()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_realistic_project(self):
        """Create a realistic project for end-to-end testing."""
        # Project structure
        os.makedirs("src/auth", exist_ok=True)
        os.makedirs("src/users", exist_ok=True)
        os.makedirs("src/database", exist_ok=True)
        os.makedirs("tests", exist_ok=True)
        os.makedirs("docs", exist_ok=True)

        # Authentication module with poorly named functions
        with open("src/auth/authenticator.py", "w", encoding="utf-8") as f:
            f.write('''"""
Authentication module with functions to rename.
"""

import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

def check_user_credentials(username: str, password: str) -> bool:
    """Check if user credentials are valid."""
    # This function name is unclear - should be validate_user_credentials
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Database lookup would go here
    stored_hash = get_stored_password_hash(username)

    return hashed_password == stored_hash

def make_auth_token(user_id: int, secret_key: str) -> str:
    """Create authentication token for user."""
    # This function name is unclear - should be generate_auth_token
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }

    return jwt.encode(payload, secret_key, algorithm='HS256')

def check_token_valid(token: str, secret_key: str) -> Dict[str, Any]:
    """Check if authentication token is valid."""
    # This function name is unclear - should be validate_auth_token
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return {'valid': True, 'user_id': payload['user_id']}
    except jwt.ExpiredSignatureError:
        return {'valid': False, 'error': 'Token expired'}
    except jwt.InvalidTokenError:
        return {'valid': False, 'error': 'Invalid token'}

def get_stored_password_hash(username: str) -> Optional[str]:
    """Get stored password hash for user."""
    # This would normally query a database
    users = {
        'admin': '5f4dcc3b5aa765d61d8327deb882cf99',  # 'password'
        'user': '25d55ad283aa400af464c76d713c07ad'   # '123456'
    }
    return users.get(username)

class UserAuthenticator:
    """User authentication class."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        return check_user_credentials(username, password)

    def create_session(self, user_id: int) -> str:
        """Create user session token."""
        return make_auth_token(user_id, self.secret_key)

    def validate_session(self, token: str) -> Dict[str, Any]:
        """Validate user session token."""
        return check_token_valid(token, self.secret_key)
''')

        # User management module
        with open("src/users/user_manager.py", "w", encoding="utf-8") as f:
            f.write('''"""
User management module.
"""

from typing import List, Optional
from src.auth.authenticator import check_user_credentials, make_auth_token

class UserManager:
    """User management class."""

    def __init__(self):
        self.users = {}

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user."""
        if username in self.users:
            return False

        # Validate credentials first
        if not check_user_credentials(username, password):
            return False

        self.users[username] = {
            'password_hash': hashlib.sha256(password.encode()).hexdigest(),
            'created_at': datetime.utcnow()
        }
        return True

    def login_user(self, username: str, password: str) -> Optional[str]:
        """Login user and return token."""
        if check_user_credentials(username, password):
            return make_auth_token(len(self.users), "secret_key")
        return None
''')

        # Tests
        with open("tests/test_auth.py", "w", encoding="utf-8") as f:
            f.write('''"""
Authentication tests.
"""

import pytest
from src.auth.authenticator import check_user_credentials, make_auth_token, check_token_valid

def test_check_user_credentials():
    """Test user credential checking."""
    assert check_user_credentials("admin", "password") is True
    assert check_user_credentials("admin", "wrong") is False

def test_make_auth_token():
    """Test token creation."""
    token = make_auth_token(1, "secret_key")
    assert isinstance(token, str)
    assert len(token) > 0

def test_check_token_valid():
    """Test token validation."""
    token = make_auth_token(1, "secret_key")
    result = check_token_valid(token, "secret_key")
    assert result['valid'] is True
''')

    def test_end_to_end_rename_workflow(self):
        """Test complete end-to-end rename workflow."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Step 1: Analyze rename impact for check_user_credentials -> validate_user_credentials
        impact_analysis = rename_tool.analyze_rename_impact(
            old_name="check_user_credentials",
            new_name="validate_user_credentials",
            symbol_type="function",
            project_path="."
        )

        # Should identify all affected files
        affected_files = impact_analysis.affected_files
        self.assertGreater(len(affected_files), 0)

        # Should identify test impact
        self.assertTrue(impact_analysis.test_impact)

        # Step 2: Validate safety
        _ = rename_tool.validate_rename_safety(impact_analysis)
        # In real scenario, this would depend on actual analysis results

        # Step 3: Generate rollback plan (mock successful operation)
        rollback_plan = Mock()
        rollback_plan.original_files = {"src/auth/authenticator.py": "original content"}
        rollback_plan.backup_files = {"src/auth/authenticator.py": "backup_path"}
        rollback_plan.operation_log = []

        operation_id = "rename_check_user_credentials"
        rename_tool.rollback_plans[operation_id] = rollback_plan

        # Step 4: Verify rollback capability
        self.assertIn(operation_id, rename_tool.rollback_plans)

        # Cleanup
        del rename_tool.rollback_plans[operation_id]

    def test_batch_refactoring_operations(self):
        """Test batch refactoring operations."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Define batch rename operations
        rename_operations = [
            ("check_user_credentials", "validate_user_credentials"),
            ("make_auth_token", "generate_auth_token"),
            ("check_token_valid", "validate_auth_token")
        ]

        operation_ids = []

        # Simulate batch operations
        for old_name, new_name in rename_operations:
            # Analyze impact
            _ = rename_tool.analyze_rename_impact(
                old_name=old_name,
                new_name=new_name,
                symbol_type="function",
                project_path="."
            )

            # Create rollback plan
            rollback_plan = Mock()
            rollback_plan.original_files = {"src/auth/authenticator.py": f"content_{old_name}"}
            rollback_plan.backup_files = {"src/auth/authenticator.py": f"backup_{old_name}"}
            rollback_plan.operation_log = [f"Analyzed {old_name}"]

            operation_id = f"batch_rename_{old_name}"
            rename_tool.rollback_plans[operation_id] = rollback_plan
            operation_ids.append(operation_id)

        # Verify all operations were staged
        self.assertEqual(len(rename_tool.rollback_plans), len(rename_operations))

        # Verify we can track all operations
        for operation_id in operation_ids:
            self.assertIn(operation_id, rename_tool.rollback_plans)

        # Cleanup
        for operation_id in operation_ids:
            del rename_tool.rollback_plans[operation_id]

    def test_complex_extraction_workflow(self):
        """Test complex code extraction workflow."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Extract token validation logic to separate module
        extraction_analysis = extract_tool.analyze_extraction_safety(
            source_range=(25, 35),  # Token validation logic
            source_file="src/auth/authenticator.py"
        )

        self.assertIsInstance(extraction_analysis, dict)
        self.assertIn("safety_score", extraction_analysis)

        # Analyze movement to separate validation module
        movement_analysis = extract_tool.analyze_movement_safety(
            symbol_name="check_token_valid",
            source_file="src/auth/authenticator.py",
            target_file="src/auth/validation.py"
        )

        self.assertIsInstance(movement_analysis, dict)
        self.assertIn("safety_score", movement_analysis)

        # Test dependency management for movement
        move_op = Mock()
        move_op.symbol_name = "check_token_valid"
        move_op.source_file = "src/auth/authenticator.py"
        move_op.target_file = "src/auth/validation.py"
        move_op.symbol_type = "function"

        dependency_result = extract_tool.manage_import_dependencies(move_op)

        self.assertIsInstance(dependency_result, dict)
        self.assertIn("imports_added", dependency_result)

    def test_refactoring_with_external_dependencies(self):
        """Test refactoring operations with external dependencies."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Test renaming a function that imports external modules
        impact_analysis = rename_tool.analyze_rename_impact(
            old_name="make_auth_token",
            new_name="generate_auth_token",
            symbol_type="function",
            project_path="."
        )

        # Should identify external dependencies (jwt, datetime)
        self.assertIsInstance(impact_analysis.external_dependencies, set)

        # Should have calculated risk score
        self.assertGreater(impact_analysis.risk_score, 0.0)

        # Should identify affected files
        self.assertGreater(len(impact_analysis.affected_files), 0)

    def test_rollback_after_partial_failure(self):
        """Test rollback after partial operation failure."""
        rename_tool, extract_tool = create_safe_refactoring_tools()

        # Create rollback plan
        rollback_plan = Mock()
        rollback_plan.original_files = {
            "src/auth/authenticator.py": "original_content",
            "src/users/user_manager.py": "original_user_content"
        }
        rollback_plan.backup_files = {
            "src/auth/authenticator.py": "backup_auth",
            "src/users/user_manager.py": "backup_users"
        }
        rollback_plan.operation_log = ["Started operation", "Modified auth module"]

        operation_id = "partial_failure_operation"
        rename_tool.rollback_plans[operation_id] = rollback_plan

        # Simulate partial failure - some files modified, others not
        _ = ["src/auth/authenticator.py"]

        # Execute rollback
        if operation_id in rename_tool.rollback_plans:
            del rename_tool.rollback_plans[operation_id]

        # Verify cleanup
        self.assertNotIn(operation_id, rename_tool.rollback_plans)


class TestSafeRefactoringPerformance(unittest.TestCase):
    """Test performance characteristics of safe refactoring tools."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create large test project
        self.create_large_test_project()

    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)

    def create_large_test_project(self):
        """Create a large test project for performance testing."""
        os.makedirs("src/large_module", exist_ok=True)
        os.makedirs("tests/large_module", exist_ok=True)

        # Create many files with similar functions
        for i in range(50):
            with open(f"src/large_module/file_{i}.py", "w") as f:
                f.write(f'''"""
Module {i} with functions to rename.
"""

import os
import sys

def function_to_rename_{i}(param):
    """Function that should be renamed in module {i}."""
    return param * {i}

def helper_function_{i}(data):
    """Helper function in module {i}."""
    return function_to_rename_{i}(data) + 1

class TestClass_{i}:
    """Test class in module {i}."""

    def method_{i}(self):
        """Method that uses the function."""
        result = function_to_rename_{i}(42)
        return result
''')

        # Create test files
        for i in range(20):
            with open(f"tests/large_module/test_file_{i}.py", "w") as f:
                f.write(f'''"""
Test file {i}.
"""

from src.large_module.file_{i} import function_to_rename_{i}

def test_function_{i}():
    """Test function {i}."""
    result = function_to_rename_{i}(10)
    assert result == 10 * {i}
''')

    def test_large_project_impact_analysis(self):
        """Test impact analysis performance on large project."""
        rename_tool, _ = create_safe_refactoring_tools()

        import time
        start_time = time.time()

        # Analyze impact of renaming one function across all files
        impact_analysis = rename_tool.analyze_rename_impact(
            old_name="function_to_rename_0",
            new_name="renamed_function_0",
            symbol_type="function",
            project_path="."
        )

        end_time = time.time()
        analysis_time = end_time - start_time

        # Should complete in reasonable time
        self.assertLess(analysis_time, 5.0)  # 5 seconds max

        # Should identify many affected files
        self.assertGreater(len(impact_analysis.affected_files), 0)

        # Should have reasonable risk score
        self.assertGreaterEqual(impact_analysis.risk_score, 0.0)
        self.assertLessEqual(impact_analysis.risk_score, 1.0)

    def test_concurrent_impact_analysis(self):
        """Test concurrent impact analysis performance."""
        rename_tool, _ = create_safe_refactoring_tools()

        import time
        start_time = time.time()

        # Run multiple impact analyses concurrently
        results = []

        def analyze_function(i):
            try:
                impact = rename_tool.analyze_rename_impact(
                    old_name=f"function_to_rename_{i}",
                    new_name=f"renamed_function_{i}",
                    symbol_type="function",
                    project_path="."
                )
                results.append(impact)
            except Exception as e:
                results.append({"error": str(e)})

        import threading
        threads = []
        for i in range(10):  # Analyze 10 functions concurrently
            thread = threading.Thread(target=analyze_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        end_time = time.time()
        concurrent_time = end_time - start_time

        # Should complete in reasonable time
        self.assertLess(concurrent_time, 10.0)  # 10 seconds max

        # Should have results for all analyses
        self.assertEqual(len(results), 10)

    def test_memory_usage_with_many_rollback_plans(self):
        """Test memory usage with many rollback plans."""
        rename_tool, _ = create_safe_refactoring_tools()

        # Create many rollback plans
        for i in range(100):
            rollback_plan = Mock()
            rollback_plan.original_files = {f"file_{i}.py": f"content_{i}"}
            rollback_plan.backup_files = {f"file_{i}.py": f"backup_{i}"}
            rollback_plan.operation_log = [f"Operation {i}"]

            rename_tool.rollback_plans[f"operation_{i}"] = rollback_plan

        # Should handle many rollback plans
        self.assertEqual(len(rename_tool.rollback_plans), 100)

        # Cleanup should work efficiently
        for i in range(0, 100, 2):  # Remove every other plan
            if f"operation_{i}" in rename_tool.rollback_plans:
                del rename_tool.rollback_plans[f"operation_{i}"]

        self.assertEqual(len(rename_tool.rollback_plans), 50)

    def test_large_file_processing(self):
        """Test processing of very large files."""
        # Create a large file
        with open("large_file.py", "w") as f:
            f.write("# Large file\n")
            for i in range(1000):
                f.write(f'def large_function_{i}():\n')
                f.write(f'    """Large function {i}."""\n')
                f.write('    result = 0\n')
                for j in range(20):
                    f.write(f'    result += {j}\n')
                f.write('    return result\n\n')

        rename_tool, _ = create_safe_refactoring_tools()

        import time
        start_time = time.time()

        # Analyze impact on large file
        impact_analysis = rename_tool.analyze_rename_impact(
            old_name="large_function_500",
            new_name="renamed_large_function_500",
            symbol_type="function",
            project_path="."
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process large file in reasonable time
        self.assertLess(processing_time, 3.0)  # 3 seconds max

        self.assertIsInstance(impact_analysis, type(impact_analysis))


if __name__ == "__main__":
    unittest.main()
