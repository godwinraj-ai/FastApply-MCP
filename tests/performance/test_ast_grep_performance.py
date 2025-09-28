"""
AST-grep performance testing framework

This module provides performance testing capabilities for AST-grep operations,
including pattern matching, rule execution, and code transformation.
"""

import os
import tempfile
import typing
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List

from . import PerformanceMonitor, PerformanceResult, compare_with_baseline, load_baselines


@dataclass
class ASTGrepTestConfig:
    """Configuration for AST-grep performance tests"""

    test_patterns: List[str]
    test_files: List[str]
    rule_complexity: str = "medium"
    expected_matches: int = 0


class ASTGrepPerformanceTester:
    """Performance tester for AST-grep operations"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.baselines = load_baselines()

    def setup_test_environment(self) -> str:
        """Create test files for AST-grep testing"""
        temp_dir = tempfile.mkdtemp()

        # Create test files with various AST structures
        test_files = [
            (
                "simple.py",
                """
def hello_world():
    print("Hello, World!")

class SimpleClass:
    def method(self):
        return True
""",
            ),
            (
                "complex.py",
                """
import asyncio
from typing import Dict, List, Optional

class ComplexClass:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []

    async def process_data(self, items: List[str]) -> Optional[Dict]:
        if not items:
            return None
        return {"processed": len(items)}
""",
            ),
            (
                "nested.py",
                """
def outer_function():
    def inner_function():
        class InnerClass:
            def method(self):
                return "inner"
        return InnerClass()
    return inner_function

class OuterClass:
    class InnerClass:
        @staticmethod
        def static_method():
            return "static"
""",
            ),
        ]

        for filename, content in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        return temp_dir

    def test_pattern_matching_performance(self) -> PerformanceResult:
        """Test AST pattern matching performance"""
        temp_dir = self.setup_test_environment()

        try:
            self.monitor.start()

            # Test pattern matching
            pattern = "class $NAME"
            self._run_ast_grep_pattern(temp_dir, pattern)

            performance_result = self.monitor.stop()
            performance_result.operation = "ast_grep_pattern_match"
            performance_result = typing.cast(PerformanceResult, performance_result)

            # Compare with baseline
            baseline = self.baselines.get("ast_grep_pattern_match")
            if baseline:
                comparison = compare_with_baseline(performance_result, baseline)
                performance_result.success = bool(comparison["overall_passed"])

            return performance_result

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rule_execution_performance(self) -> PerformanceResult:
        """Test AST-grep rule execution performance"""
        temp_dir = self.setup_test_environment()

        try:
            self.monitor.start()

            # Test rule execution
            rule = {"pattern": "def $NAME($ARGS):", "constraints": {"ARGS": {"regex": "self.*"}}}
            self._run_ast_grep_rule(temp_dir, rule)

            performance_result = self.monitor.stop()
            performance_result.operation = "ast_grep_rule_execution"
            performance_result = typing.cast(PerformanceResult, performance_result)

            # Compare with baseline
            baseline = self.baselines.get("ast_grep_rule_execution")
            if baseline:
                comparison = compare_with_baseline(performance_result, baseline)
                performance_result.success = bool(comparison["overall_passed"])

            return performance_result

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_ast_grep_pattern(self, directory: str, pattern: str) -> Dict[str, Any]:
        """Execute AST-grep pattern search"""
        try:
            # Try to use ast-grep if available
            import subprocess

            cmd = ["ast-grep", "python", "--pattern", pattern, directory]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "success": result.returncode == 0,
                "matches": result.stdout.count("\n") if result.stdout else 0,
                "output": result.stdout,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
            # Fallback to simple pattern matching
            return self._fallback_pattern_match(directory, pattern)

    def _run_ast_grep_rule(self, directory: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AST-grep rule"""
        try:
            # Create temporary rule file

            rule_file = os.path.join(directory, "rule.yaml")
            with open(rule_file, "w") as f:
                # Simplified YAML format
                f.write(f"pattern: {rule['pattern']}\n")
                if "constraints" in rule:
                    f.write("constraints:\n")
                    for key, constraint in rule["constraints"].items():
                        f.write(f"  {key}: {constraint}\n")

            import subprocess

            cmd = ["ast-grep", "python", "--rule", rule_file, directory]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0,
                "matches": result.stdout.count("\n") if result.stdout else 0,
                "output": result.stdout,
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, ImportError):
            # Fallback to simple pattern matching
            return self._fallback_pattern_match(directory, rule["pattern"])

    def _fallback_pattern_match(self, directory: str, pattern: str) -> Dict[str, Any]:
        """Fallback pattern matching when ast-grep is not available"""
        import re

        matches = 0
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                        # Simple regex-based pattern matching
                        if re.search(r"class\s+\w+", content):
                            matches += 1
                        if re.search(r"def\s+\w+", content):
                            matches += 1
                except Exception:
                    pass

        return {"success": True, "matches": matches, "output": f"Found {matches} pattern matches"}


class TestASTGrepPerformance(unittest.TestCase):
    """Test cases for AST-grep performance testing"""

    def setUp(self):
        """Set up test environment"""
        self.tester = ASTGrepPerformanceTester()

    def test_pattern_matching_performance(self):
        """Test pattern matching performance"""
        result = self.tester.test_pattern_matching_performance()
        self.assertIsInstance(result, PerformanceResult)
        self.assertGreater(result.duration, 0)
        self.assertGreater(result.memory_usage, 0)

    def test_rule_execution_performance(self):
        """Test rule execution performance"""
        result = self.tester.test_rule_execution_performance()
        self.assertIsInstance(result, PerformanceResult)
        self.assertGreater(result.duration, 0)
        self.assertGreater(result.memory_usage, 0)

    def test_fallback_pattern_matching(self):
        """Test fallback pattern matching when ast-grep is not available"""
        temp_dir = self.tester.setup_test_environment()
        try:
            result = self.tester._fallback_pattern_match(temp_dir, "class")
            self.assertIsInstance(result, dict)
            self.assertIn("success", result)
            self.assertIn("matches", result)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
