"""
Performance testing framework for FastApply MCP Server

This module provides comprehensive performance testing capabilities for
ripgrep integration, AST-grep operations, and MCP tools.
"""

import os
import tempfile
import typing
import unittest
from typing import Any, Dict

from . import PerformanceBaseline, PerformanceMonitor, PerformanceResult, compare_with_baseline, load_baselines


class RipgrepPerformanceTester:
    """Performance tester for ripgrep operations"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.baselines = load_baselines()

    def setup_test_environment(self) -> str:
        """Create test files for performance testing"""
        temp_dir = tempfile.mkdtemp()

        # Create test files with various content
        test_files = [
            ("simple.py", "print('Hello World')\ndef test_func():\n    return True\n"),
            (
                "complex.py",
                """
import asyncio
import json
from typing import Dict, List, Optional

class ComplexClass:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []

    async def process_data(self, items: List[str]) -> Optional[Dict]:
        if not items:
            return None
        processed = [self._process_item(item) for item in items]
        return {"results": processed}

    def _process_item(self, item: str) -> Dict:
        return {"item": item, "length": len(item)}
""",
            ),
            ("large_file.py", "x = " + " ".join([str(i) for i in range(1000)])),
        ]

        for filename, content in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        return temp_dir

    def test_simple_search_performance(self) -> PerformanceResult:
        """Test performance of simple ripgrep search"""
        temp_dir = self.setup_test_environment()

        try:
            self.monitor.start()

            # Simple search operation
            search_pattern = "print"
            self._run_ripgrep_search(temp_dir, search_pattern)
            performance_result = self.monitor.stop()
            performance_result.operation = "ripgrep_simple_search"
            performance_result = typing.cast(PerformanceResult, performance_result)

            # Compare with baseline
            baseline = self.baselines.get("ripgrep_simple_search")
            if baseline:
                comparison = compare_with_baseline(performance_result, baseline)
                performance_result.success = bool(comparison["overall_passed"])

            return performance_result

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_complex_pattern_performance(self) -> PerformanceResult:
        """Test performance of complex pattern matching"""
        temp_dir = self.setup_test_environment()

        try:
            self.monitor.start()

            # Complex pattern search
            search_pattern = r"class\s+\w+.*:|def\s+\w+.*\(|import\s+\w+"
            self._run_ripgrep_search(temp_dir, search_pattern)
            performance_result = self.monitor.stop()
            performance_result.operation = "ripgrep_complex_pattern"
            performance_result = typing.cast(PerformanceResult, performance_result)

            # Compare with baseline
            baseline = self.baselines.get("ripgrep_complex_pattern")
            if baseline:
                comparison = compare_with_baseline(performance_result, baseline)
                performance_result.success = bool(comparison["overall_passed"])

            return performance_result

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_ripgrep_search(self, directory: str, pattern: str) -> Dict[str, Any]:
        """Execute ripgrep search and return results"""
        import subprocess

        cmd = ["rg", pattern, directory]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "success": result.returncode == 0,
                "matches": result.stdout.count("\n") if result.stdout else 0,
                "output": result.stdout,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "timeout"}
        except FileNotFoundError:
            return {"success": False, "error": "ripgrep not found"}


class TestRipgrepPerformance(unittest.TestCase):
    """Test cases for ripgrep performance testing"""

    def setUp(self):
        """Set up test environment"""
        self.tester = RipgrepPerformanceTester()

    def test_simple_search_performance(self):
        """Test simple search performance"""
        result = self.tester.test_simple_search_performance()
        self.assertIsInstance(result, PerformanceResult)
        self.assertGreater(result.duration, 0)
        self.assertGreater(result.memory_usage, 0)

    def test_complex_pattern_performance(self):
        """Test complex pattern performance"""
        result = self.tester.test_complex_pattern_performance()
        self.assertIsInstance(result, PerformanceResult)
        self.assertGreater(result.duration, 0)
        self.assertGreater(result.memory_usage, 0)

    def test_baseline_comparison(self):
        """Test baseline comparison functionality"""
        result = self.tester.test_simple_search_performance()
        baseline = PerformanceBaseline(operation="test_operation", max_duration=10.0, max_memory=1000.0, max_cpu=100.0)

        comparison = compare_with_baseline(result, baseline)
        self.assertIn("duration_passed", comparison)
        self.assertIn("memory_passed", comparison)
        self.assertIn("cpu_passed", comparison)
        self.assertIn("overall_passed", comparison)


if __name__ == "__main__":
    unittest.main()
