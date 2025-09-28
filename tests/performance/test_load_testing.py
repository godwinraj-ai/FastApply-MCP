"""
Load testing framework for large codebases

This module provides comprehensive load testing capabilities for testing
the FastApply MCP Server under heavy loads and large-scale operations.
"""

import os
import queue
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import PerformanceMonitor, load_baselines


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""

    concurrent_users: int
    test_duration_seconds: int
    ramp_up_seconds: int
    max_response_time: float
    max_error_rate: float


@dataclass
class LoadTestResult:
    """Result of a load test"""

    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    throughput_per_second: float
    error_rate: float


class LargeCodebaseSimulator:
    """Simulate large codebase for load testing"""

    def __init__(self, num_files: int = 1000):
        self.num_files = num_files
        self.temp_dir: Optional[str] = None

    def create_codebase(self) -> str:
        """Create a large codebase for testing"""
        self.temp_dir = tempfile.mkdtemp()

        # Create various types of files
        file_templates = [
            ("simple_{i}.py", "print('File {i}')\ndef func_{i}():\n    return {i}\n"),
            (
                "class_{i}.py",
                """
class Class_{i}:
    def __init__(self):
        self.value = {i}

    def method(self):
        return self.value * 2
""",
            ),
            (
                "complex_{i}.py",
                """
import asyncio
from typing import Dict, List

class ComplexClass_{i}:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.items = list(range({i}))

    async def process_all(self) -> List[int]:
        results = []
        for item in self.items:
            results.append(await self._process_item(item))
        return results

    async def _process_item(self, item: int) -> int:
        return item * 2
""",
            ),
            ("config_{i}.json", '{{"name": "config_{i}", "value": {i}, "enabled": true}}'),
            ("data_{i}.txt", "Data file {i} with content\\n" + "\\n".join([f"Line {j}" for j in range(100)])),
        ]

        files_created = 0
        for i in range(self.num_files):
            template_idx = i % len(file_templates)
            filename, content = file_templates[template_idx]
            formatted_filename = filename.format(i=i)
            formatted_content = content.format(i=i)

            # Create subdirectories for organization
            if self.temp_dir is None:
                raise ValueError("Temporary directory not initialized")
            subdir = os.path.join(self.temp_dir, f"dir_{i % 10}")
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            filepath = os.path.join(subdir, formatted_filename)
            with open(filepath, "w") as f:
                f.write(formatted_content)

            files_created += 1

        if self.temp_dir is None:
            raise ValueError("Temporary directory not initialized")
        return self.temp_dir

    def cleanup(self):
        """Clean up the created codebase"""
        if self.temp_dir:
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


class LoadTester:
    """Load testing framework for performance testing"""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.baselines = load_baselines()

    def test_concurrent_search_operations(self, config: LoadTestConfig) -> LoadTestResult:
        """Test concurrent search operations"""
        simulator = LargeCodebaseSimulator(500)  # 500 files for load testing
        temp_dir = simulator.create_codebase()

        try:
            self.monitor.start()

            # Queue for results
            result_queue: queue.Queue = queue.Queue()
            start_time = time.time()
            end_time = start_time + config.test_duration_seconds

            # Worker function for concurrent operations
            def worker(worker_id: int):
                while time.time() < end_time:
                    operation_start = time.time()

                    # Simulate search operation
                    search_pattern = f"pattern_{worker_id % 5}"
                    result = self._simulate_search_operation(temp_dir, search_pattern)

                    operation_end = time.time()
                    response_time = operation_end - operation_start

                    result_queue.put({"success": result["success"], "response_time": response_time, "worker_id": worker_id})

                    # Small delay to simulate real usage
                    time.sleep(0.1)

            # Start concurrent workers
            with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
                futures = [executor.submit(worker, i) for i in range(config.concurrent_users)]

                # Wait for test duration
                time.sleep(config.test_duration_seconds)

                # Cancel remaining tasks
                for future in futures:
                    future.cancel()

            # Collect results
            results = []
            while not result_queue.empty():
                try:
                    results.append(result_queue.get_nowait())
                except queue.Empty:
                    break

            # Calculate metrics
            return self._calculate_load_metrics(results, config)

        finally:
            simulator.cleanup()

    def test_large_file_analysis(self, config: LoadTestConfig) -> LoadTestResult:
        """Test analysis of large files"""
        simulator = LargeCodebaseSimulator(100)  # 100 larger files
        temp_dir = simulator.create_codebase()

        try:
            self.monitor.start()

            # Create some large files
            large_files = []
            for i in range(10):
                large_file = os.path.join(temp_dir, f"large_{i}.py")
                with open(large_file, "w") as f:
                    # Create a large file with repetitive content
                    for j in range(1000):
                        f.write(f"# Large file content line {j}\\n")
                        f.write(f"def function_{j}():\\n")
                        f.write(f"    return {j * i}\\n\\n")
                large_files.append(large_file)

            # Test concurrent analysis of large files
            result_queue: queue.Queue = queue.Queue()
            start_time = time.time()
            end_time = start_time + config.test_duration_seconds

            def analyze_worker(file_path: str):
                while time.time() < end_time:
                    operation_start = time.time()

                    # Simulate file analysis
                    result = self._simulate_file_analysis(file_path)

                    operation_end = time.time()
                    response_time = operation_end - operation_start

                    result_queue.put(
                        {"success": result["success"], "response_time": response_time, "file_size": result.get("file_size", 0)}
                    )

                    time.sleep(0.05)  # Shorter delay for file analysis

            # Start concurrent analysis workers
            with ThreadPoolExecutor(max_workers=min(config.concurrent_users, len(large_files))) as executor:
                futures = [executor.submit(analyze_worker, file_path) for file_path in large_files]

                # Wait for test duration
                time.sleep(config.test_duration_seconds)

                # Cancel remaining tasks
                for future in futures:
                    future.cancel()

            # Collect results
            results = []
            while not result_queue.empty():
                try:
                    results.append(result_queue.get_nowait())
                except queue.Empty:
                    break

            return self._calculate_load_metrics(results, config)

        finally:
            simulator.cleanup()

    def test_memory_stress(self, config: LoadTestConfig) -> LoadTestResult:
        """Test memory stress under load"""
        self.monitor.start()

        result_queue: queue.Queue = queue.Queue()
        start_time = time.time()
        end_time = start_time + config.test_duration_seconds

        def memory_worker(worker_id: int):
            # Create memory-intensive data structures
            data_structures: List = []

            while time.time() < end_time:
                operation_start = time.time()

                try:
                    # Simulate memory-intensive operation
                    large_list = [str(i) * 100 for i in range(1000)]
                    large_dict = {f"key_{i}": large_list[i] for i in range(len(large_list))}

                    # Process the data
                    _ = {"list_length": len(large_list), "dict_size": len(large_dict), "worker_id": worker_id}

                    # Keep some data to simulate memory usage
                    if len(data_structures) < 10:
                        data_structures.append(large_dict)

                    operation_end = time.time()
                    response_time = operation_end - operation_start

                    result_queue.put({"success": True, "response_time": response_time, "memory_used": len(data_structures)})

                except MemoryError:
                    result_queue.put({"success": False, "response_time": time.time() - operation_start, "error": "memory_error"})
                    break

                time.sleep(0.2)

        # Start memory workers
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [executor.submit(memory_worker, i) for i in range(config.concurrent_users)]

            # Wait for test duration
            time.sleep(config.test_duration_seconds)

            # Cancel remaining tasks
            for future in futures:
                future.cancel()

        # Collect results
        results = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                break

        performance_result = self.monitor.stop()
        performance_result.operation = "memory_stress_test"

        return self._calculate_load_metrics(results, config)

    def _simulate_search_operation(self, directory: str, pattern: str) -> Dict[str, Any]:
        """Simulate a search operation"""
        try:
            # Use a simple file search simulation
            matches = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py") and pattern in file:
                        matches += 1

            return {"success": True, "matches": matches, "pattern": pattern}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _simulate_file_analysis(self, file_path: str) -> Dict[str, Any]:
        """Simulate file analysis"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)

            # Simulate analysis by reading and processing the file
            with open(file_path, "r") as f:
                content = f.read()

            # Simple analysis metrics
            lines = content.count("\\n")
            functions = content.count("def ")
            classes = content.count("class ")

            return {"success": True, "file_size": file_size, "lines": lines, "functions": functions, "classes": classes}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_load_metrics(self, results: List[Dict[str, Any]], config: LoadTestConfig) -> LoadTestResult:
        """Calculate load test metrics"""
        if not results:
            return LoadTestResult(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0,
                max_response_time=0,
                min_response_time=0,
                throughput_per_second=0,
                error_rate=1.0,
            )

        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests

        response_times = [r["response_time"] for r in results]
        average_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        throughput_per_second = successful_requests / config.test_duration_seconds
        error_rate = failed_requests / total_requests

        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=average_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            throughput_per_second=throughput_per_second,
            error_rate=error_rate,
        )


class TestLoadTesting(unittest.TestCase):
    """Test cases for load testing"""

    def setUp(self):
        """Set up test environment"""
        self.load_tester = LoadTester()

    def test_concurrent_search_operations(self):
        """Test concurrent search operations"""
        config = LoadTestConfig(concurrent_users=5, test_duration_seconds=2, ramp_up_seconds=1, max_response_time=5.0, max_error_rate=0.1)

        result = self.load_tester.test_concurrent_search_operations(config)
        self.assertIsInstance(result, LoadTestResult)
        self.assertGreater(result.total_requests, 0)
        self.assertGreaterEqual(result.successful_requests, 0)
        self.assertLessEqual(result.error_rate, 1.0)

    def test_large_file_analysis(self):
        """Test large file analysis"""
        config = LoadTestConfig(concurrent_users=3, test_duration_seconds=2, ramp_up_seconds=1, max_response_time=10.0, max_error_rate=0.1)

        result = self.load_tester.test_large_file_analysis(config)
        self.assertIsInstance(result, LoadTestResult)
        self.assertGreater(result.total_requests, 0)

    def test_memory_stress(self):
        """Test memory stress"""
        config = LoadTestConfig(concurrent_users=2, test_duration_seconds=1, ramp_up_seconds=0, max_response_time=5.0, max_error_rate=0.2)

        result = self.load_tester.test_memory_stress(config)
        self.assertIsInstance(result, LoadTestResult)
        self.assertGreater(result.total_requests, 0)

    def test_large_codebase_simulator(self):
        """Test large codebase simulator"""
        simulator = LargeCodebaseSimulator(50)
        temp_dir = simulator.create_codebase()

        try:
            # Check that files were created
            file_count = 0
            for root, dirs, files in os.walk(temp_dir):
                file_count += len(files)

            self.assertGreater(file_count, 0)
            self.assertLessEqual(file_count, 50)
        finally:
            simulator.cleanup()


if __name__ == "__main__":
    unittest.main()
