#!/usr/bin/env python3
"""
Integration tests for security and quality analysis MCP tools.

Tests the integration of security and quality analysis tools with the main MCP server
and other FastApply components, including real-world scenarios and error handling.

Phase 7 Implementation Tests - Security & Quality Integration
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapply.main import call_tool


class TestSecurityQualityMCPIntegration(unittest.TestCase):
    """Test integration of security and quality MCP tools."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_project_with_vulnerabilities(self):
        """Create a test project with security vulnerabilities and quality issues."""
        project_structure = {
            "main.py": """
import os
import sqlite3

# Hardcoded secrets (security issue)
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "supersecret123"

def get_user_data(user_id):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_id
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def render_user_input(user_input):
    # XSS vulnerability
    return f"<div>{user_input}</div>"

def read_config_file(filename):
    # Path traversal vulnerability
    config_path = "/app/config/" + filename
    with open(config_path, 'r') as f:
        return f.read()

# Complex function with quality issues
def process_complex_data(data):
    result = []
    for item in data:
        if item is not None:
            if isinstance(item, dict):
                if item.get('active'):
                    if item.get('verified'):
                        if item.get('score', 0) > 50:
                            if item.get('category') == 'premium':
                                result.append(item)
    return result

def main():
    user_id = input("Enter user ID: ")
    user_data = get_user_data(user_id)
    print(f"User data: {user_data}")

    user_input = input("Enter message: ")
    rendered = render_user_input(user_input)
    print(f"Rendered: {rendered}")

if __name__ == "__main__":
    main()
""",
            "requirements.txt": """
requests==2.28.0  # Known vulnerable version
django==4.2.0
flask==2.0.0      # Known vulnerable version
sqlalchemy==1.4.0
""",
            "config.py": """
# Configuration file with security issues
DEBUG = True  # Security risk in production
SECRET_KEY = "insecure-secret-key"  # Hardcoded secret
ALLOWED_HOSTS = ['*']  # Insecure host configuration

DATABASE_URL = "sqlite:///database.db"
API_VERSION = "v1"
""",
            "utils.py": """
import hashlib
import base64

def weak_hash_password(password):
    # Weak password hashing
    return hashlib.md5(password.encode()).hexdigest()

def insecure_encode(data):
    # Insecure encoding
    return base64.b64encode(data.encode()).decode()

def duplicate_function(items):
    # Code duplication
    total = 0
    for item in items:
        total += item['price'] * item['quantity']
    return total

def another_duplicate_function(items):
    # Duplicate code
    total = 0
    for item in items:
        total += item['price'] * item['quantity']
    return total
""",
        }

        for file_name, content in project_structure.items():
            file_path = Path(self.temp_dir) / file_name
            file_path.write_text(content.strip())

        return self.temp_dir

    async def test_security_scan_comprehensive_mcp_tool(self):
        """Test security_scan_comprehensive MCP tool."""
        project_path = self.create_test_project_with_vulnerabilities()

        arguments = {
            "project_path": project_path,
            "scan_types": ["pattern", "dependencies", "configuration"],
            "output_format": "json"
        }

        result = await call_tool("security_scan_comprehensive", arguments)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # Should return one text response

        # The result should contain JSON with security analysis data
        response_text = result[0]["text"]
        self.assertIn("security_score", response_text)
        self.assertIn("total_vulnerabilities", response_text)
        self.assertIn("vulnerabilities", response_text)

    async def test_quality_assessment_comprehensive_mcp_tool(self):
        """Test quality_assessment_comprehensive MCP tool."""
        project_path = self.create_test_project_with_vulnerabilities()

        arguments = {
            "project_path": project_path,
            "analysis_types": ["complexity", "code_smells", "maintainability"],
            "output_format": "json"
        }

        result = await call_tool("quality_assessment_comprehensive", arguments)
        self.assertIsInstance(result, list)

        response_text = result[0]["text"]
        self.assertIn("overall_score", response_text)
        self.assertIn("file_metrics", response_text)

    async def test_compliance_reporting_generate_mcp_tool(self):
        """Test compliance_reporting_generate MCP tool."""
        project_path = self.create_test_project_with_vulnerabilities()

        arguments = {
            "project_path": project_path,
            "compliance_standards": ["owasp_top_10", "pci_dss"],
            "output_format": "json"
        }

        result = await call_tool("compliance_reporting_generate", arguments)
        self.assertIsInstance(result, list)

        response_text = result[0]["text"]
        self.assertIn("overall_compliance_score", response_text)
        self.assertIn("owasp_top_10", response_text)

    async def test_quality_gates_evaluate_mcp_tool(self):
        """Test quality_gates_evaluate MCP tool."""
        project_path = self.create_test_project_with_vulnerabilities()

        arguments = {
            "project_path": project_path,
            "output_format": "json"
        }

        result = await call_tool("quality_gates_evaluate", arguments)
        self.assertIsInstance(result, list)

        response_text = result[0]["text"]
        self.assertIn("overall_result", response_text)
        self.assertIn("passed_gates", response_text)
        self.assertIn("failed_gates", response_text)

    async def test_vulnerability_database_check_mcp_tool(self):
        """Test vulnerability_database_check MCP tool."""
        project_path = self.create_test_project_with_vulnerabilities()

        arguments = {
            "project_path": project_path,
            "check_dependencies": True,
            "check_patterns": True,
            "output_format": "json"
        }

        result = await call_tool("vulnerability_database_check", arguments)
        self.assertIsInstance(result, list)

        response_text = result[0]["text"]
        self.assertIn("vulnerabilities_found", response_text)
        self.assertIn("dependency_vulnerabilities", response_text)

    async def test_custom_quality_gates_mcp_tool(self):
        """Test quality gates with custom gate definitions."""
        project_path = self.create_test_project_with_vulnerabilities()

        custom_gates = [
            {
                "name": "Complexity Limit",
                "metric": "cyclomatic_complexity",
                "threshold": 10,
                "operator": "<=",
                "severity": "high",
                "enabled": True
            },
            {
                "name": "Maintainability Minimum",
                "metric": "maintainability_index",
                "threshold": 70,
                "operator": ">=",
                "severity": "medium",
                "enabled": True
            }
        ]

        arguments = {
            "project_path": project_path,
            "gates": custom_gates,
            "fail_on_error": True,
            "output_format": "json"
        }

        result = await call_tool("quality_gates_evaluate", arguments)
        self.assertIsInstance(result, list)

        response_text = result[0]["text"]
        self.assertIn("custom_gates", response_text)

    async def test_security_quality_integration_workflow(self):
        """Test complete security and quality analysis workflow."""
        project_path = self.create_test_project_with_vulnerabilities()

        # Step 1: Security scan
        security_args = {
            "project_path": project_path,
            "scan_types": ["pattern", "dependencies"],
            "output_format": "json"
        }
        security_result = await call_tool("security_scan_comprehensive", security_args)

        # Step 2: Quality assessment
        quality_args = {
            "project_path": project_path,
            "analysis_types": ["complexity", "code_smells"],
            "output_format": "json"
        }
        quality_result = await call_tool("quality_assessment_comprehensive", quality_args)

        # Step 3: Compliance reporting
        compliance_args = {
            "project_path": project_path,
            "compliance_standards": ["owasp_top_10"],
            "output_format": "json"
        }
        compliance_result = await call_tool("compliance_reporting_generate", compliance_args)

        # Step 4: Quality gates evaluation
        gates_args = {
            "project_path": project_path,
            "output_format": "json"
        }
        gates_result = await call_tool("quality_gates_evaluate", gates_args)

        # Verify all steps worked
        self.assertIsInstance(security_result, list)
        self.assertIsInstance(quality_result, list)
        self.assertIsInstance(compliance_result, list)
        self.assertIsInstance(gates_result, list)

        # Verify security issues detected
        security_text = security_result[0]["text"]
        self.assertIn("vulnerabilities", security_text)

        # Verify quality issues detected
        quality_text = quality_result[0]["text"]
        self.assertIn("overall_score", quality_text)

        # Verify compliance calculated
        compliance_text = compliance_result[0]["text"]
        self.assertIn("overall_compliance_score", compliance_text)

        # Verify gates evaluated
        gates_text = gates_result[0]["text"]
        self.assertIn("overall_result", gates_text)


class TestSecurityQualityErrorHandling(unittest.TestCase):
    """Test error handling in security and quality operations."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_invalid_project_path_handling(self):
        """Test handling of invalid project paths."""
        arguments = {"project_path": "/nonexistent/path", "output_format": "json"}

        result = await call_tool("security_scan_comprehensive", arguments)
        self.assertIsInstance(result, list)

        # Should handle error gracefully
        response_text = result[0]["text"]
        self.assertIn("error", response_text.lower())

    async def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a directory with restricted permissions
        restricted_dir = Path(self.temp_dir) / "restricted"
        restricted_dir.mkdir(mode=0o000)

        try:
            arguments = {"project_path": str(restricted_dir), "output_format": "json"}
            result = await call_tool("quality_assessment_comprehensive", arguments)
            self.assertIsInstance(result, list)

            # Should handle permission error gracefully
            response_text = result[0]["text"]
            self.assertIn("error", response_text.lower())
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    async def test_invalid_compliance_standards(self):
        """Test handling of invalid compliance standards."""
        arguments = {
            "project_path": self.temp_dir,
            "compliance_standards": ["invalid_standard"],
            "output_format": "json"
        }

        result = await call_tool("compliance_reporting_generate", arguments)
        self.assertIsInstance(result, list)

        # Should handle invalid standard gracefully
        response_text = result[0]["text"]
        self.assertIn("error", response_text.lower())

    async def test_invalid_quality_gate_config(self):
        """Test handling of invalid quality gate configuration."""
        invalid_gates = [
            {
                "name": "Invalid Gate",
                "metric": "invalid_metric",  # Invalid metric
                "threshold": 10,
                "operator": "invalid_operator",  # Invalid operator
                "severity": "invalid_severity",  # Invalid severity
                "enabled": True
            }
        ]

        arguments = {
            "project_path": self.temp_dir,
            "gates": invalid_gates,
            "output_format": "json"
        }

        result = await call_tool("quality_gates_evaluate", arguments)
        self.assertIsInstance(result, list)

        # Should handle invalid configuration gracefully
        response_text = result[0]["text"]
        self.assertIn("error", response_text.lower())


class TestSecurityQualityPerformance(unittest.TestCase):
    """Test performance characteristics of security and quality operations."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_large_test_project(self, num_files=20):
        """Create a large test project for performance testing."""
        for i in range(num_files):
            file_path = Path(self.temp_dir) / f"test_file_{i}.py"
            content = f"""
# Test file {i}
import os
import sys

# Security issue: hardcoded secret
API_KEY_{i} = "sk-1234567890abcdef-{i}"

def complex_function_{i}(data):
    # Quality issue: high complexity
    result = []
    for item in data:
        if item is not None:
            if isinstance(item, dict):
                if item.get('active'):
                    if item.get('verified'):
                        if item.get('score', 0) > 50:
                            result.append(item)
    return result

def sql_injection_{i}(user_id):
    # Security issue: SQL injection
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute_query(query)

if __name__ == "__main__":
    print("Test file {i}")
"""
            file_path.write_text(content.strip())

        return self.temp_dir

    async def test_large_project_security_scan_performance(self):
        """Test performance of security scanning on large projects."""
        import time

        project_path = self.create_large_test_project(15)

        start_time = time.time()
        result = await call_tool("security_scan_comprehensive", {
            "project_path": project_path,
            "scan_types": ["pattern"],
            "output_format": "json"
        })
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Security scan performance: {execution_time:.2f}s for 15 files")

        self.assertIsInstance(result, list)
        self.assertLess(execution_time, 30.0, f"Security scan took {execution_time:.2f}s, expected < 30s")

    async def test_large_project_quality_assessment_performance(self):
        """Test performance of quality assessment on large projects."""
        import time

        project_path = self.create_large_test_project(10)

        start_time = time.time()
        result = await call_tool("quality_assessment_comprehensive", {
            "project_path": project_path,
            "analysis_types": ["complexity"],
            "output_format": "json"
        })
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Quality assessment performance: {execution_time:.2f}s for 10 files")

        self.assertIsInstance(result, list)
        self.assertLess(execution_time, 20.0, f"Quality assessment took {execution_time:.2f}s, expected < 20s")


if __name__ == "__main__":
    unittest.main()
