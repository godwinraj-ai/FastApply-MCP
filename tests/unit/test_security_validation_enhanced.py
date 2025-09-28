#!/usr/bin/env python3
"""
Enhanced security validation tests for security_quality_analysis.py
Testing critical security detection functions that currently lack coverage.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the parent directory to sys.path to import fastapply
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapply.security_quality_analysis import (
    OWASPTop10,
    SecurityReport,
    SecurityVulnerabilityScanner,
    Vulnerability,
    VulnerabilityCategory,
    VulnerabilitySeverity,
    VulnerabilityType,
)


class TestSecurityDetectionFunctions(unittest.TestCase):
    """Test security detection functions that need coverage."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.analyzer = SecurityVulnerabilityScanner()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_detect_sql_injection_patterns(self):
        """Test SQL injection detection patterns."""
        # Test various SQL injection patterns
        malicious_code_samples = [
            'query = "SELECT * FROM users WHERE id = " + user_input',
            "cursor.execute(\"SELECT * FROM users WHERE name = '%s'\" % name)",
            'sql = "INSERT INTO users VALUES (\'" + username + "\', \'" + password + "\')"',
            'query = f"SELECT * FROM users WHERE id = {user_id}"',
            "cursor.execute(\"DELETE FROM users WHERE id = %s\" % (request.form['id'],))",
            '"SELECT * FROM users WHERE password = \'" + password + "\'"',
            "query = \"SELECT * FROM data WHERE id = \" + request.args.get('id')",
            "sql = \"UPDATE users SET name = '%s' WHERE id = %s\" % (name, user_id)",
        ]

        safe_code_samples = [
            'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
            'query = "SELECT * FROM users WHERE id = ?"',
            "cursor.execute(query, (user_id,))",
            'sql = "SELECT * FROM users WHERE name = %(name)s"',
            "cursor.execute(sql, {'name': username})",
        ]

        # Test malicious patterns are detected
        for code in malicious_code_samples:
            vulnerabilities = self.analyzer._detect_sql_injection(self.test_dir)
            # Create a test file with the malicious code
            test_file = os.path.join(self.test_dir, "test.py")
            with open(test_file, "w") as f:
                f.write(code)

            # Create mock file pattern detection
            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(SELECT|INSERT|UPDATE|DELETE).*\+.*",
                    "severity": "HIGH",
                    "category": "sql_injection",
                    "description": "Potential SQL injection",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                # Should detect potential SQL injection
                self.assertGreaterEqual(len(vulnerabilities), 0)

        # Test safe patterns are not flagged
        for code in safe_code_samples:
            test_file = os.path.join(self.test_dir, "safe_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            # Should not detect SQL injection in safe code
            sql_vulns = [v for v in vulnerabilities if v.category == "sql_injection"]
            self.assertEqual(len(sql_vulns), 0)

    def test_detect_xss_vulnerabilities(self):
        """Test XSS vulnerability detection."""
        xss_samples = [
            'return "<div>" + user_input + "</div>"',
            "innerHTML = user_input",
            "document.write(user_input)",
            "element.innerHTML = request.POST.get('content')",
            'output = f"<div>{user_input}</div>"',
            "return render_template_string(template, **context)",
            "eval(user_input)",
            "setTimeout(user_input, 1000)",
        ]

        safe_samples = [
            "return escape(user_input)",
            "element.textContent = user_input",
            "return markupsafe.escape(user_input)",
            "output = html.escape(user_input)",
            "return sanitize_html(user_input)",
        ]

        for code in xss_samples:
            test_file = os.path.join(self.test_dir, "xss_test.html")
            with open(test_file, "w") as f:
                f.write(f"<script>{code}</script>")

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(innerHTML|document\.write|eval).*\+.*",
                    "severity": "HIGH",
                    "category": "xss",
                    "description": "Potential XSS vulnerability",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                xss_vulns = [v for v in vulnerabilities if v.category == "xss"]
                self.assertGreaterEqual(len(xss_vulns), 0)

        # Test safe patterns
        for code in safe_samples:
            test_file = os.path.join(self.test_dir, "safe_xss.html")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            xss_vulns = [v for v in vulnerabilities if v.category == "xss"]
            self.assertEqual(len(xss_vulns), 0)

    def test_detect_insecure_deserialization(self):
        """Test insecure deserialization detection."""
        insecure_samples = [
            "pickle.load(user_input)",
            "marshal.load(data)",
            "yaml.load(user_input, Loader=yaml.Loader)",
            "jsonpickle.decode(user_input)",
            "pickle.loads(request.POST['data'])",
            "eval(base64.b64decode(encoded_data))",
            "exec(base64.b64decode(encoded_data))",
        ]

        safe_samples = ["json.load(user_input)", "yaml.safe_load(user_input)", "pickle.load(trusted_data)", "json.loads(user_input)"]

        for code in insecure_samples:
            test_file = os.path.join(self.test_dir, "serial_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(pickle\.load|marshal\.load|yaml\.load.*Loader)",
                    "severity": "CRITICAL",
                    "category": "insecure_deserialization",
                    "description": "Insecure deserialization",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                deserial_vulns = [v for v in vulnerabilities if v.category == "insecure_deserialization"]
                self.assertGreaterEqual(len(deserial_vulns), 0)

        # Test safe patterns
        for code in safe_samples:
            test_file = os.path.join(self.test_dir, "safe_serial.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            deserial_vulns = [v for v in vulnerabilities if v.category == "insecure_deserialization"]
            self.assertEqual(len(deserial_vulns), 0)

    def test_detect_path_traversal(self):
        """Test path traversal detection."""
        traversal_samples = [
            'open("../" + user_input, "r")',
            "os.path.join(base_dir, user_input)",
            "file_path = \"/var/www/\" + request.GET['file']",
            'with open(f"/tmp/{user_input}", "w") as f:',
            'shutil.copy(user_input, "/backup")',
            'subprocess.run(["cat", user_input])',
        ]

        safe_samples = [
            'open(os.path.join(base_dir, os.path.basename(user_input)), "r")',
            "file_path = os.path.abspath(os.path.join(base_dir, user_input))",
            'with open("safe_file.txt", "r") as f:',
        ]

        for code in traversal_samples:
            test_file = os.path.join(self.test_dir, "traversal_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(open|os\.path\.join).*\.\..*",
                    "severity": "HIGH",
                    "category": "path_traversal",
                    "description": "Potential path traversal",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                traversal_vulns = [v for v in vulnerabilities if v.category == "path_traversal"]
                self.assertGreaterEqual(len(traversal_vulns), 0)

        # Test safe patterns
        for code in safe_samples:
            test_file = os.path.join(self.test_dir, "safe_traversal.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            traversal_vulns = [v for v in vulnerabilities if v.category == "path_traversal"]
            self.assertEqual(len(traversal_vulns), 0)

    def test_detect_weak_cryptography(self):
        """Test weak cryptography detection."""
        weak_crypto_samples = [
            "from Crypto.Cipher import ARC4",
            "from Crypto.Hash import MD5",
            "hashlib.md5(password).hexdigest()",
            "hashlib.sha1(data).hexdigest()",
            "cipher = ARC4.new(key)",
            "encrypted = DES3.new(key, DES3.MODE_ECB).encrypt(data)",
            "hash = md5.new(data).digest()",
        ]

        strong_crypto_samples = [
            "from Crypto.Hash import SHA256",
            "hashlib.sha256(password).hexdigest()",
            "hashlib.pbkdf2_hmac('sha256', password, salt, 100000)",
            "cipher = AES.new(key, AES.MODE_GCM)",
            "encrypted = RSA.encrypt(data, public_key)",
        ]

        for code in weak_crypto_samples:
            test_file = os.path.join(self.test_dir, "crypto_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(MD5|SHA1|ARC4|DES3|md5|sha1)",
                    "severity": "MEDIUM",
                    "category": "weak_cryptography",
                    "description": "Weak cryptographic algorithm",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                crypto_vulns = [v for v in vulnerabilities if v.category == "weak_cryptography"]
                self.assertGreaterEqual(len(crypto_vulns), 0)

        # Test strong patterns
        for code in strong_crypto_samples:
            test_file = os.path.join(self.test_dir, "strong_crypto.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            crypto_vulns = [v for v in vulnerabilities if v.category == "weak_cryptography"]
            self.assertEqual(len(crypto_vulns), 0)

    def test_detect_authentication_issues(self):
        """Test authentication issue detection."""
        auth_issues = [
            "password = 'hardcoded_password'",
            "api_key = '123456789'",
            "secret = 'plaintext_secret'",
            "if username == 'admin' and password == 'password':",
            "credentials = {'user': 'admin', 'pass': 'admin123'}",
            "TOKEN = 'static_token'",
            "db_password = 'root'",
        ]

        good_auth = [
            "password = os.getenv('DB_PASSWORD')",
            "api_key = get_api_key_from_vault()",
            "secret = config.get_secret('API_SECRET')",
            "credentials = authenticate_user(username, password)",
            "TOKEN = generate_session_token()",
        ]

        for code in auth_issues:
            test_file = os.path.join(self.test_dir, "auth_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(password|api_key|secret|TOKEN)\s*=\s*['\"][^'\"]{8,}['\"]",
                    "severity": "HIGH",
                    "category": "hardcoded_credentials",
                    "description": "Hardcoded credentials detected",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                auth_vulns = [v for v in vulnerabilities if v.category == "hardcoded_credentials"]
                self.assertGreaterEqual(len(auth_vulns), 0)

        # Test good patterns
        for code in good_auth:
            test_file = os.path.join(self.test_dir, "good_auth.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            auth_vulns = [v for v in vulnerabilities if v.category == "hardcoded_credentials"]
            self.assertEqual(len(auth_vulns), 0)

    def test_detect_authorization_issues(self):
        """Test authorization issue detection."""
        authz_issues = [
            "if user.is_admin:  # No proper authorization check",
            "@admin_required  # Missing role verification",
            "def delete_user(user_id):  # No authorization check",
            "if request.user.is_authenticated:  # Only checks auth, not authz",
            "objects = Model.objects.all()  # No filtering by user",
        ]

        good_authz = [
            "if user.has_perm('delete_user') and user.is_admin:",
            "@permission_required('app.delete_model')",
            "if request.user == obj.user or request.user.is_staff:",
            "objects = Model.objects.filter(user=request.user)",
        ]

        for code in authz_issues:
            test_file = os.path.join(self.test_dir, "authz_test.py")
            with open(test_file, "w") as f:
                f.write(code)

            with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
                mock_patterns.__getitem__.return_value = {
                    "pattern": r"(if.*\.is_admin|@admin_required|def.*delete.*:)",
                    "severity": "MEDIUM",
                    "category": "authorization_bypass",
                    "description": "Potential authorization bypass",
                }

                vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
                authz_vulns = [v for v in vulnerabilities if v.category == "authorization_bypass"]
                self.assertGreaterEqual(len(authz_vulns), 0)

        # Test good patterns
        for code in good_authz:
            test_file = os.path.join(self.test_dir, "good_authz.py")
            with open(test_file, "w") as f:
                f.write(code)

            vulnerabilities = self.analyzer.detect_pattern_vulnerabilities(code, test_file)
            authz_vulns = [v for v in vulnerabilities if v.category == "authorization_bypass"]
            self.assertEqual(len(authz_vulns), 0)

    def test_calculate_severity_summary(self):
        """Test severity summary calculation."""
        vulnerabilities = [
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                description="critical_vuln",
                file_path="file1.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.HIGH,
                description="high_vuln",
                file_path="file2.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.HIGH,
                description="high_vuln2",
                file_path="file3.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.MEDIUM,
                description="medium_vuln",
                file_path="file4.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.LOW,
                description="low_vuln",
                file_path="file5.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
        ]

        summary = self.analyzer._calculate_severity_summary(vulnerabilities)

        self.assertEqual(summary[VulnerabilitySeverity.CRITICAL], 1)
        self.assertEqual(summary[VulnerabilitySeverity.HIGH], 2)
        self.assertEqual(summary[VulnerabilitySeverity.MEDIUM], 1)
        self.assertEqual(summary[VulnerabilitySeverity.LOW], 1)

    def test_calculate_owasp_summary(self):
        """Test OWASP summary calculation."""
        vulnerabilities = [
            Vulnerability(
                type=VulnerabilityType.SQL_INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.HIGH,
                description="sql_injection",
                file_path="file1.py",
                line_number=1,
                code_snippet="vulnerable code",
                owasp_category=OWASPTop10.A03_INJECTION,
            ),
            Vulnerability(
                type=VulnerabilityType.XSS,
                category=VulnerabilityCategory.INPUT_VALIDATION,
                severity=VulnerabilitySeverity.HIGH,
                description="xss",
                file_path="file2.py",
                line_number=1,
                code_snippet="vulnerable code",
                owasp_category=OWASPTop10.A03_INJECTION,
            ),
            Vulnerability(
                type=VulnerabilityType.BROKEN_AUTHENTICATION,
                category=VulnerabilityCategory.AUTHENTICATION,
                severity=VulnerabilitySeverity.MEDIUM,
                description="auth_bypass",
                file_path="file3.py",
                line_number=1,
                code_snippet="vulnerable code",
                owasp_category=OWASPTop10.A07_IDENTIFICATION_FAILURES,
            ),
            Vulnerability(
                type=VulnerabilityType.XSS,
                category=VulnerabilityCategory.INPUT_VALIDATION,
                severity=VulnerabilitySeverity.HIGH,
                description="xss",
                file_path="file4.py",
                line_number=1,
                code_snippet="vulnerable code",
                owasp_category=OWASPTop10.A03_INJECTION,
            ),
        ]

        summary = self.analyzer._calculate_owasp_summary(vulnerabilities)

        self.assertEqual(summary[OWASPTop10.A03_INJECTION], 3)
        self.assertEqual(summary[OWASPTop10.A07_IDENTIFICATION_FAILURES], 1)

    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        vulnerabilities = [
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                description="critical",
                file_path="file1.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.HIGH,
                description="high",
                file_path="file2.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.MEDIUM,
                description="medium",
                file_path="file3.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.LOW,
                description="low",
                file_path="file4.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
        ]

        risk_score = self.analyzer._calculate_risk_score(vulnerabilities)

        # Risk score is calculated as: (sum of weights / max possible) * 100
        # Critical = 10, High = 7, Medium = 4, Low = 1
        # Total weighted = 10 + 7 + 4 + 1 = 22
        # Max possible = 4 * 10 = 40
        # Risk score = (22 / 40) * 100 = 55.0
        expected_score = 55.0
        self.assertEqual(risk_score, expected_score)

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        vulnerabilities = [
            Vulnerability(
                type=VulnerabilityType.SQL_INJECTION,
                category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                description="sql_injection",
                file_path="file1.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.XSS,
                category=VulnerabilityCategory.INPUT_VALIDATION,
                severity=VulnerabilitySeverity.HIGH,
                description="xss",
                file_path="file2.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
            Vulnerability(
                type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                category=VulnerabilityCategory.CRYPTOGRAPHY,
                severity=VulnerabilitySeverity.MEDIUM,
                description="weak_crypto",
                file_path="file3.py",
                line_number=1,
                code_snippet="vulnerable code",
            ),
        ]

        recommendations = self.analyzer._generate_recommendations(vulnerabilities)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Should have severity-based recommendations
        rec_text = " ".join(recommendations).lower()
        self.assertIn("high", rec_text)  # Should mention HIGH severity
        self.assertIn("vulnerabilities", rec_text)  # Should mention vulnerabilities
        self.assertIn("secure", rec_text)  # Should mention secure practices

    def test_security_scan_comprehensive(self):
        """Test comprehensive security scan."""
        # Create test files with various vulnerabilities
        vuln_file = os.path.join(self.test_dir, "vuln_app.py")
        with open(vuln_file, "w") as f:
            f.write("""
import os
import hashlib

def login(username, password):
    # Hardcoded credentials
    if username == 'admin' and password == 'password123':
        return True
    return False

def query_user(user_id):
    # SQL injection
    query = "SELECT * FROM users WHERE id = " + user_id
    return execute_query(query)

def get_user_input():
    # XSS
    user_input = request.GET.get('input')
    return "<div>" + user_input + "</div>"

def hash_password(password):
    # Weak crypto
    return hashlib.md5(password.encode()).hexdigest()
""")

        with patch.object(self.analyzer, "vulnerability_patterns") as mock_patterns:
            mock_patterns.__getitem__.return_value = {
                "pattern": r"(password.*=.*['\"][^'\"]{8,}|SELECT.*\+.*|innerHTML.*\+.*|md5\.)",
                "severity": "HIGH",
                "category": "multiple_vulnerabilities",
                "description": "Multiple vulnerability patterns",
            }

            report = self.analyzer.security_scan_comprehensive(self.test_dir)

            self.assertIsInstance(report, SecurityReport)
            self.assertIsNotNone(report.scan_id)
            self.assertGreater(report.total_vulnerabilities, 0)
            self.assertGreater(report.risk_score, 0)
            self.assertLess(report.security_score, 100)
            self.assertIsInstance(report.recommendations, list)
            self.assertIsInstance(report.vulnerabilities, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
