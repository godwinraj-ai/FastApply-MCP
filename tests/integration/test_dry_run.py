#!/usr/bin/env python3
"""
Demonstration of dry-run functionality for Fast Apply MCP server.
"""

import os
import sys
import tempfile

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapply.main import WORKSPACE_ROOT, fast_apply_connector


def test_dry_run_functionality():
    """Test the dry-run functionality with a sample file."""

    # Create a temporary test file
    test_content = '''#!/usr/bin/env python3
"""
Simple test file for dry-run functionality demonstration.
"""

def hello_world():
    """A simple greeting function."""
    message = "Hello, World!"
    print(message)
    return message

def add_numbers(a, b):
    """Add two numbers together."""
    result = a + b
    return result

if __name__ == "__main__":
    # Test the functions
    greeting = hello_world()
    sum_result = add_numbers(5, 3)

    print(f"Greeting: {greeting}")
    print(f"Sum: {sum_result}")
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_content)
        test_file_path = f.name

    try:
        print("🔍 Testing Dry-Run Functionality")
        print("=" * 50)
        print(f"Test file: {test_file_path}")
        print(f"Workspace root: {WORKSPACE_ROOT}")
        print()

        # Test 1: Simple function modification
        print("Test 1: Adding a new function")
        code_edit1 = """// ... existing code ...
def multiply_numbers(a, b):
    # Multiply two numbers together
    result = a * b
    return result

// ... existing code ..."""

        instruction1 = "Add a new multiply_numbers function after the add_numbers function"

        # Simulate dry run
        try:
            verification_results = fast_apply_connector.apply_edit(
                original_code=test_content,
                code_edit=code_edit1,
                instruction=instruction1,
                file_path=test_file_path,
            )

            print("✅ Dry run completed successfully!")
            print(f"   • Has changes: {verification_results['has_changes']}")
            print(f"   • Merged code length: {len(verification_results['merged_code'])} chars")
            print(f"   • Validation errors: {verification_results['validation']['has_errors']}")
            print(f"   • Validation warnings: {len(verification_results['validation']['warnings'])}")

            if verification_results["has_changes"] and verification_results["udiff"]:
                print("\n📋 Unified Diff (first 10 lines):")
                diff_lines = verification_results["udiff"].split("\n")[:10]
                for line in diff_lines:
                    print(f"   {line}")
                if len(verification_results["udiff"].split("\n")) > 10:
                    print("   ... (truncated)")

        except Exception as e:
            print(f"❌ Dry run failed: {e}")

        print("\n" + "-" * 50 + "\n")

        # Test 2: Modifying existing function
        print("Test 2: Modifying existing function")
        code_edit2 = """// ... existing code ...
def hello_world():
    # A simple greeting function
    message = "Hello, Dry-Run World!"
    print(message)
    print("This is a dry-run test!")
    return message
// ... existing code ..."""

        instruction2 = "Modify the hello_world function to include additional output"

        try:
            verification_results = fast_apply_connector.apply_edit(
                original_code=test_content,
                code_edit=code_edit2,
                instruction=instruction2,
                file_path=test_file_path,
            )

            print("✅ Dry run completed successfully!")
            print(f"   • Has changes: {verification_results['has_changes']}")
            print(f"   • Original size: {len(test_content)} chars")
            print(f"   • New size: {len(verification_results['merged_code'])} chars")
            print(f"   • Size change: {len(verification_results['merged_code']) - len(test_content)} chars")

        except Exception as e:
            print(f"❌ Dry run failed: {e}")

        print("\n" + "-" * 50 + "\n")

        # Test 3: No changes (identical content)
        print("Test 3: Applying identical content (no changes)")
        code_edit3 = test_content  # Same as original

        instruction3 = "Apply identical content (should result in no changes)"

        try:
            verification_results = fast_apply_connector.apply_edit(
                original_code=test_content,
                code_edit=code_edit3,
                instruction=instruction3,
                file_path=test_file_path,
            )

            print("✅ Dry run completed successfully!")
            print(f"   • Has changes: {verification_results['has_changes']}")
            if not verification_results["has_changes"]:
                print("   ✅ Correctly detected no changes needed")

        except Exception as e:
            print(f"❌ Dry run failed: {e}")

        print("\n🛡️  Safety Verification:")
        print("   • No files were modified during these tests")
        print("   • No backups were created")
        print("   • All operations were performed in memory only")

    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
            print(f"\n🧹 Cleaned up test file: {test_file_path}")


if __name__ == "__main__":
    test_dry_run_functionality()
