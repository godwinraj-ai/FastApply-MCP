# FastApply MCP Server Test Suite

This directory contains comprehensive tests for the FastApply MCP server implementation.

## Test Structure

```
tests/
├── README.md                 # This file
├── run_tests.py             # Simple test runner (no external dependencies)
├── test_mcp_server.py       # Comprehensive pytest-based test suite
└── sample_files/            # Test files in different programming languages
    ├── calculator.py        # Python calculator class
    ├── user-manager.js      # JavaScript user management system
    ├── TaskManager.java     # Java task management system
    └── inventory.go         # Go inventory management system
```

## Running Tests

### Option 1: Simple Test Runner (Recommended for quick testing)

```bash
cd tests
python run_tests.py
```

This runs basic functionality tests without requiring external dependencies.

### Option 2: Full Test Suite with pytest

```bash
# Install dev dependencies first
pip install -e .[dev]

# Run the full test suite
pytest tests/test_mcp_server.py -v

# Or run directly
python tests/test_mcp_server.py
```

### Option 3: Integration Tests with Live Server

For testing with a real Fast Apply server:

1. Start your Fast Apply server (e.g., LM Studio with fastapply model)
2. Set environment variables:
   ```bash
   export FAST_APPLY_URL="http://localhost:1234/v1"
   export FAST_APPLY_MODEL="fastapply-1.5b"
   export FAST_APPLY_API_KEY="optional-key"
   ```
3. Run integration tests:
   ```bash
   python tests/test_mcp_server.py
   ```

## Test Coverage

### Unit Tests
- ✅ FastApplyConnector initialization
- ✅ Markdown code block stripping
- ✅ Configuration updates
- ✅ Error handling and validation
- ✅ MCP tool schema validation
- ✅ Security path validation

### Integration Tests
- ✅ MCP server startup
- ✅ Tool listing functionality
- ✅ File edit operations
- ✅ Multi-language file support
- ✅ Real API integration (when server available)

### Sample Files
The test suite includes sample files in multiple programming languages:

1. **calculator.py** - Python class with methods for basic math operations
2. **user-manager.js** - JavaScript user management with Map-based storage
3. **TaskManager.java** - Java task management with inner Task class
4. **inventory.go** - Go inventory system with struct and methods

## Test Scenarios

### Basic Edit Tests
Each sample file can be tested with realistic code edits:

```python
# Example: Add a power method to Python calculator
{
    "file": "calculator.py",
    "instructions": "Add a power method to the Calculator class",
    "code_edit": "def power(self, a, b): ...",
    "expected": "def power(self, a, b):"
}
```

### Error Handling Tests
- Invalid file paths (directory traversal attempts)
- Missing files
- Empty or invalid parameters
- API connection failures
- Malformed responses

### Security Tests
- Path validation prevents directory traversal
- File existence validation
- Parameter sanitization
- Proper error messages without information leakage

## Environment Configuration

The MCP server can be configured via environment variables:

```bash
# Fast Apply API Configuration
export FAST_APPLY_URL="http://localhost:1234/v1"          # API endpoint
export FAST_APPLY_MODEL="fastapply-1.5b"                  # Model name
export FAST_APPLY_API_KEY="your-api-key"                  # API key (optional)
export FAST_APPLY_TIMEOUT="30.0"                          # Request timeout
export FAST_APPLY_MAX_TOKENS="8000"                       # Max response tokens
export FAST_APPLY_TEMPERATURE="0.05"                      # Generation temperature
```

## Expected Test Results

When running `python run_tests.py`, you should see:

```
🧪 Fast Apply MCP Server Test Suite
============================================================
🚀 Testing server startup...
✅ Server module imported successfully
✅ FastApplyConnector initialized successfully
✅ Tools available: 1

🔧 Testing MCP tool structure...
✅ MCP tool structure is valid

📁 Testing file operations...
✅ Created test file: /tmp/tmpXXXXXX/test.py
✅ Path validation working correctly
✅ File existence validation working correctly
✅ Parameter validation working correctly

✂️  Testing markdown stripping...
✅ Markdown stripping working correctly

🌍 Testing environment configuration...
✅ Environment configuration working correctly

🎯 Testing with sample files...
✅ Found sample file: tests/sample_files/calculator.py
✅ Found sample file: tests/sample_files/user-manager.js
✅ Found sample file: tests/sample_files/TaskManager.java
✅ Found sample file: tests/sample_files/inventory.go
✅ All 4 sample files are ready for testing

============================================================
📊 TEST REPORT
============================================================
✅ PASS Server Startup
✅ PASS MCP Tool Structure
✅ PASS File Operations
✅ PASS Markdown Stripping
✅ PASS Environment Configuration
✅ PASS Sample Files
------------------------------------------------------------
Total tests: 6
Passed: 6
Failed: 0
Success rate: 100.0%

🎉 All tests passed! The MCP server is ready for use.
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root directory
2. **File not found errors**: Ensure sample files exist in `tests/sample_files/`
3. **API connection errors**: Verify Fast Apply server is running and accessible
4. **Permission errors**: Check file permissions in the test directory

### Debug Mode

For more detailed output, you can modify the test scripts to include debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Testing

You can also test individual components manually:

```python
# Test the connector directly
from main import FastApplyConnector
connector = FastApplyConnector()

# Test tool listing
from main import list_tools
tools = list_tools()
print(json.dumps(tools, indent=2))

# Test a specific edit (requires running server)
from main import call_tool
result = call_tool("edit_file", {
    "target_file": "tests/sample_files/calculator.py",
    "instructions": "Add a comment at the top",
    "code_edit": "# Modified by Fast Apply MCP"
})
```

## Contributing

When adding new tests:

1. Add unit tests to `test_mcp_server.py`
2. Add integration scenarios to `run_tests.py`
3. Create new sample files in appropriate languages
4. Update this README with new test descriptions
5. Ensure all tests pass before submitting

## Performance Notes

- Unit tests run in <1 second
- Integration tests depend on Fast Apply server response time
- Sample file operations are typically <100ms each
- Full test suite completes in <10 seconds with a local server
