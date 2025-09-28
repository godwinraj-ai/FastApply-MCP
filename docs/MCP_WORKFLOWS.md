# FastApply MCP Workflows Documentation

## Table of Contents
- [Getting Started with MCP](#getting-started-with-mcp)
- [Project Onboarding via MCP](#project-onboarding-via-mcp)
- [Session Management via MCP](#session-management-via-mcp)
- [Code Analysis Workflows via MCP](#code-analysis-workflows-via-mcp)
- [Multi-File Editing via MCP](#multi-file-editing-via-mcp)
- [Debugging via MCP](#debugging-via-mcp)
- [Refactoring via MCP](#refactoring-via-mcp)
- [Security & Quality via MCP](#security--quality-via-mcp)
- [Performance Optimization via MCP](#performance-optimization-via-mcp)
- [Advanced Patterns via MCP](#advanced-patterns-via-mcp)
- [Best Practices via MCP](#best-practices-via-mcp)
- [MCP Tool Reference](#mcp-tool-reference)

---

## Getting Started with MCP

### MCP Server Setup
```bash
# Start FastApply MCP Server
python -m fastapply.main

# Or use MCP client configuration
{
  "mcpServers": {
    "fastapply": {
      "command": "python",
      "args": ["-m", "fastapply.main"],
      "env": {
        "OPENAI_API_KEY": "your_api_key_here",
        "WORKSPACE_ROOT": "/path/to/your/project"
      }
    }
  }
}
```

### Basic MCP Tool Calls
```python
# Example: Basic file search via MCP
result = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "*.py",
    "excludePatterns": ["__pycache__", "*.pyc"]
})

# Example: Code structure analysis
result = await mcp_client.call_tool("analyze_code_structure", {
    "file_path": "./src/main.py"
})
```

---

## Project Onboarding via MCP

### 1. Initial Project Discovery
```python
# Comprehensive project analysis using auto_ast_intelligence
discovery_result = await mcp_client.call_tool("auto_ast_intelligence", {
    "query": "analyze entire project structure and identify key components",
    "context": "project onboarding and architecture discovery",
    "auto_execute": True
})

# Alternative: Manual step-by-step analysis
files_result = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "*.py",
    "excludePatterns": ["__pycache__", "*.pyc", "node_modules/*"]
})

structure_result = await mcp_client.call_tool("analyze_code_structure", {
    "file_path": "./src/main.py"
})

print(f"Discovered {len(files_result.get('files', []))} Python files")
print(f"Main structure: {structure_result.get('structure', {})}")
```

### 2. Project Intelligence Collection
```python
# Store project knowledge via Qdrant
project_intelligence = {
    "project_name": "my-web-app",
    "architecture": "microservices",
    "main_languages": ["python", "javascript"],
    "frameworks": ["django", "react"],
    "key_modules": ["auth", "api", "database"],
    "complexity_level": "medium",
    "team_size": 5
}

# Store via MCP Qdrant integration
await mcp_client.call_tool("qdrant-store", {
    "information": json.dumps(project_intelligence),
    "metadata": {
        "type": "project_intelligence",
        "project": "my-web-app",
        "created_date": "2025-09-28",
        "analysis_type": "onboarding"
    }
})
```

### 3. Establish Quality Baseline
```python
# Read main file for baseline analysis
main_file_content = await mcp_client.call_tool("read_multiple_files", {
    "paths": ["./src/main.py"]
})

# Create comprehensive baseline
baseline_result = await mcp_client.call_tool("llm_analyze_code", {
    "code": main_file_content,
    "language": "python",
    "analysis_type": "comprehensive",
    "use_collective_memory": True
})

# Store baseline metrics
await mcp_client.call_tool("qdrant-store", {
    "information": json.dumps(baseline_result),
    "metadata": {
        "type": "quality_baseline",
        "project": "my-web-app",
        "baseline_date": "2025-09-28",
        "metrics": baseline_result.get("metrics", {})
    }
})
```

### 4. Pattern Recognition
```python
# Find architectural patterns
pattern_result = await mcp_client.call_tool("llm_search_pattern", {
    "query": "find MVC architecture patterns and module organization",
    "language": "python",
    "path": "./src",
    "use_collective_memory": True,
    "max_results": 15
})

print(f"Found {len(pattern_result.get('patterns', []))} architectural patterns")
```

---

## Session Management via MCP

### 1. Session Initialization
```python
# Load existing session context
session_result = await mcp_client.call_tool("qdrant-find", {
    "query": "project session state my-web-app current context",
    "limit": 1
})

if session_result and len(session_result) > 0:
    # Restore previous context
    previous_state = json.loads(session_result[0]["content"])
    print(f"Restored session from {previous_state.get('last_updated', 'unknown')}")
else:
    # Initialize new session
    session_state = {
        "session_id": str(uuid.uuid4()),
        "project": "my-web-app",
        "start_time": datetime.now().isoformat(),
        "tasks_completed": [],
        "current_focus": "project_onboarding"
    }

    # Store new session
    await mcp_client.call_tool("qdrant-store", {
        "information": json.dumps(session_state),
        "metadata": {
            "type": "session_state",
            "session_id": session_state["session_id"],
            "project": "my-web-app",
            "status": "active"
        }
    })
```

### 2. Session Checkpointing
```python
async def create_session_checkpoint(session_state):
    """Create periodic session checkpoints via MCP"""
    checkpoint_data = {
        "session_id": session_state["session_id"],
        "timestamp": datetime.now().isoformat(),
        "current_task": session_state["current_focus"],
        "tasks_completed": session_state["tasks_completed"],
        "context_summary": "Project onboarding in progress",
        "next_actions": ["Complete architectural analysis", "Set up testing framework"]
    }

    await mcp_client.call_tool("qdrant-store", {
        "information": json.dumps(checkpoint_data),
        "metadata": {
            "type": "session_checkpoint",
            "session_id": session_state["session_id"],
            "project": session_state["project"],
            "timestamp": checkpoint_data["timestamp"]
        }
    })

# Example usage
await create_session_checkpoint(current_session_state)
```

### 3. Session Persistence
```python
async def save_session_state(session_state):
    """Save complete session state via MCP"""
    complete_state = {
        **session_state,
        "end_time": datetime.now().isoformat(),
        "total_duration": "2h 15m",
        "key_insights": ["Microservices architecture confirmed", "Auth module needs refactoring"],
        "next_steps": ["Refactor auth module", "Add integration tests"],
        "open_issues": ["Database connection pooling", "API rate limiting"]
    }

    await mcp_client.call_tool("qdrant-store", {
        "information": json.dumps(complete_state),
        "metadata": {
            "type": "session_complete",
            "session_id": session_state["session_id"],
            "project": session_state["project"],
            "duration_minutes": 135,
            "tasks_completed": len(session_state["tasks_completed"])
        }
    })
```

---

## Code Analysis Workflows via MCP

### 1. Structural Analysis
```python
# Analyze code structure
structure_result = await mcp_client.call_tool("analyze_code_structure", {
    "file_path": "./src/auth/authorization.py"
})

print("=== Code Structure Analysis ===")
print(f"Functions: {len(structure_result.get('functions', []))}")
print(f"Classes: {len(structure_result.get('classes', []))}")
print(f"Imports: {len(structure_result.get('imports', []))}")
print(f"Complexity: {structure_result.get('complexity_metrics', {})}")

# Deep analysis with LLM
file_content = await mcp_client.call_tool("read_multiple_files", {
    "paths": ["./src/auth/authorization.py"]
})

deep_analysis = await mcp_client.call_tool("llm_analyze_code", {
    "code": file_content,
    "language": "python",
    "analysis_type": "architecture",
    "use_collective_memory": True
})

print("=== Architectural Analysis ===")
for insight in deep_analysis.get("insights", []):
    print(f"- {insight}")
```

### 2. Pattern Discovery
```python
# Find specific code patterns
auth_patterns = await mcp_client.call_tool("llm_search_pattern", {
    "query": "find authentication middleware patterns with JWT tokens",
    "language": "python",
    "path": "./src",
    "use_collective_memory": True,
    "max_results": 10
})

print(f"Found {len(auth_patterns.get('matches', []))} authentication patterns")

# Generate custom analysis rules
auth_rule = await mcp_client.call_tool("llm_generate_rule", {
    "query": "authentication patterns with JWT tokens and user validation",
    "language": "python",
    "examples": [
        "def authenticate_user(token):",
        "jwt.decode(token, secret_key, algorithms=['HS256'])",
        "return User.objects.get(id=payload['user_id'])"
    ],
    "rule_type": "pattern",
    "use_collective_memory": True
})

print(f"Generated rule: {auth_rule.get('rule', 'No rule generated')}")
```

### 3. Dependency Analysis
```python
# Find symbol references
symbol_refs = await mcp_client.call_tool("find_references", {
    "symbol": "User",
    "path": "./src",
    "symbol_type": "class"
})

print(f"Found {len(symbol_refs.get('references', []))} references to User class:")
for ref in symbol_refs.get("references", []):
    print(f"  - {ref.get('file', 'unknown')}:{ref.get('line', 0)} ({ref.get('context', 'no context')})")

# Enhanced analysis with relationship mapping
enhanced_analysis = await mcp_client.call_tool("llm_analyze_code", {
    "code": "\n".join([open(f).read() for f in symbol_refs.get('files', [])[:3]]),
    "language": "python",
    "analysis_type": "dependencies",
    "use_collective_memory": True
})

print("=== Dependency Analysis ===")
for dep in enhanced_analysis.get("dependencies", []):
    print(f"{dep.get('source', 'unknown')} -> {dep.get('target', 'unknown')} ({dep.get('type', 'unknown')})")
```

### 4. Multi-File Analysis
```python
# Analyze multiple files simultaneously
multi_file_result = await mcp_client.call_tool("read_multiple_files", {
    "paths": [
        "./src/auth/authorization.py",
        "./src/auth/middleware.py",
        "./src/auth/models.py"
    ]
})

comprehensive_analysis = await mcp_client.call_tool("llm_analyze_code", {
    "code": multi_file_result,
    "language": "python",
    "analysis_type": "comprehensive",
    "use_collective_memory": True
})

print("=== Multi-File Analysis ===")
for category, findings in comprehensive_analysis.get("findings", {}).items():
    print(f"{category}: {len(findings)} items")
    for finding in findings[:3]:  # Show first 3
        print(f"  - {finding}")
```

---

## Multi-File Editing via MCP

### 1. Batch Pattern Replacement
```python
# Find all files with a specific pattern
files_to_edit = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "console.log(",
    "excludePatterns": ["*.min.js", "node_modules/*"]
})

print(f"Found {len(files_to_edit.get('files', []))} files to edit")

# Execute batch edits
edit_results = []
for file_path in files_to_edit.get("files", []):
    result = await mcp_client.call_tool("edit_file", {
        "target_file": file_path,
        "instruction": "Replace console.log with proper logging",
        "code_edit": """
# Replace console.log statements with proper logging
# console.log('debug info')  ->  logger.debug('debug info')
# console.log('error info')  ->  logger.error('error info')
""",
        "force": False
    })

    edit_results.append(result)
    print(f"Edited {file_path}: {result.get('success', False)}")

successful_edits = sum(1 for r in edit_results if r.get('success', False))
print(f"Successfully edited {successful_edits}/{len(edit_results)} files")
```

### 2. Safe Refactoring Across Files
```python
# Find all references to a symbol
symbol_refs = await mcp_client.call_tool("find_references", {
    "symbol": "calculateTotal",
    "path": "./src",
    "symbol_type": "function"
})

print(f"Found {len(symbol_refs.get('references', []))} references to refactor")

# Dry run preview for each file
for ref in symbol_refs.get("references", []):
    preview_result = await mcp_client.call_tool("dry_run_edit_file", {
        "target_file": ref.get("file", ""),
        "instruction": f"Rename function from calculateTotal to calculate_order_total",
        "code_edit": "calculateTotal -> calculate_order_total"
    })

    if preview_result.get("success"):
        print(f"Preview for {ref.get('file', '')}:")
        print(preview_result.get("preview", "No preview available"))
        print("---")

        # Execute actual edit
        edit_result = await mcp_client.call_tool("edit_file", {
            "target_file": ref.get("file", ""),
            "instruction": f"Rename function from calculateTotal to calculate_order_total",
            "code_edit": "calculateTotal -> calculate_order_total",
            "force": False
        })

        print(f"Edit result: {edit_result.get('success', False)}")
```

### 3. Cross-File Code Transformation
```python
# Generate transformation rule
transformation_rule = await mcp_client.call_tool("llm_generate_rule", {
    "query": "convert Promise chains to async/await patterns in JavaScript",
    "language": "javascript",
    "examples": [
        "fetch('/api/data').then(res => res.json()).then(data => console.log(data))",
        "async function fetchData() { const res = await fetch('/api/data'); const data = await res.json(); console.log(data); }"
    ],
    "rule_type": "transformation",
    "use_collective_memory": True
})

# Find JavaScript files
js_files = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "*.js",
    "excludePatterns": ["*.min.js", "node_modules/*"]
})

# Apply transformation
transformation_results = []
for js_file in js_files.get("files", []):
    result = await mcp_client.call_tool("edit_file", {
        "target_file": js_file,
        "instruction": "Convert Promise chains to async/await patterns",
        "code_edit": transformation_rule.get("rule", ""),
        "force": False
    })

    transformation_results.append({
        "file": js_file,
        "success": result.get("success", False),
        "error": result.get("error")
    })

    print(f"Transformed {js_file}: {result.get('success', False)}")
```

### 4. Multi-File Content Insertion
```python
# Add import statements to multiple files
files_needing_imports = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "def.*logger.*:",
    "excludePatterns": ["*.min.js", "node_modules/*"]
})

import_insertion = """
# Add logging import at the top of the file
import logging

logger = logging.getLogger(__name__)
"""

for file_path in files_needing_imports.get("files", []):
    result = await mcp_client.call_tool("edit_file", {
        "target_file": file_path,
        "instruction": "Add logging import at the top of the file",
        "code_edit": import_insertion,
        "force": False
    })

    print(f"Added import to {file_path}: {result.get('success', False)}")
```

---

## Debugging via MCP

### 1. Error Analysis Workflow
```python
async def debug_code_error(file_path, error_line, error_message):
    """Comprehensive error debugging via MCP"""

    # 1. Read the problematic file
    file_content = await mcp_client.call_tool("read_multiple_files", {
        "paths": [file_path]
    })

    # 2. Analyze error context
    context_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_content,
        "language": "python",
        "analysis_type": "debug",
        "use_collective_memory": True
    })

    # 3. Search for similar error patterns
    similar_errors = await mcp_client.call_tool("qdrant-find", {
        "query": f"python error {error_message} debugging solution",
        "limit": 5
    })

    # 4. Generate debugging suggestions
    debug_suggestions = await mcp_client.call_tool("llm_search_pattern", {
        "query": f"fix {error_message} in python code near line {error_line}",
        "language": "python",
        "path": file_path,
        "use_collective_memory": True
    })

    # 5. Generate fix suggestions
    fix_suggestions = await mcp_client.call_tool("llm_generate_rule", {
        "query": f"fix {error_message} in python code",
        "language": "python",
        "examples": [
            f"# Code with {error_message}",
            f"# Fixed code without {error_message}"
        ],
        "rule_type": "debug_fix",
        "use_collective_memory": True
    })

    return {
        "context_analysis": context_analysis,
        "similar_errors": similar_errors,
        "suggestions": debug_suggestions,
        "fix_suggestions": fix_suggestions
    }

# Usage example
debug_result = await debug_code_error(
    "./src/auth/authorization.py",
    45,
    "TypeError: 'NoneType' object is not callable"
)

print("=== Debug Analysis ===")
for suggestion in debug_result.get("suggestions", {}).get("matches", []):
    print(f"- {suggestion}")
```

### 2. Performance Bottleneck Analysis
```python
async def analyze_performance_bottlenecks(file_path):
    """Analyze performance issues via MCP"""

    # 1. Structural complexity analysis
    structure = await mcp_client.call_tool("analyze_code_structure", {
        "file_path": file_path
    })

    # 2. Read file content
    file_content = await mcp_client.call_tool("read_multiple_files", {
        "paths": [file_path]
    })

    # 3. LLM-based performance analysis
    perf_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_content,
        "language": "python",
        "analysis_type": "performance",
        "use_collective_memory": True
    })

    # 4. Generate optimization suggestions
    optimizations = await mcp_client.call_tool("llm_generate_rule", {
        "query": "optimize performance bottlenecks in python code",
        "language": "python",
        "examples": [
            "# Slow nested loops\nfor i in range(1000):\n    for j in range(1000):\n        result.append(i * j)",
            "# Optimized with list comprehension\nresult = [i * j for i in range(1000) for j in range(1000)]"
        ],
        "rule_type": "optimization",
        "use_collective_memory": True
    })

    return {
        "complexity_metrics": structure.get("complexity_metrics", {}),
        "performance_issues": perf_analysis.get("performance_issues", []),
        "optimization_suggestions": optimizations.get("rule", ""),
        "expected_improvement": perf_analysis.get("expected_improvement", "Unknown")
    }

# Usage example
perf_result = await analyze_performance_bottlenecks("./src/auth/authorization.py")
print(f"Found {len(perf_result.get('performance_issues', []))} performance issues")
```

### 3. Code Quality Issues Diagnosis
```python
async def diagnose_quality_issues(project_path):
    """Comprehensive quality diagnosis via MCP"""

    # 1. Find Python files to analyze
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Read sample files for analysis
    sample_files = python_files.get("files", [])[:5]  # Analyze first 5 files
    file_contents = await mcp_client.call_tool("read_multiple_files", {
        "paths": sample_files
    })

    # 3. Security analysis
    security_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "security",
        "use_collective_memory": True
    })

    # 4. Code smell detection
    code_smells = await mcp_client.call_tool("llm_search_pattern", {
        "query": "identify code smells and anti-patterns in python code",
        "language": "python",
        "path": project_path,
        "use_collective_memory": True
    })

    # 5. Maintainability assessment
    maintainability = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "maintainability",
        "use_collective_memory": True
    })

    return {
        "files_analyzed": len(sample_files),
        "security_issues": security_analysis.get("security_issues", []),
        "code_smells": code_smells.get("matches", []),
        "maintainability_score": maintainability.get("maintainability_score", 0),
        "improvement_priorities": maintainability.get("improvement_priorities", [])
    }

# Usage example
quality_result = await diagnose_quality_issues("./src")
print(f"Analyzed {quality_result.get('files_analyzed', 0)} files")
print(f"Maintainability score: {quality_result.get('maintainability_score', 0)}")
```

---

## Refactoring via MCP

### 1. Safe Function Refactoring
```python
async def safe_refactor_function(file_path, function_name, new_implementation):
    """Safely refactor a function via MCP"""

    # 1. Find all references to the function
    references = await mcp_client.call_tool("find_references", {
        "symbol": function_name,
        "path": "./",
        "symbol_type": "function"
    })

    print(f"Refactoring will affect {len(references.get('references', []))} files")

    # 2. Generate impact analysis
    impact_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": new_implementation,
        "language": "python",
        "analysis_type": "refactor_impact",
        "use_collective_memory": True
    })

    # 3. Dry run preview for main file
    preview = await mcp_client.call_tool("dry_run_edit_file", {
        "target_file": file_path,
        "instruction": f"Refactor function {function_name}",
        "code_edit": new_implementation
    })

    if preview.get("success"):
        print(f"Preview for {file_path}:")
        print(preview.get("preview", "No preview available"))
        print("---")

        # 4. Execute refactoring on main file
        result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": f"Refactor function {function_name}",
            "code_edit": new_implementation,
            "force": False
        })

        # 5. Update dependent files if needed
        if result.get("success") and len(references.get("references", [])) > 1:
            await update_dependent_files(references.get("references", []), function_name)

        return {
            "success": result.get("success", False),
            "files_affected": len(references.get("references", [])),
            "impact_analysis": impact_analysis
        }
    else:
        return {"success": False, "error": "Dry run failed"}

async def update_dependent_files(references, function_name):
    """Update files that reference the refactored function"""
    for ref in references[1:]:  # Skip the main file
        if ref.get("file"):
            update_result = await mcp_client.call_tool("edit_file", {
                "target_file": ref["file"],
                "instruction": f"Update calls to {function_name} after refactoring",
                "code_edit": f"# Update calls to {function_name} as needed",
                "force": False
            })
            print(f"Updated {ref['file']}: {update_result.get('success', False)}")
```

### 2. Class Structure Refactoring
```python
async def refactor_class_structure(file_path, class_name, refactoring_plan):
    """Refactor class structure via MCP"""

    # 1. Analyze current class structure
    structure = await mcp_client.call_tool("analyze_code_structure", {
        "file_path": file_path
    })

    # 2. Generate new class structure
    new_class = await generate_refactored_class(structure, refactoring_plan)

    # 3. Validate refactoring
    validation = await mcp_client.call_tool("llm_analyze_code", {
        "code": new_class,
        "language": "python",
        "analysis_type": "validation",
        "use_collective_memory": True
    })

    # 4. Execute refactoring if validation passes
    if validation.get("validation_passed", False):
        result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": f"Refactor class {class_name} with improved structure",
            "code_edit": new_class,
            "force": False
        })

        return {
            "success": result.get("success", False),
            "validation": validation,
            "refactored_class": class_name
        }
    else:
        return {
            "success": False,
            "error": "Validation failed",
            "validation_issues": validation.get("issues", [])
        }
```

### 3. Module Restructuring
```python
async def reorganize_module_structure(module_path, new_structure):
    """Reorganize module file structure via MCP"""

    # 1. Analyze current structure
    current_files = await mcp_client.call_tool("search_files", {
        "path": module_path,
        "pattern": "*.py"
    })

    # 2. Create restructuring plan
    restructuring_plan = await create_restructuring_plan(
        current_files.get("files", []),
        new_structure
    )

    # 3. Execute reorganization
    executed_changes = []
    for plan_item in restructuring_plan:
        if plan_item["action"] == "move":
            # Create new directory structure
            new_dir = os.path.dirname(plan_item["target_path"])
            if new_dir and not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Move file (this would be done via file system operations)
            print(f"Moving {plan_item['source_path']} -> {plan_item['target_path']}")
            executed_changes.append(plan_item)

        # Update imports in related files
        await update_imports_after_restructure(plan_item)

    return {
        "success": True,
        "restructured_files": len(executed_changes),
        "changes": executed_changes
    }
```

---

## Security & Quality via MCP

### 1. Comprehensive Security Audit
```python
async def comprehensive_security_audit(project_path):
    """Perform comprehensive security audit via MCP"""

    # 1. Find Python files to analyze
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Read sample files for security analysis
    sample_files = python_files.get("files", [])[:10]
    file_contents = await mcp_client.call_tool("read_multiple_files", {
        "paths": sample_files
    })

    # 3. Security vulnerability scan
    security_scan = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "security",
        "use_collective_memory": True
    })

    # 4. Generate security recommendations
    security_recommendations = await mcp_client.call_tool("llm_generate_rule", {
        "query": "security vulnerability fixes and best practices for python code",
        "language": "python",
        "examples": security_scan.get("vulnerabilities", []),
        "rule_type": "security_fix",
        "use_collective_memory": True
    })

    # 5. Compliance checking
    compliance_report = await generate_compliance_report(security_scan)

    return {
        "files_analyzed": len(sample_files),
        "security_issues": security_scan.get("security_issues", []),
        "vulnerabilities": security_scan.get("vulnerabilities", []),
        "recommendations": security_recommendations.get("rule", ""),
        "compliance_report": compliance_report,
        "risk_score": calculate_security_risk_score(security_scan)
    }

# Usage example
security_result = await comprehensive_security_audit("./src")
print(f"Found {len(security_result.get('security_issues', []))} security issues")
print(f"Risk score: {security_result.get('risk_score', 0)}")
```

### 2. Code Quality Assessment
```python
async def assess_code_quality(project_path):
    """Comprehensive code quality assessment via MCP"""

    # 1. Find Python files
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Read sample files for analysis
    sample_files = python_files.get("files", [])[:8]
    file_contents = await mcp_client.call_tool("read_multiple_files", {
        "paths": sample_files
    })

    # 3. Quality metrics analysis
    quality_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "quality",
        "use_collective_memory": True
    })

    # 4. Documentation analysis
    doc_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "documentation",
        "use_collective_memory": True
    })

    # 5. Maintainability assessment
    maintainability = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "maintainability",
        "use_collective_memory": True
    })

    # 6. Generate improvement recommendations
    improvements = await mcp_client.call_tool("llm_search_pattern", {
        "query": "code quality improvements and best practices for python",
        "language": "python",
        "path": project_path,
        "use_collective_memory": True
    })

    return {
        "files_analyzed": len(sample_files),
        "quality_metrics": quality_analysis.get("quality_metrics", {}),
        "documentation_score": doc_analysis.get("documentation_score", 0),
        "maintainability_score": maintainability.get("maintainability_score", 0),
        "overall_quality_grade": calculate_quality_grade(maintainability.get("maintainability_score", 0)),
        "improvement_recommendations": improvements.get("matches", []),
        "critical_issues": quality_analysis.get("critical_issues", [])
    }

# Usage example
quality_result = await assess_code_quality("./src")
print(f"Quality grade: {quality_result.get('overall_quality_grade', 'Unknown')}")
print(f"Maintainability score: {quality_result.get('maintainability_score', 0)}")
```

---

## Performance Optimization via MCP

### 1. Performance Profiling
```python
async def profile_code_performance(file_path):
    """Profile and analyze code performance via MCP"""

    # 1. Structural complexity analysis
    structure = await mcp_client.call_tool("analyze_code_structure", {
        "file_path": file_path
    })

    # 2. Read file content
    file_content = await mcp_client.call_tool("read_multiple_files", {
        "paths": [file_path]
    })

    # 3. Performance bottleneck identification
    bottlenecks = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_content,
        "language": "python",
        "analysis_type": "performance",
        "use_collective_memory": True
    })

    # 4. Generate optimization suggestions
    optimizations = await mcp_client.call_tool("llm_generate_rule", {
        "query": "performance optimizations and best practices for python code",
        "language": "python",
        "examples": [
            "# Inefficient database queries\nfor item in items:\n    db.query(f'SELECT * FROM table WHERE id = {item.id}')",
            "# Optimized bulk query\nids = [item.id for item in items]\ndb.query(f'SELECT * FROM table WHERE id IN ({','.join(map(str, ids))})')"
        ],
        "rule_type": "optimization",
        "use_collective_memory": True
    })

    # 5. Generate optimization plan
    optimization_plan = await create_optimization_plan(
        bottlenecks.get("performance_bottlenecks", []),
        optimizations.get("rule", "")
    )

    return {
        "complexity_metrics": structure.get("complexity_metrics", {}),
        "bottlenecks": bottlenecks.get("performance_bottlenecks", []),
        "optimizations": optimizations.get("rule", ""),
        "optimization_plan": optimization_plan,
        "expected_improvement": bottlenecks.get("expected_improvement", "Unknown")
    }

# Usage example
perf_result = await profile_code_performance("./src/auth/authorization.py")
print(f"Found {len(perf_result.get('bottlenecks', []))} performance bottlenecks")
print(f"Expected improvement: {perf_result.get('expected_improvement', 'Unknown')}")
```

### 2. Memory Usage Optimization
```python
async def optimize_memory_usage(project_path):
    """Analyze and optimize memory usage via MCP"""

    # 1. Find Python files
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Read sample files for memory analysis
    sample_files = python_files.get("files", [])[:5]
    file_contents = await mcp_client.call_tool("read_multiple_files", {
        "paths": sample_files
    })

    # 3. Memory pattern analysis
    memory_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": file_contents,
        "language": "python",
        "analysis_type": "memory",
        "use_collective_memory": True
    })

    # 4. Memory leak detection
    memory_leaks = await mcp_client.call_tool("llm_search_pattern", {
        "query": "memory leaks and inefficient memory usage patterns in python",
        "language": "python",
        "path": project_path,
        "use_collective_memory": True
    })

    # 5. Memory optimization suggestions
    optimizations = await mcp_client.call_tool("llm_generate_rule", {
        "query": "memory optimization techniques for python code",
        "language": "python",
        "examples": [
            "# Memory inefficient\nlarge_list = [process_item(item) for item in huge_dataset]",
            "# Memory efficient with generators\nfor item in huge_dataset:\n    yield process_item(item)"
        ],
        "rule_type": "memory_optimization",
        "use_collective_memory": True
    })

    return {
        "files_analyzed": len(sample_files),
        "memory_analysis": memory_analysis.get("memory_analysis", {}),
        "memory_leaks": memory_leaks.get("matches", []),
        "optimizations": optimizations.get("rule", ""),
        "memory_saving_potential": memory_analysis.get("memory_saving_potential", "Unknown")
    }

# Usage example
memory_result = await optimize_memory_usage("./src")
print(f"Memory saving potential: {memory_result.get('memory_saving_potential', 'Unknown')}")
```

---

## Advanced Patterns via MCP

### 1. Intelligent Code Generation
```python
async def generate_intelligent_code(requirements, context, language="python"):
    """Generate code based on requirements via MCP"""

    # 1. Analyze requirements
    requirements_analysis = await mcp_client.call_tool("llm_analyze_code", {
        "code": requirements,
        "language": "natural_language",
        "analysis_type": "requirements",
        "use_collective_memory": True
    })

    # 2. Find similar patterns
    similar_patterns = await mcp_client.call_tool("qdrant-find", {
        "query": f"code generation patterns for {requirements_analysis.get('domain', 'general')}",
        "limit": 10
    })

    # 3. Generate code based on patterns
    generated_code = await mcp_client.call_tool("llm_generate_rule", {
        "query": requirements,
        "language": language,
        "examples": [pattern.get("content", "") for pattern in similar_patterns],
        "rule_type": "code_generation",
        "use_collective_memory": True
    })

    # 4. Validate generated code
    validation = await mcp_client.call_tool("llm_analyze_code", {
        "code": generated_code.get("rule", ""),
        "language": language,
        "analysis_type": "validation",
        "use_collective_memory": True
    })

    return {
        "generated_code": generated_code.get("rule", ""),
        "validation": validation,
        "confidence_score": validation.get("confidence_score", 0),
        "suggested_improvements": validation.get("improvements", []),
        "requirements_analysis": requirements_analysis
    }

# Usage example
code_result = await generate_intelligent_code(
    "Create a REST API endpoint for user authentication with JWT tokens",
    "web application backend",
    "python"
)

print(f"Generated code confidence: {code_result.get('confidence_score', 0)}")
print(f"Code length: {len(code_result.get('generated_code', ''))}")
```

### 2. Collective Learning Integration
```python
async def learn_from_success_patterns(successful_operations):
    """Learn and store successful patterns via MCP"""

    stored_patterns = []
    for operation in successful_operations:
        pattern_data = {
            "operation_type": operation["type"],
            "problem_description": operation["problem"],
            "solution": operation["solution"],
            "success_metrics": operation["metrics"],
            "language": operation.get("language", "python"),
            "context": operation.get("context", {}),
            "timestamp": datetime.now().isoformat()
        }

        # Store in collective memory
        await mcp_client.call_tool("qdrant-store", {
            "information": json.dumps(pattern_data),
            "metadata": {
                "type": "success_pattern",
                "operation": operation["type"],
                "language": pattern_data["language"],
                "success_rate": operation["metrics"].get("success_rate", 1.0),
                "timestamp": pattern_data["timestamp"]
            }
        })

        stored_patterns.append(pattern_data)
        print(f"Stored successful pattern: {operation['type']}")

    return {"patterns_stored": len(stored_patterns)}

# Usage example
success_ops = [
    {
        "type": "authentication_refactor",
        "problem": "Insecure password handling",
        "solution": "Implemented bcrypt hashing with salt rounds",
        "metrics": {"success_rate": 1.0, "performance_improvement": "15%"},
        "language": "python",
        "context": {"project": "web_app", "module": "auth"}
    }
]

learning_result = await learn_from_success_patterns(success_ops)
print(f"Stored {learning_result.get('patterns_stored', 0)} patterns")
```

### 3. Automated Testing Integration
```python
async def generate_test_cases(code_file, test_framework="pytest"):
    """Generate comprehensive test cases via MCP"""

    # 1. Analyze code structure
    structure = await mcp_client.call_tool("analyze_code_structure", {
        "file_path": code_file
    })

    # 2. Read file content
    file_content = await mcp_client.call_tool("read_multiple_files", {
        "paths": [code_file]
    })

    # 3. Generate test cases
    test_cases = await mcp_client.call_tool("llm_generate_rule", {
        "query": f"generate {test_framework} test cases for python code",
        "language": "python",
        "examples": [
            "def test_addition():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0",
            "def test_edge_cases():\n    assert divide(10, 2) == 5\n    with pytest.raises(ZeroDivisionError):\n        divide(10, 0)"
        ],
        "rule_type": "test_generation",
        "use_collective_memory": True
    })

    # 4. Create test file content
    test_content = await generate_test_file_content(
        structure,
        test_cases.get("rule", ""),
        test_framework
    )

    # 5. Write test file
    test_file_path = code_file.replace(".py", "_test.py")
    result = await mcp_client.call_tool("edit_file", {
        "target_file": test_file_path,
        "instruction": f"Generate {test_framework} test cases",
        "code_edit": test_content,
        "force": True
    })

    return {
        "test_file": test_file_path,
        "test_cases_generated": len(structure.get("functions", [])),
        "result": result
    }

# Usage example
test_result = await generate_test_cases("./src/auth/authorization.py")
print(f"Generated test file: {test_result.get('test_file', 'Unknown')}")
print(f"Test cases generated: {test_result.get('test_cases_generated', 0)}")
```

---

## Best Practices via MCP

### 1. Code Quality Best Practices
```python
async def apply_quality_best_practices(project_path):
    """Apply code quality best practices via MCP"""

    # 1. Find Python files
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Generate type hints rule
    type_hints_rule = await mcp_client.call_tool("llm_generate_rule", {
        "query": "add comprehensive type hints to functions and variables",
        "language": "python",
        "examples": [
            "def calculate_total(items):",
            "def calculate_total(items: List[Dict[str, Any]]) -> float:"
        ],
        "rule_type": "type_hints",
        "use_collective_memory": True
    })

    # 3. Generate error handling rule
    error_handling_rule = await mcp_client.call_tool("llm_generate_rule", {
        "query": "add specific exception handling and proper error messages",
        "language": "python",
        "examples": [
            "try:\n    result = risky_operation()\nexcept:\n    return None",
            "try:\n    result = risky_operation()\nexcept ValueError as e:\n    logger.error(f'Invalid value: {e}')\n    raise\nexcept Exception as e:\n    logger.error(f'Unexpected error: {e}')\n    raise"
        ],
        "rule_type": "error_handling",
        "use_collective_memory": True
    })

    # 4. Generate documentation rule
    documentation_rule = await mcp_client.call_tool("llm_generate_rule", {
        "query": "add comprehensive docstrings with parameter descriptions",
        "language": "python",
        "examples": [
            "def process_data(data):",
            "def process_data(data: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"Process input data and return transformed result.\n    \n    Args:\n        data: Input data dictionary\n        \n    Returns:\n        Transformed data dictionary\n        \n    Raises:\n        ValueError: If input data is invalid\n    \"\"\""
        ],
        "rule_type": "documentation",
        "use_collective_memory": True
    })

    # 5. Apply rules to files
    applied_practices = []
    for file_path in python_files.get("files", []):
        # Apply type hints
        type_result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": "Add comprehensive type hints",
            "code_edit": type_hints_rule.get("rule", ""),
            "force": False
        })

        if type_result.get("success"):
            applied_practices.append({
                "practice": "type_hints",
                "file": file_path,
                "status": "applied"
            })

        # Apply error handling
        error_result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": "Improve error handling",
            "code_edit": error_handling_rule.get("rule", ""),
            "force": False
        })

        if error_result.get("success"):
            applied_practices.append({
                "practice": "error_handling",
                "file": file_path,
                "status": "applied"
            })

        # Apply documentation
        doc_result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": "Improve documentation",
            "code_edit": documentation_rule.get("rule", ""),
            "force": False
        })

        if doc_result.get("success"):
            applied_practices.append({
                "practice": "documentation",
                "file": file_path,
                "status": "applied"
            })

    return {"applied_practices": applied_practices}

# Usage example
quality_result = await apply_quality_best_practices("./src")
print(f"Applied {len(quality_result.get('applied_practices', []))} quality improvements")
```

### 2. Security Best Practices
```python
async def apply_security_best_practices(project_path):
    """Apply security best practices via MCP"""

    # 1. Find Python files
    python_files = await mcp_client.call_tool("search_files", {
        "path": project_path,
        "pattern": "*.py",
        "excludePatterns": ["__pycache__", "*.pyc"]
    })

    # 2. Generate input validation rule
    input_validation_rule = await mcp_client.call_tool("llm_generate_rule", {
        "query": "add input validation and sanitization for security",
        "language": "python",
        "examples": [
            "def process_user_input(user_input):\n    return user_input.upper()",
            "def process_user_input(user_input: str) -> str:\n    if not isinstance(user_input, str):\n        raise TypeError('Input must be string')\n    if len(user_input) > 1000:\n        raise ValueError('Input too long')\n    sanitized = re.sub(r'[<>\"&\\']', '', user_input)\n    return sanitized.upper()"
        ],
        "rule_type": "input_validation",
        "use_collective_memory": True
    })

    # 3. Generate SQL injection prevention rule
    sql_injection_rule = await mcp_client.call_tool("llm_generate_rule", {
        "query": "convert raw SQL queries to parameterized queries",
        "language": "python",
        "examples": [
            "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
            "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
        ],
        "rule_type": "sql_injection_prevention",
        "use_collective_memory": True
    })

    # 4. Apply security rules
    applied_practices = []
    for file_path in python_files.get("files", []):
        # Apply input validation
        validation_result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": "Add input validation and sanitization",
            "code_edit": input_validation_rule.get("rule", ""),
            "force": False
        })

        if validation_result.get("success"):
            applied_practices.append({
                "practice": "input_validation",
                "file": file_path,
                "status": "applied"
            })

        # Apply SQL injection prevention
        sql_result = await mcp_client.call_tool("edit_file", {
            "target_file": file_path,
            "instruction": "Prevent SQL injection with parameterized queries",
            "code_edit": sql_injection_rule.get("rule", ""),
            "force": False
        })

        if sql_result.get("success"):
            applied_practices.append({
                "practice": "sql_injection_prevention",
                "file": file_path,
                "status": "applied"
            })

    return {"applied_practices": applied_practices}

# Usage example
security_result = await apply_security_best_practices("./src")
print(f"Applied {len(security_result.get('applied_practices', []))} security improvements")
```

---

## MCP Tool Reference

### Core Tools

#### `auto_ast_intelligence`
```python
result = await mcp_client.call_tool("auto_ast_intelligence", {
    "query": "analyze authentication patterns",
    "context": "security review",
    "language": "python",
    "path": "./src/auth",
    "auto_execute": True
})
```

#### `edit_file`
```python
result = await mcp_client.call_tool("edit_file", {
    "target_file": "./src/auth/authorization.py",
    "instruction": "Add input validation",
    "code_edit": "# Add validation code here",
    "force": False,
    "output_format": "json"
})
```

#### `dry_run_edit_file`
```python
result = await mcp_client.call_tool("dry_run_edit_file", {
    "target_file": "./src/auth/authorization.py",
    "instruction": "Preview function refactoring",
    "code_edit": "# Proposed changes here",
    "output_format": "json"
})
```

#### `search_files`
```python
result = await mcp_client.call_tool("search_files", {
    "path": "./src",
    "pattern": "*.py",
    "excludePatterns": ["__pycache__", "*.pyc"]
})
```

#### `analyze_code_structure`
```python
result = await mcp_client.call_tool("analyze_code_structure", {
    "file_path": "./src/main.py"
})
```

#### `find_references`
```python
result = await mcp_client.call_tool("find_references", {
    "symbol": "User",
    "path": "./src",
    "symbol_type": "class"
})
```

### Analysis Tools

#### `llm_analyze_code`
```python
result = await mcp_client.call_tool("llm_analyze_code", {
    "code": "your code here",
    "language": "python",
    "analysis_type": "comprehensive",
    "use_collective_memory": True
})
```

#### `llm_generate_rule`
```python
result = await mcp_client.call_tool("llm_generate_rule", {
    "query": "generate authentication patterns",
    "language": "python",
    "examples": ["example code here"],
    "rule_type": "pattern",
    "use_collective_memory": True
})
```

#### `llm_search_pattern`
```python
result = await mcp_client.call_tool("llm_search_pattern", {
    "query": "find authentication middleware",
    "language": "python",
    "path": "./src",
    "use_collective_memory": True,
    "max_results": 10
})
```

### Memory Tools

#### `qdrant-store`
```python
await mcp_client.call_tool("qdrant-store", {
    "information": "your content here",
    "metadata": {
        "type": "pattern",
        "language": "python",
        "project": "my-project"
    }
})
```

#### `qdrant-find`
```python
result = await mcp_client.call_tool("qdrant-find", {
    "query": "authentication patterns python",
    "limit": 5
})
```

### Utility Tools

#### `read_multiple_files`
```python
result = await mcp_client.call_tool("read_multiple_files", {
    "paths": ["./src/main.py", "./src/auth.py"]
})
```

#### `call_tool`
```python
result = await mcp_client.call_tool("call_tool", {
    "name": "edit_file",
    "arguments": {
        "target_file": "./src/main.py",
        "instruction": "Add logging",
        "code_edit": "# Add logging code"
    }
})
```

---

## Conclusion

This MCP-focused workflow documentation provides comprehensive examples for using FastApply through the Model Context Protocol interface. The key advantages of using MCP include:

### Key Benefits

1. **Standardized Interface**: All interactions use consistent MCP tool calls
2. **Language Agnostic**: Works with any programming language that supports MCP clients
3. **Remote Execution**: Can run FastApply on remote servers or containers
4. **Integration Ready**: Easily integrates with existing MCP-compatible tools and platforms
5. **Stateless Operations**: Each tool call is independent and self-contained
6. **Error Handling**: Comprehensive error responses and status codes

### MCP vs Direct API Comparison

| Feature | Direct API | MCP Interface |
|---------|------------|---------------|
| **Setup** | Direct Python import | MCP server configuration |
| **Language** | Python only | Any MCP-compatible language |
| **Deployment** | Local only | Remote/Container/Cloud |
| **Integration** | Custom integration needed | Standard MCP integration |
| **Error Handling** | Python exceptions | MCP error responses |
| **State Management** | Manual | Stateless operations |
| **Scalability** | Limited by local resources | Scales with MCP infrastructure |

### Best Practices for MCP Usage

1. **Always check tool availability** before calling specific tools
2. **Use dry_run_edit_file** for previewing changes
3. **Handle errors gracefully** with proper error checking
4. **Batch operations** when possible for better performance
5. **Use collective memory** through qdrant tools for pattern learning
6. **Validate results** after each operation
7. **Store context** using qdrant for session persistence

By following these MCP workflows, development teams can leverage FastApply's powerful code intelligence capabilities through a standardized, language-agnostic interface that integrates seamlessly with modern development environments and tools.