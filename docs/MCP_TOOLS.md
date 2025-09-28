# FastApply MCP Tools Documentation

**Last Updated**: 2025-09-27
**Total Tools**: 44 MCP Tools
**Categories**: 8 Functional Categories

---

## 📋 Table of Contents

1. [Core File Operations](#core-file-operations)
2. [Code Search & Pattern Matching](#code-search--pattern-matching)
3. [AST Intelligence & Analysis](#ast-intelligence--analysis)
4. [Ripgrep Integration](#ripgrep-integration)
5. [Enhanced Search Infrastructure](#enhanced-search-infrastructure)
6. [Semantic Analysis & Relationship Mapping](#semantic-analysis--relationship-mapping)
7. [Safe Refactoring Tools](#safe-refactoring-tools)
8. [Batch Operations](#batch-operations)
9. [System & Health Monitoring](#system--health-monitoring)

---

## 🗂️ Core File Operations

### `edit_file`
Apply code edits to a file using Fast Apply with intelligent pattern matching and safety validation.

**Parameters**:
- `target_file` (required): Path to target file
- `instructions` (required): Edit instructions
- `code_edit` (required): Code edit snippet
- `force` (optional): Override safety checks
- `output_format` (optional): Response format (text/json)

### `dry_run_edit_file`
Preview an edit without writing changes to test transformations safely.

**Parameters**:
- `target_file` (required): Target file path
- `instruction` (optional): Edit instructions
- `code_edit` (required): Code to edit
- `output_format` (optional): Response format

### `read_multiple_files`
Read multiple small files with concatenated output for efficient batch processing.

**Parameters**:
- `paths` (required): Array of file paths to read

---

## 🔍 Code Search & Pattern Matching

### `search_files`
Search for files by simple substring pattern with flexible filtering options.

**Parameters**:
- `pattern` (required): Search pattern
- `path` (optional): Search path
- `excludePatterns` (optional): Patterns to exclude

### `search_code_patterns`
Search for semantic code patterns using AST-based matching (e.g., 'function $name($args) { $body }').

**Parameters**:
- `pattern` (required): AST pattern to search for
- `language` (required): Target language (python, javascript, typescript)
- `path` (required): File or directory path
- `exclude_patterns` (optional): Patterns to exclude

### `analyze_code_structure`
Analyze structural components (functions, classes, imports) of a code file.

**Parameters**:
- `file_path` (required): Path to the code file to analyze

### `find_references`
Find all references to a specific symbol (function, class, variable) across the codebase.

**Parameters**:
- `symbol` (required): Symbol name to search for
- `path` (required): Search path
- `symbol_type` (optional): Type of symbol (function, class, variable, any)

---

## 🧠 AST Intelligence & Analysis

### `dump_syntax_tree`
Dump syntax tree for code using ast-grep (official tool) for deep code understanding.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (required): Programming language
- `dump_format` (optional): Output format (pattern/cst/ast)

### `test_match_code_rule`
Test YAML rule against code using ast-grep for rule validation and debugging.

**Parameters**:
- `code` (required): Source code to test
- `rule_yaml` (required): YAML rule definition

### `find_code`
Find code using ast-grep patterns with official tool integration.

**Parameters**:
- `pattern` (required): Search pattern
- `language` (required): Programming language
- `path` (optional): Directory to search
- `output_format` (optional): Output format
- `max_results` (optional): Maximum results

### `find_code_by_rule`
Find code using YAML rules with ast-grep for complex pattern matching.

**Parameters**:
- `rule_yaml` (required): YAML rule definition
- `path` (optional): Directory to search
- `output_format` (optional): Output format
- `max_results` (optional): Maximum results

### `llm_analyze_code`
LLM-based deep code analysis using AST intelligence and reasoning patterns.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (required): Programming language
- `analysis_type` (optional): Analysis type (general, complexity, security, performance, architecture)
- `use_collective_memory` (optional): Use Qdrant collective consciousness

### `llm_generate_rule`
Generate AST rules dynamically using LLM reasoning from examples and queries.

**Parameters**:
- `query` (required): Natural language description of the rule
- `language` (required): Target programming language
- `examples` (optional): Example code snippets
- `rule_type` (optional): Rule type (pattern, relational, composite)
- `use_collective_memory` (optional): Use learned patterns

### `llm_search_pattern`
Intelligent pattern search using LLM reasoning and collective consciousness.

**Parameters**:
- `query` (required): Natural language search query
- `language` (required): Programming language
- `path` (optional): Search path
- `use_collective_memory` (optional): Use collective consciousness
- `max_results` (optional): Maximum results

### `auto_ast_intelligence`
Auto-invocation system that detects user intent and executes optimal AST intelligence tools.

**Parameters**:
- `query` (required): Natural language query or command
- `context` (optional): Task context
- `language` (optional): Default programming language
- `path` (optional): Default path for operations
- `auto_execute` (optional): Auto-execute detected tools

---

## ⚡ Ripgrep Integration

### `ripgrep_search`
Ultra-fast pattern discovery using ripgrep for large codebases with advanced filtering.

**Parameters**:
- `pattern` (required): Search pattern
- `path` (required): Search path
- `search_type` (optional): Search type (pattern, literal, word, regex)
- `case_sensitive` (optional): Case sensitivity
- `include_patterns` (optional): File patterns to include
- `exclude_patterns` (optional): File patterns to exclude
- `max_results` (optional): Maximum results
- `context_lines` (optional): Context lines
- `file_types` (optional): Programming language types
- `max_filesize` (optional): Maximum file size
- `max_depth` (optional): Directory depth
- `follow_symlinks` (optional): Follow symbolic links
- `output_format` (optional): Output format

### `ripgrep_search_code`
Search for code patterns specific to a programming language using ripgrep.

**Parameters**:
- `pattern` (required): Code pattern
- `language` (required): Programming language
- `path` (required): Search path
- Plus all ripgrep_search filtering options

### `ripgrep_find_symbols`
Find potential symbol candidates using ripgrep pattern matching with context.

**Parameters**:
- `symbol_name` (required): Symbol name to find
- `path` (required): Search path
- `symbol_type` (optional): Symbol type (function, class, variable, any)
- `language` (optional): Programming language
- `max_results` (optional): Maximum results
- `context_lines` (optional): Context lines

### `ripgrep_file_metrics`
Analyze file metrics using ripgrep and system tools for comprehensive insights.

**Parameters**:
- `file_path` (required): Path to the file to analyze

### `ripgrep_batch_search`
Perform multiple ripgrep searches concurrently for efficient batch processing.

**Parameters**:
- `patterns` (required): List of patterns to search
- `path` (required): Search path
- Plus concurrent search configuration options

---

## 🔎 Enhanced Search Infrastructure

### `enhanced_search`
Intelligent multi-strategy search combining ripgrep speed with semantic understanding.

**Parameters**:
- `query` (required): Search query
- `path` (required): Search path
- `strategy` (optional): Search strategy (exact, fuzzy, semantic, hybrid)
- `file_types` (optional): Programming language types
- `exclude_patterns` (optional): Patterns to exclude
- `include_patterns` (optional): Patterns to include
- `max_results` (optional): Maximum results
- `context_lines` (optional): Context lines
- `case_sensitive` (optional): Case sensitivity
- `ranking` (optional): Result ranking strategy
- `timeout` (optional): Search timeout

### `enhanced_search_intelligent`
Smart intent-aware search that auto-detects optimal strategy and parameters.

**Parameters**:
- `query` (required): Natural language search query
- `path` (required): Search path
- `context` (optional): Search context
- `auto_detect_strategy` (optional): Auto-detect strategy
- `language` (optional): Preferred language
- `max_results` (optional): Maximum results
- `optimize_for` (optional): Optimization goal

### `search_with_context`
Context-preserving search that maintains state across multiple operations.

**Parameters**:
- `query` (required): Search query
- `path` (required): Search path
- `context_id` (required): Unique context identifier
- `previous_results` (optional): Previous results for context
- `refine` (optional): Refine previous results
- `incremental` (optional): Incremental search mode
- `max_results` (optional): Maximum results

### `search_performance_optimize`
Optimize search performance through pattern analysis and cache management.

**Parameters**:
- `action` (required): Optimization action (analyze_patterns, optimize_cache, tune_pipeline, benchmark)
- `path` (optional): Path for pattern analysis
- `common_patterns` (optional): Patterns to optimize for
- `cache_size` (optional): Desired cache size
- `clear_cache` (optional): Clear existing cache

---

## 🧩 Semantic Analysis & Relationship Mapping

### `deep_semantic_analysis`
Perform deep semantic analysis of code including intent, behavior, patterns, and quality.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language
- `analysis_depth` (optional): Analysis depth (basic, comprehensive, deep)
- `include_patterns` (optional): Include design pattern detection
- `include_quality` (optional): Include quality assessment
- `context` (optional): Additional context

### `understand_code_intent`
Analyze the intent and purpose of code with confidence scoring.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language
- `context` (optional): Additional context

### `analyze_runtime_behavior`
Analyze runtime behavior patterns, side effects, and performance characteristics.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language

### `identify_design_patterns`
Identify design patterns and anti-patterns in code with improvement suggestions.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language
- `include_suggestions` (optional): Include improvement suggestions

### `assess_code_quality`
Comprehensive code quality assessment across multiple dimensions with recommendations.

**Parameters**:
- `code` (required): Source code to assess
- `language` (optional): Programming language
- `include_recommendations` (optional): Include improvement recommendations

### `map_relationships`
Map code relationships and architectural dependencies for comprehensive understanding.

**Parameters**:
- `code` (required): Source code to analyze
- `context` (optional): File path or context
- `project_path` (optional): Root project path
- `include_dependencies` (optional): Include dependency analysis
- `include_coupling` (optional): Include coupling analysis
- `include_cohesion` (optional): Include cohesion analysis

### `detect_circular_dependencies`
Detect circular dependencies and provide resolution suggestions for architectural health.

**Parameters**:
- `project_path` (required): Project root path
- `include_impact` (optional): Include impact analysis
- `include_suggestions` (optional): Include resolution suggestions

### `analyze_coupling_cohesion`
Analyze module coupling and cohesion for architectural insights and improvements.

**Parameters**:
- `project_path` (required): Project root path
- `threshold` (optional): Threshold for flagging issues
- `include_recommendations` (optional): Include improvement recommendations

### `map_control_flow`
Map control flow within code for execution path analysis and complexity assessment.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language
- `include_complexity` (optional): Include complexity metrics

### `analyze_data_flow`
Analyze data flow patterns and variable dependencies for comprehensive understanding.

**Parameters**:
- `code` (required): Source code to analyze
- `language` (optional): Programming language
- `include_tracking` (optional): Include variable tracking

---

## 🛡️ Safe Refactoring Tools

### `safe_rename_symbol`
Safely rename a symbol with automatic reference updating and rollback capability.

**Parameters**:
- `old_name` (required): Current symbol name
- `new_name` (required): New symbol name
- `symbol_type` (optional): Type of symbol
- `scope` (optional): Optional scope/context
- `project_path` (optional): Project root path

### `analyze_rename_impact`
Analyze the impact and risks of renaming a symbol before performing the operation.

**Parameters**:
- `old_name` (required): Current symbol name
- `new_name` (required): New symbol name
- `symbol_type` (optional): Type of symbol
- `scope` (optional): Optional scope/context
- `project_path` (optional): Project root path

### `safe_extract_function`
Safely extract a function to a separate file with dependency management.

**Parameters**:
- `source_range` (required): Source range [start_line, end_line]
- `function_name` (required): Function name
- `target_file` (required): Target file path
- `source_file` (required): Source file path
- `project_path` (optional): Project root path

### `safe_move_symbol`
Safely move a symbol (function/class) to another file with reference updating.

**Parameters**:
- `symbol_name` (required): Symbol name to move
- `source_file` (required): Current file path
- `target_file` (required): Target file path
- `symbol_type` (optional): Symbol type
- `scope` (optional): Optional scope/context
- `project_path` (optional): Project root path

### `execute_rollback`
Execute a rollback operation for a previous refactoring change with full restoration.

**Parameters**:
- `operation_id` (required): Operation ID to rollback

---

## 📊 Batch Operations

### `batch_analyze_project`
Large-scale project analysis with batch processing for 1000+ files and comprehensive reporting.

**Parameters**:
- `project_path` (required): Project root path
- `analysis_types` (optional): Analysis types (complexity, dependencies, quality, security, performance)
- `max_workers` (optional): Maximum concurrent workers
- `timeout` (optional): Timeout in seconds
- `output_format` (optional): Output format

### `batch_transform_code`
Bulk code transformations with safety validation and rollback capability for large-scale refactoring.

**Parameters**:
- `transformation_type` (required): Transformation type (rename, extract, move, pattern_replace)
- `project_path` (required): Project root path
- `targets` (required): Target files, patterns, or symbols
- `parameters` (optional): Transformation-specific parameters
- `validation_level` (optional): Safety validation level
- `dry_run` (optional): Preview without applying
- `max_workers` (optional): Maximum concurrent workers

### `monitor_batch_progress`
Monitor progress of batch operations with real-time metrics and status updates.

**Parameters**:
- `operation_id` (required): Operation ID to monitor
- `include_details` (optional): Include detailed progress
- `include_metrics` (optional): Include performance metrics
- `include_errors` (optional): Include error information
- `refresh_interval` (optional): Refresh interval

### `schedule_batch_operations`
Schedule and manage batch operations with priority and resource management for complex workflows.

**Parameters**:
- `operations` (required): Operations to schedule with priorities
- `max_concurrent` (optional): Maximum concurrent operations
- `resource_limits` (optional): Resource constraints
- `schedule_mode` (optional): Scheduling mode

### `validate_batch_operations`
Validate batch operations for safety and potential conflicts before execution.

**Parameters**:
- `operation_plan` (required): Operation plan to validate
- `project_path` (required): Project root path
- `validation_level` (optional): Validation depth
- `include_performance` (optional): Include performance analysis
- `include_security` (optional): Include security analysis

### `execute_batch_rename`
Execute batch symbol renaming across multiple files with dependency tracking and reference updates.

**Parameters**:
- `rename_operations` (required): Rename operations to execute
- `project_path` (required): Project root path
- `update_references` (optional): Update references automatically
- `dry_run` (optional): Preview without applying
- `create_backups` (optional): Create backups

### `batch_extract_components`
Extract multiple components (functions, classes) to separate files with dependency management.

**Parameters**:
- `extractions` (required): Components to extract with metadata
- `project_path` (required): Project root path
- `manage_imports` (optional): Auto-manage imports
- `create_backups` (optional): Create backups
- `max_workers` (optional): Maximum concurrent extractions

### `generate_batch_report`
Generate comprehensive reports for batch operations with metrics, analysis, and recommendations.

**Parameters**:
- `operation_id` (required): Operation ID to report on
- `report_type` (optional): Report type (summary, detailed, executive, technical)
- `include_metrics` (optional): Include performance metrics
- `include_errors` (optional): Include error analysis
- `include_recommendations` (optional): Include improvement recommendations
- `output_format` (optional): Output format

---

## 🏥 System & Health Monitoring

### `health_status`
Return server health and configuration info (non-sensitive) for system monitoring and debugging.

**Parameters**: None required

---

## 📈 Tool Categories Summary

| Category | Tool Count | Purpose |
|----------|------------|---------|
| Core File Operations | 3 | Basic file editing and reading |
| Code Search & Pattern Matching | 4 | Finding code patterns and references |
| AST Intelligence & Analysis | 8 | Advanced code analysis and rule generation |
| Ripgrep Integration | 5 | Ultra-fast search capabilities |
| Enhanced Search Infrastructure | 4 | Intelligent search strategies |
| Semantic Analysis & Relationship Mapping | 10 | Deep code understanding and architecture |
| Safe Refactoring Tools | 5 | Secure code transformations |
| Batch Operations | 8 | Large-scale project operations |
| System & Health Monitoring | 1 | System status and diagnostics |
| **TOTAL** | **44** | **Comprehensive code analysis and transformation** |

---

## 🚀 Usage Examples

### Basic Code Search
```bash
# Find all references to a function
find_references --symbol "calculateTotal" --path "./src" --symbol_type "function"

# Search for code patterns
search_code_patterns --pattern "class $NAME extends React.Component" --language "javascript" --path "./components"
```

### Advanced Analysis
```bash
# Deep semantic analysis
deep_semantic_analysis --code "$(cat src/algorithm.js)" --language "javascript" --analysis_depth "comprehensive"

# Understand code intent
understand_code_intent --code "$(cat src/auth.py)" --language "python" --context "authentication module"
```

### Safe Refactoring
```bash
# Analyze rename impact before executing
analyze_rename_impact --old_name "getUserData" --new_name "fetchUserData" --symbol_type "function" --project_path "./src"

# Safely rename with rollback capability
safe_rename_symbol --old_name "getUserData" --new_name "fetchUserData" --symbol_type "function" --project_path "./src"
```

### Batch Operations
```bash
# Large-scale project analysis
batch_analyze_project --project_path "./src" --analysis_types ["complexity", "dependencies", "quality"] --max_workers 4

# Bulk transformations with safety validation
batch_transform_code --transformation_type "rename" --project_path "./src" --targets ["oldFunc1", "oldFunc2"] --parameters '{"new_names": ["newFunc1", "newFunc2"]}' --dry_run true
```

---

## 🔧 Integration Guide

### MCP Server Configuration
All tools are available through the FastApply MCP server. Use the `list_tools` endpoint to get real-time tool availability and schema information.

### Best Practices
1. **Start with analysis**: Use `analyze_code_structure` or `deep_semantic_analysis` before refactoring
2. **Validate changes**: Use `dry_run_edit_file` and `analyze_rename_impact` before applying changes
3. **Monitor operations**: Use `monitor_batch_progress` for long-running batch operations
4. **Safety first**: Always use rollback-capable tools for critical operations
5. **Batch efficiently**: Use `batch_analyze_project` for large-scale analysis

### Error Handling
All tools return structured error information and suggestions for recovery. Check tool responses for:
- Success/failure status
- Error messages and context
- Recovery suggestions
- Rollback operation IDs when applicable

---

*This documentation covers all 44 MCP tools available in the FastApply system as of 2025-09-27.*