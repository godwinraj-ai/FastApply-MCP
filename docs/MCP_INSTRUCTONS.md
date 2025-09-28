# FastApply MCP Server Instructions & Architecture Overview

## 🎯 Purpose

**Primary Mission**: Provide comprehensive code analysis, search, and transformation capabilities through a sophisticated multi-layered architecture combining ripgrep integration, AST intelligence, and advanced analysis tools.

## 🔧 Core Capabilities

### **Multi-Tool Architecture**
- **Ripgrep Integration**: High-performance pattern discovery via `ripgrep_integration.py`
- **AST Intelligence**: LLM-enhanced reasoning through `ast_rule_intelligence.py`
- **Enhanced Search**: Multi-strategy intelligent search pipeline via `enhanced_search.py`
- **Symbol Operations**: Advanced symbol detection and analysis via `symbol_operations.py`
- **Security & Quality**: Comprehensive analysis via `security_quality_analysis.py`

### **Performance Architecture**
- **Caching Strategy**: Intelligent result caching with LRU eviction
- **Parallel Processing**: Multi-threaded batch operations and concurrent execution
- **Graceful Degradation**: Robust fallback chains for missing dependencies
- **Memory Management**: Configurable limits and resource optimization

---

## 🚀 Implementation Architecture

### **System Components (14 Specialized Modules)**

#### **🔍 Foundation Layer**
1. **`ripgrep_integration.py`** - Ripgrep wrapper with search options and result processing
2. **`ast_search_official.py`** - Official ast-grep CLI integration
3. **`ast_search.py`** - Custom AST analysis capabilities
4. **`__init__.py`** - Module initialization and exports

#### **🧠 Intelligence Layer**
5. **`ast_rule_intelligence.py`** - LLM-based reasoning with Qdrant integration
6. **`enhanced_search.py`** - Multi-strategy search with caching and ranking
7. **`symbol_operations.py`** - Advanced symbol detection and reference analysis
8. **`deep_semantic_analysis.py`** - Semantic understanding and pattern recognition

#### **🔗 Analysis & Operations**
9. **`relationship_mapping.py`** - Dependency and coupling analysis with NetworkX
10. **`navigation_tools.py`** - Code navigation and exploration
11. **`security_quality_analysis.py`** - Security vulnerability scanning and quality assessment
12. **`safe_refactoring.py`** - Safe code transformation with validation

#### **⚡ Processing & Integration**
13. **`batch_operations.py`** - Bulk processing with scheduling and progress monitoring
14. **`main.py`** - MCP server orchestration and tool exposure

---

## 🛠️ Available Tools & Functions

### **Core File Operations**

#### **1. `edit_file`**
```python
# Apply code edits to a file using Fast Apply
edit_file(
    target_file: str,           # Path to target file
    instructions: str,          # Edit instructions
    code_edit: str,             # Code edit snippet
    force: bool = False,        # Override safety checks
    output_format: str = "text" # text, json
)
```
**Features:**
- Atomic file operations with backup creation
- Code quality validation
- Optimistic concurrency control
- Structured logging and error handling

#### **2. `dry_run_edit_file`**
```python
# Preview an edit without writing changes
dry_run_edit_file(
    target_file: str,           # Path to target file
    instruction: str,           # Edit instruction
    code_edit: str,             # Code edit snippet
    output_format: str = "text" # text, json
)
```

### **Search & Discovery Tools**

#### **3. `search_files`**
```python
# Search for files by simple substring pattern
search_files(
    path: str,                          # Directory to search
    pattern: str,                        # Search pattern
    excludePatterns: List[str] = None   # Patterns to exclude
)
```
**Performance:** Recursive directory scanning with security isolation

#### **4. `search_code_patterns`**
```python
# Search for semantic code patterns using AST-based matching
search_code_patterns(
    pattern: str,                    # AST pattern (e.g., "function $name($args) { $body }")
    language: str,                   # Target language (python, javascript, typescript)
    path: str,                       # File or directory path
    exclude_patterns: List[str] = None
)
```

#### **5. `analyze_code_structure`**
```python
# Analyze structural components of a code file
analyze_code_structure(
    file_path: str   # Path to the code file to analyze
)
```
**Returns:** Functions, classes, imports, complexity metrics

#### **6. `find_references`**
```python
# Find all references to a specific symbol
find_references(
    symbol: str,                              # Symbol name to search for
    path: str,                                # Search path
    symbol_type: str = "any"                  # function, class, variable, any
)
```

### **Advanced Analysis Tools**

#### **7. `dump_syntax_tree`**
```python
# Syntax tree analysis using official ast-grep
dump_syntax_tree(
    code: str,                       # Source code
    language: str,                   # Programming language
    dump_format: str = "pattern"     # pattern, cst, ast
)
```

#### **8. `find_code`**
```python
# Pattern-based code search with ast-grep
find_code(
    pattern: str,                    # Search pattern
    language: str,                   # Programming language
    path: str = ".",                 # Directory to search
    output_format: str = "json",     # json, text
    max_results: int = None          # Result limit
)
```

#### **9. `find_code_by_rule`**
```python
# Advanced search using YAML rules
find_code_by_rule(
    rule_yaml: str,                  # YAML rule definition
    path: str = ".",                 # Directory to search
    output_format: str = "json",     # json, text
    max_results: int = None          # Result limit
)
```

### **LLM-Enhanced Intelligence Tools**

#### **10. `llm_analyze_code`**
```python
# Deep semantic analysis using LLM + AST
llm_analyze_code(
    code: str,                                    # Source code
    language: str,                                 # Programming language
    analysis_type: str = "general",                # general, complexity, security, performance, architecture
    use_collective_memory: bool = True             # Use Qdrant patterns
)
```

#### **11. `llm_generate_rule`**
```python
# Dynamic rule generation from examples
llm_generate_rule(
    query: str,                                    # Natural language description
    language: str,                                 # Target language
    examples: List[str] = None,                    # Example code snippets
    rule_type: str = "pattern",                    # pattern, relational, composite
    use_collective_memory: bool = True             # Use learned patterns
)
```

#### **12. `llm_search_pattern`**
```python
# Intelligent pattern search with reasoning
llm_search_pattern(
    query: str,                                    # Natural language description
    language: str,                                 # Target language
    path: str = ".",                               # Search path
    use_collective_memory: bool = True,            # Use Qdrant intelligence
    max_results: int = None                        # Result limit
)
```

#### **13. `auto_ast_intelligence`**
```python
# Auto-detection and execution of optimal tools
auto_ast_intelligence(
    query: str,                                    # Natural language query
    context: str = None,                           # Task context
    language: str = None,                          # Default language
    path: str = None,                              # Default path
    auto_execute: bool = True                      # Execute detected sequence
)
```

### **Utility Tools**

#### **14. `read_multiple_files`**
```python
# Read multiple small files (concatenated output)
read_multiple_files(
    paths: List[str]   # List of file paths to read
)
```

#### **15. `call_tool`**
```python
# Generic tool invocation with unified branching
call_tool(
    name: str,          # Tool name to execute
    arguments: dict     # Tool arguments
)
```

---

## 🔄 Integration Patterns

### **1. Progressive Analysis Workflow**
```python
# Example: Comprehensive code analysis with graceful degradation
def analyze_code_comprehensive(code: str, language: str):
    # Stage 1: Fast ripgrep analysis (if available)
    if RIPGREP_AVAILABLE:
        fast_results = ripgrep_integration.search_pattern(code)

    # Stage 2: AST structural analysis (if available)
    if AST_SEARCH_AVAILABLE:
        structural_results = ast_search.analyze_structure(code)

    # Stage 3: Enhanced search with caching (if available)
    if ENHANCED_SEARCH_AVAILABLE:
        enhanced_results = enhanced_search_instance.analyze(code)

    # Stage 4: LLM semantic enhancement (if available)
    if AST_INTELLIGENCE_AVAILABLE:
        semantic_results = llm_analyze_code(code, language)

    return combine_results(fast_results, structural_results, enhanced_results, semantic_results)
```

### **2. Batch Processing Pattern**
```python
# Example: Large-scale project analysis
def analyze_project_batches(project_path: str):
    if BATCH_OPERATIONS_AVAILABLE:
        config = BatchConfig(
            max_concurrent_operations=4,
            timeout_per_operation=300,
            enable_progress_tracking=True
        )

        analyzer = BatchAnalysisSystem(config)
        results = analyzer.analyze_project_batches(
            project_path=project_path,
            analysis_types=["structure", "quality", "dependencies"]
        )

        return results.get_detailed_report()
    else:
        # Fallback to sequential processing
        return sequential_analysis(project_path)
```

### **3. Security & Quality Assessment**
```python
# Example: Comprehensive security analysis
def security_quality_audit(project_path: str):
    results = {}

    if SECURITY_QUALITY_AVAILABLE:
        # Security scanning
        security_report = security_scanner.security_scan_comprehensive(project_path)
        results['security'] = security_report

        # Quality assessment
        quality_assessment = quality_analyzer.assess_code_quality(project_path)
        results['quality'] = quality_assessment

        # Compliance reporting
        compliance_report = compliance_reporter.generate_compliance_report(
            security_report, quality_assessment
        )
        results['compliance'] = compliance_report

    return results
```

### **4. Symbol Operations Pattern**
```python
# Example: Advanced symbol analysis
def analyze_symbol_usage(symbol_name: str, project_path: str):
    if SYMBOL_OPERATIONS_AVAILABLE:
        # Find symbol references
        references = symbol_operations.find_symbol_references(
            symbol_name, project_path
        )

        # Analyze symbol relationships
        relationships = relationship_mapper.analyze_relationships(
            symbol_name, project_path
        )

        # Generate impact analysis
        impact_analysis = safe_renamer.analyze_rename_impact(
            symbol_name, project_path
        )

        return {
            'references': references,
            'relationships': relationships,
            'impact_analysis': impact_analysis
        }
```

---

## 🎯 Best Practices

### **Performance Optimization**
- **Leverage Caching**: Use `EnhancedSearchInfrastructure` for repeated searches
- **Batch Operations**: Process multiple files concurrently with `BatchAnalysisSystem`
- **Graceful Degradation**: Always check availability flags before using tools
- **Resource Management**: Monitor memory usage with configurable limits

### **Quality Improvement**
- **Cross-Tool Validation**: Verify results across multiple analysis tools
- **Progressive Enhancement**: Start with fast tools, add depth as needed
- **Error Handling**: Implement robust exception handling for tool failures
- **Logging**: Use structured logging for debugging and monitoring

### **Security & Safety**
- **Path Isolation**: Always use `_secure_resolve` for file paths
- **Validation**: Implement comprehensive input validation
- **Backup Strategy**: Use atomic operations with backup creation
- **Access Control**: Respect workspace boundaries and file permissions

---

## 🔧 Configuration & Setup

### **Environment Variables**
```bash
# Workspace and security
WORKSPACE_ROOT=/path/to/project
FAST_APPLY_STRICT_PATHS=1

# Performance tuning
FAST_APPLY_MAX_FILE_BYTES=10485760    # 10MB
FAST_APPLY_MAX_REQUEST_BYTES=20971520 # 20MB
FAST_APPLY_ALLOWED_EXTS=.py,.js,.ts,.jsx,.tsx,.md,.json,.yaml,.yml

# LLM configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4

# FastApply API configuration
FAST_APPLY_URL=http://localhost:1234/v1
FAST_APPLY_MODEL=fastapply-1.5b
FAST_APPLY_TIMEOUT=30.0
```

### **Tool Availability Detection**
```python
# Check available capabilities
availability_status = {
    "ast_search_available": AST_SEARCH_AVAILABLE,
    "ast_intelligence_available": AST_INTELLIGENCE_AVAILABLE,
    "ast_grep_available": AST_GREP_AVAILABLE,
    "enhanced_search_available": ENHANCED_SEARCH_AVAILABLE,
    "semantic_analysis_available": SEMANTIC_ANALYSIS_AVAILABLE,
    "relationship_mapping_available": RELATIONSHIP_MAPPING_AVAILABLE,
    "navigation_tools_available": NAVIGATION_TOOLS_AVAILABLE,
    "safe_refactoring_available": SAFE_REFACTORING_AVAILABLE,
    "batch_operations_available": BATCH_OPERATIONS_AVAILABLE,
    "security_quality_available": SECURITY_QUALITY_AVAILABLE,
    "ripgrep_available": RIPGREP_AVAILABLE,
    "qdrant_available": QDRANT_AVAILABLE
}
```

---

## 📊 Implementation Features

### **Advanced Capabilities**

#### **1. Intelligent Search System**
- **Multi-Strategy Search**: EXACT, FUZZY, SEMANTIC, HYBRID strategies
- **Result Ranking**: RELEVANCE, FREQUENCY, RECENCY, CONFIDENCE, COMBINED
- **Smart Caching**: LRU-based caching with access tracking
- **Context Awareness**: File type, language, and pattern-aware search

#### **2. Relationship Analysis**
- **NetworkX Integration**: Advanced graph-based relationship mapping
- **Coupling Analysis**: CONTENT, COMMON, EXTERNAL, CONTROL, STAMP, DATA, TEMPORAL
- **Dependency Mapping**: DIRECT, INDIRECT, CYCLICAL, TRANSITIVE, WEAK, STRONG
- **Pattern Recognition**: Complex architectural relationship detection

#### **3. Security & Quality Framework**
- **Vulnerability Scanning**: SQL injection, XSS, insecure deserialization, path traversal
- **Quality Metrics**: Cyclomatic complexity, cognitive complexity, maintainability index
- **Compliance Reporting**: OWASP Top 10, ISO 27001, SOC 2, GDPR, HIPAA
- **Code Smells Detection**: Duplicate code, long methods, large classes, god objects

#### **4. Batch Processing System**
- **Concurrent Execution**: Thread pool-based parallel processing
- **Progress Monitoring**: Real-time progress tracking and reporting
- **Error Handling**: Comprehensive error recovery and reporting
- **Resource Management**: Memory-aware processing with configurable limits

---

## 🔒 Security Architecture

### **Security Features**
- **Path Isolation**: Strict workspace confinement with `_secure_resolve`
- **File Size Limits**: Prevent memory exhaustion with configurable limits
- **Extension Filtering**: Control allowed file types through configuration
- **Input Validation**: Comprehensive validation of all inputs and outputs
- **Safe Refactoring**: Dependency analysis and impact assessment

### **Quality Assurance**
- **Code Validation**: Language-specific syntax and semantic validation
- **Backup Strategy**: Automatic backup creation before modifications
- **Atomic Operations**: Ensure data consistency during file operations
- **Rollback Planning**: Generate safe rollback strategies for changes

---

## 🚀 Advanced Workflows

### **1. Collective Intelligence Integration**
```python
# Leverage Qdrant-based learning system
if QDRANT_AVAILABLE:
    # Store successful patterns
    await qdrant_store(
        content="successful ast-grep pattern for async functions",
        metadata={
            "success_rate": 0.95,
            "language": "python",
            "pattern_type": "async",
            "usage_count": 150
        }
    )

    # Retrieve similar experiences
    experiences = await qdrant_find("async error handling patterns")
```

### **2. Multi-Language Support**
- **Python**: Full AST analysis with comprehensive symbol operations
- **JavaScript/TypeScript**: Complete support including JSX/TSX
- **Java**: Structural analysis and pattern matching
- **C/C++**: Syntax tree analysis and refactoring support
- **Go/Rust**: Modern language features and pattern recognition

### **3. Auto-Intent Detection**
```python
# Natural language to tool selection
response = await auto_ast_intelligence(
    query="find all async functions with error handling",
    context="analyzing Python backend code",
    language="python",
    path="./src",
    auto_execute=True
)
```

---

## 🏆 Implementation Quality

### **Architecture Strengths**
- **Modular Design**: Clean separation of concerns with specialized modules
- **Graceful Degradation**: Robust fallback chains for missing dependencies
- **Performance Optimization**: Multi-layered caching and parallel processing
- **Security-First**: Comprehensive security measures and validation
- **Extensible**: Plugin architecture with optional dependency integration

### **Code Quality Metrics**
- **Type Safety**: Extensive use of dataclasses and type hints
- **Error Handling**: Comprehensive exception handling and logging
- **Documentation**: Detailed docstrings and architectural clarity
- **Testing**: Robust testing infrastructure with validation
- **Maintainability**: High cohesion, low coupling design

---

## 🔄 Future Enhancements

### **Planned Features**
- **Real-time Collaboration**: Multi-user analysis and pattern sharing
- **Enhanced LLM Integration**: Support for advanced AI models
- **Visual Dependency Graphs**: Interactive architecture visualization
- **Performance Profiling**: Runtime performance analysis
- **Advanced Security Analysis**: SAST/DAST integration

### **Community Features**
- **Pattern Marketplace**: Share and discover analysis patterns
- **Rule Templates**: Pre-built rules for common use cases
- **Performance Benchmarks**: Community optimization patterns
- **Integration Plugins**: IDE and CI/CD pipeline integration

---

## 🎯 Conclusion

FastApply MCP Server v2 represents a sophisticated implementation of modern software architecture principles, combining:

**Technical Excellence:**
- Clean, modular design with clear separation of concerns
- Robust error handling and graceful degradation
- Performance optimization through caching and parallel processing
- Comprehensive security measures and validation

**Functional Capabilities:**
- Multi-tool integration with intelligent orchestration
- Advanced code analysis and pattern recognition
- Security vulnerability scanning and quality assessment
- Batch processing with progress monitoring

**Innovation Features:**
- LLM-enhanced reasoning with collective intelligence
- Auto-intent detection and tool selection
- Cross-language support with extensibility
- Community-driven pattern learning

**Bottom Line**: FastApply v2 delivers a production-ready, enterprise-grade code intelligence platform that combines unprecedented speed with deep semantic understanding and continuous learning capabilities.

---

*This documentation reflects the actual implementation as of 2025-09-28, based on comprehensive architectural analysis of the source code.*
