"""
Advanced Symbol Operations - Intelligent Symbol Detection and Analysis

Implements sophisticated symbol finding, scope resolution, and semantic analysis
to provide powerful code navigation and understanding capabilities.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .ast_rule_intelligence import LLMAstReasoningEngine
from .enhanced_search import EnhancedSearchInfrastructure, SearchContext, SearchStrategy
from .ripgrep_integration import RipgrepIntegration, SearchOptions, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolType(Enum):
    """Types of symbols that can be found and analyzed."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    IMPORT = "import"
    MODULE = "module"
    PROPERTY = "property"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    DECORATOR = "decorator"
    GENERATOR = "generator"
    ASYNC_FUNCTION = "async_function"


class SymbolScope(Enum):
    """Symbol scope levels."""

    GLOBAL = "global"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    LOCAL = "local"
    BLOCK = "block"


class ReferenceType(Enum):
    """Types of symbol references."""

    READ = "read"  # Reading a variable/property
    WRITE = "write"  # Writing to a variable/property
    CALL = "call"  # Calling a function/method
    IMPORT = "import"  # Importing a module/symbol
    INHERIT = "inherit"  # Inheriting from a class
    IMPLEMENT = "implement"  # Implementing an interface
    OVERRIDE = "override"  # Overriding a method
    REFERENCE = "reference"  # General reference
    TYPE_ANNOTATION = "type_annotation"  # Using in type annotation
    DECORATE = "decorate"  # Using as decorator


@dataclass
class SymbolInfo:
    """Comprehensive information about a discovered symbol."""

    name: str
    symbol_type: SymbolType
    file_path: str
    line_number: int
    column_number: int = 0
    scope: SymbolScope = SymbolScope.GLOBAL
    parent_symbol: Optional[str] = None
    documentation: Optional[str] = None
    signature: Optional[str] = None
    return_type: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_static: bool = False
    is_abstract: bool = False
    confidence_score: float = 1.0
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceInfo:
    """Detailed information about a symbol reference."""

    symbol_name: str
    reference_type: ReferenceType
    file_path: str
    line_number: int
    column_number: int = 0
    context: str = ""
    is_definition: bool = False
    confidence_score: float = 1.0
    language: str = "python"
    scope: SymbolScope = SymbolScope.GLOBAL
    container_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedScope:
    """Result of symbol scope resolution."""

    symbol_name: str
    resolved_path: str
    scope_level: SymbolScope
    context: Dict[str, Any]
    confidence: float
    alternative_matches: List[SymbolInfo] = field(default_factory=list)


@dataclass
class SymbolSearchContext:
    """Context for symbol search operations."""

    symbol_name: str
    symbol_type: Optional[SymbolType] = None
    scope: Optional[str] = None
    file_path: Optional[str] = None
    language: str = "python"
    include_imports: bool = True
    include_definitions: bool = True
    include_references: bool = False
    max_results: int = 100
    case_sensitive: bool = True
    use_semantic_search: bool = True


class ReferenceAnalysis:
    """Enhanced reference analysis with comprehensive symbol tracking."""

    def __init__(
        self,
        ripgrep_integration: Optional[RipgrepIntegration] = None,
        ast_engine: Optional[LLMAstReasoningEngine] = None,
        search_infrastructure: Optional[EnhancedSearchInfrastructure] = None,
    ):
        """Initialize reference analysis system."""
        self.ripgrep = ripgrep_integration or RipgrepIntegration()
        self.ast_engine = ast_engine or LLMAstReasoningEngine()
        self.search_infra = search_infrastructure or EnhancedSearchInfrastructure()

        # Cache for reference analysis results
        self._reference_cache: Dict[str, List[ReferenceInfo]] = {}
        self._dependency_cache: Dict[str, Dict[str, List[ReferenceInfo]]] = {}

        # Language-specific reference patterns
        self._reference_patterns = {
            "python": {
                ReferenceType.READ: [
                    r"(\w+)(?=\s*[\+\-\*/%&|^]|==|!=|<=|>=|<|>|is|in)",  # Used in expressions
                    r"print\s*\(\s*(\w+)",  # Used in print statements
                    r"return\s+(\w+)",  # Returned
                ],
                ReferenceType.WRITE: [
                    r"(\w+)\s*=",  # Assignment
                    r"(\w+)\s*[\+\-\*/%&|^]?=",  # Compound assignment
                ],
                ReferenceType.CALL: [
                    r"(\w+)\s*\(",  # Function call
                    r"await\s+(\w+)\s*\(",  # Async call
                    r"(\w+)\.(\w+)\s*\(",  # Method call
                ],
                ReferenceType.IMPORT: [
                    r"import\s+(\w+)",  # Direct import
                    r"from\s+(\w+)\s+import",  # From import
                ],
                ReferenceType.INHERIT: [
                    r"class\s+\w+\s*\(\s*(\w+)\s*\)",  # Class inheritance
                ],
                ReferenceType.TYPE_ANNOTATION: [
                    r":\s*(\w+)",  # Type annotation
                    r"->\s*(\w+)",  # Return type annotation
                ],
                ReferenceType.DECORATE: [
                    r"@(\w+)",  # Decorator
                ],
            },
            "javascript": {
                ReferenceType.READ: [
                    r"(\w+)(?=\s*[\+\-\*/%&|^]|==|!=|<=|>=|<|>|===|!==)",  # Used in expressions
                    r"console\.log\s*\(\s*(\w+)",  # Console log
                    r"return\s+(\w+)",  # Returned
                ],
                ReferenceType.WRITE: [
                    r"(?:let|const|var)\s+(\w+)\s*=",  # Declaration with assignment
                    r"(\w+)\s*=",  # Assignment
                    r"(\w+)\s*[\+\-\*/%&|^]?=",  # Compound assignment
                ],
                ReferenceType.CALL: [
                    r"(\w+)\s*\(",  # Function call
                    r"await\s+(\w+)\s*\(",  # Async call
                    r"(\w+)\.(\w+)\s*\(",  # Method call
                ],
                ReferenceType.IMPORT: [
                    r"import\s+.*?from\s+['\"](\w+)['\"]",  # ES6 import
                    r"require\s*\(\s*['\"](\w+)['\"]\s*\)",  # CommonJS require
                ],
                ReferenceType.INHERIT: [
                    r"class\s+\w+\s+extends\s+(\w+)",  # Class extends
                ],
            },
            "typescript": {
                ReferenceType.READ: [
                    r"(\w+)(?=\s*[\+\-\*/%&|^]|==|!=|<=|>=|<|>|===|!==)",  # Used in expressions
                    r"console\.log\s*\(\s*(\w+)",  # Console log
                    r"return\s+(\w+)",  # Returned
                ],
                ReferenceType.WRITE: [
                    r"(?:let|const|var)\s+(\w+)\s*[:=]",  # Declaration with assignment
                    r"(\w+)\s*=",  # Assignment
                    r"(\w+)\s*[\+\-\*/%&|^]?=",  # Compound assignment
                ],
                ReferenceType.CALL: [
                    r"(\w+)\s*\(",  # Function call
                    r"await\s+(\w+)\s*\(",  # Async call
                    r"(\w+)\.(\w+)\s*\(",  # Method call
                ],
                ReferenceType.IMPORT: [
                    r"import\s+.*?from\s+['\"](\w+)['\"]",  # ES6 import
                    r"import\s+.*?=\s*require\s*\(\s*['\"](\w+)['\"]\s*\)",  # TypeScript require
                ],
                ReferenceType.INHERIT: [
                    r"class\s+\w+\s+extends\s+(\w+)",  # Class extends
                    r"implements\s+(\w+)",  # Interface implementation
                ],
                ReferenceType.TYPE_ANNOTATION: [
                    r":\s*(\w+)",  # Type annotation
                    r"->\s*(\w+)",  # Return type annotation
                    r"interface\s+\w+\s+extends\s+(\w+)",  # Interface extends
                ],
            },
        }

    def analyze_symbol_references(
        self, symbol_info: SymbolInfo, include_definitions: bool = False, max_results: int = 1000
    ) -> List[ReferenceInfo]:
        """
        Analyze all references to a symbol with comprehensive classification.

        Args:
            symbol_info: SymbolInfo object to analyze references for
            include_definitions: Whether to include symbol definitions in results
            max_results: Maximum number of references to return

        Returns:
            List of ReferenceInfo objects with detailed reference information
        """
        # Generate cache key
        cache_key = f"{symbol_info.name}:{symbol_info.file_path}:{include_definitions}:{max_results}"

        # Check cache first
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key]

        references = []

        try:
            # Search for all occurrences of the symbol name
            search_context = SearchContext(
                query=symbol_info.name,
                file_types=[symbol_info.language],
                max_results=max_results,
                strategy=SearchStrategy.EXACT,
            )

            results, _ = self.search_infra.search(search_context)

            # Analyze each result to determine reference type
            for result in results:
                # Skip definition if requested
                if not include_definitions and result.file_path == symbol_info.file_path and result.line_number == symbol_info.line_number:
                    continue

                # Determine reference type and create ReferenceInfo
                reference_info = self._analyze_reference_context(
                    result, symbol_info.name, symbol_info.language
                )

                if reference_info:
                    references.append(reference_info)

            # Sort by confidence score and line number
            references.sort(key=lambda x: (-x.confidence_score, x.line_number))

        except Exception as e:
            logger.warning(f"Error analyzing symbol references: {e}")

        # Cache result
        self._reference_cache[cache_key] = references

        return references

    def get_symbol_dependencies(
        self, symbol_info: SymbolInfo, include_transitive: bool = False
    ) -> Dict[str, List[ReferenceInfo]]:
        """
        Get all dependencies of a symbol with detailed analysis.

        Args:
            symbol_info: SymbolInfo object to analyze dependencies for
            include_transitive: Whether to include transitive dependencies

        Returns:
            Dictionary mapping dependency types to lists of ReferenceInfo objects
        """
        cache_key = f"{symbol_info.name}:{symbol_info.file_path}:{include_transitive}"

        # Check cache first
        if cache_key in self._dependency_cache:
            return self._dependency_cache[cache_key]

        dependencies: Dict[str, List[ReferenceInfo]] = {
            "direct": [],
            "transitive": [],
            "imports": [],
            "type_dependencies": [],
            "runtime_dependencies": [],
        }

        try:
            # Read and analyze the symbol's file
            if symbol_info.file_path:
                with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Analyze file content for dependencies
                file_deps = self._analyze_file_dependencies(content, symbol_info.language)
                dependencies["direct"] = file_deps["direct"]
                dependencies["imports"] = file_deps["imports"]
                dependencies["type_dependencies"] = file_deps["type_dependencies"]
                dependencies["runtime_dependencies"] = file_deps["runtime_dependencies"]

                # Analyze transitive dependencies if requested
                if include_transitive:
                    transitive_deps = self._analyze_transitive_dependencies(file_deps["direct"], symbol_info.language)
                    dependencies["transitive"] = transitive_deps

        except Exception as e:
            logger.warning(f"Error getting symbol dependencies: {e}")

        # Cache result
        self._dependency_cache[cache_key] = dependencies

        return dependencies

    def analyze_refactoring_safety(self, symbol_info: SymbolInfo) -> Dict[str, Any]:
        """
        Analyze the safety of refactoring operations on a symbol.

        Args:
            symbol_info: SymbolInfo object to analyze for refactoring safety

        Returns:
            Dictionary with safety analysis results
        """
        safety_analysis: Dict[str, Any] = {
            "is_safe_to_rename": True,
            "risk_factors": [],
            "impact_score": 0.0,
            "affected_files": set(),
            "reference_count": 0,
            "critical_references": [],
            "recommendations": [],
        }

        try:
            # Get all references to the symbol
            references = self.analyze_symbol_references(symbol_info, include_definitions=False)

            safety_analysis["reference_count"] = len(references)

            # Analyze each reference for safety concerns
            for ref in references:
                # Track affected files
                safety_analysis["affected_files"].add(ref.file_path)

                # Check for critical references that might make renaming unsafe
                if self._is_critical_reference(ref):
                    safety_analysis["critical_references"].append(ref)
                    safety_analysis["risk_factors"].append(f"Critical reference at {ref.file_path}:{ref.line_number}")

            # Calculate impact score (0.0 = low impact, 1.0 = high impact)
            safety_analysis["impact_score"] = self._calculate_refactoring_impact(references)

            # Determine if it's safe to rename
            if len(safety_analysis["critical_references"]) > 0:
                safety_analysis["is_safe_to_rename"] = False

            if safety_analysis["impact_score"] > 0.8:
                safety_analysis["risk_factors"].append("High impact refactoring")

            # Generate recommendations
            safety_analysis["recommendations"] = self._generate_refactoring_recommendations(safety_analysis)

        except Exception as e:
            logger.warning(f"Error analyzing refactoring safety: {e}")
            safety_analysis["is_safe_to_rename"] = False
            safety_analysis["risk_factors"].append(f"Analysis error: {str(e)}")

        # Convert set to list for JSON serialization
        safety_analysis["affected_files"] = list(safety_analysis["affected_files"])

        return safety_analysis

    def get_reference_statistics(self, symbol_info: SymbolInfo) -> Dict[str, Any]:
        """
        Get comprehensive statistics about symbol references.

        Args:
            symbol_info: SymbolInfo object to analyze

        Returns:
            Dictionary with reference statistics
        """
        references = self.analyze_symbol_references(symbol_info)

        stats: Dict[str, Any] = {
            "total_references": len(references),
            "references_by_type": {},
            "references_by_file": {},
            "references_by_scope": {},
            "average_confidence": 0.0,
            "reference_density": 0.0,
        }

        if not references:
            return stats

        # Count by reference type
        for ref in references:
            ref_type = ref.reference_type.value
            ref_by_type: Dict[str, int] = stats["references_by_type"]
            ref_by_type[ref_type] = ref_by_type.get(ref_type, 0) + 1

        # Count by file
        for ref in references:
            file_path = ref.file_path
            ref_by_file: Dict[str, int] = stats["references_by_file"]
            ref_by_file[file_path] = ref_by_file.get(file_path, 0) + 1

        # Count by scope
        for ref in references:
            scope = ref.scope.value
            ref_by_scope: Dict[str, int] = stats["references_by_scope"]
            ref_by_scope[scope] = ref_by_scope.get(scope, 0) + 1

        # Calculate average confidence
        stats["average_confidence"] = sum(r.confidence_score for r in references) / len(references)

        # Calculate reference density (references per 100 lines)
        if symbol_info.file_path:
            try:
                with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                    lines = len(f.readlines())
                if lines > 0:
                    stats["reference_density"] = (len(references) / lines) * 100
            except Exception:
                pass

        return stats

    def find_unused_symbols(self, file_path: str, language: str = "python") -> List[SymbolInfo]:
        """
        Find potentially unused symbols in a file.

        Args:
            file_path: Path to the file to analyze
            language: Programming language of the file

        Returns:
            List of SymbolInfo objects that appear to be unused
        """
        unused_symbols = []

        try:
            # First, find all defined symbols in the file
            symbol_ops = AdvancedSymbolOperations(
                ripgrep_integration=self.ripgrep,
                ast_engine=self.ast_engine,
                search_infrastructure=self.search_infra,
            )

            # Find all symbols defined in the file
            defined_symbols = symbol_ops.find_symbols_by_pattern(r"(def |class |import |from )", language)
            defined_symbols = [s for s in defined_symbols if s.file_path == file_path]

            # For each defined symbol, check if it has references outside its definition
            for symbol in defined_symbols:
                references = self.analyze_symbol_references(symbol, include_definitions=False)

                # Filter out references from the same symbol (e.g., recursive calls)
                external_references = [
                    r for r in references
                    if not (r.file_path == symbol.file_path and r.line_number == symbol.line_number)
                ]

                # Consider symbols with no external references as potentially unused
                # (with some exceptions for common patterns)
                if not external_references and not self._is_exemption_from_unused_check(symbol):
                    unused_symbols.append(symbol)

        except Exception as e:
            logger.warning(f"Error finding unused symbols: {e}")

        return unused_symbols

    def _analyze_reference_context(self, search_result: Any, symbol_name: str, language: str) -> Optional[ReferenceInfo]:
        """Analyze the context of a search result to determine reference type."""
        try:
            line_text = search_result.line_content
            reference_type = ReferenceType.REFERENCE
            confidence_score = 1.0
            context = line_text.strip()

            # Determine reference type based on language patterns
            if language in self._reference_patterns:
                for ref_type, patterns in self._reference_patterns[language].items():
                    for pattern in patterns:
                        if re.search(pattern, line_text):
                            reference_type = ref_type
                            confidence_score = 0.9  # High confidence for pattern matches
                            break

            # Additional context analysis
            container_symbol = self._extract_container_symbol(line_text, language)
            scope = self._determine_reference_scope(line_text, language)

            return ReferenceInfo(
                symbol_name=symbol_name,
                reference_type=reference_type,
                file_path=search_result.file_path,
                line_number=search_result.line_number,
                context=context,
                confidence_score=confidence_score,
                language=language,
                scope=scope,
                container_symbol=container_symbol,
                metadata={"line_content": line_text},
            )

        except Exception as e:
            logger.warning(f"Error analyzing reference context: {e}")
            return None

    def _analyze_file_dependencies(self, content: str, language: str) -> Dict[str, List[ReferenceInfo]]:
        """Analyze dependencies within a file's content."""
        dependencies: Dict[str, List[ReferenceInfo]] = {
            "direct": [],
            "imports": [],
            "type_dependencies": [],
            "runtime_dependencies": [],
        }

        # This is a simplified implementation
        # In a full implementation, you would use AST analysis for accurate dependency detection
        try:
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                line = line.strip()

                # Look for import statements
                if language == "python":
                    if line.startswith("import ") or line.startswith("from "):
                        # Extract imported module/symbol
                        import_match = re.search(r"import\s+(\w+)", line) or re.search(r"from\s+(\w+)", line)
                        if import_match:
                            dep_info = ReferenceInfo(
                                symbol_name=import_match.group(1),
                                reference_type=ReferenceType.IMPORT,
                                file_path="",  # Will be set by caller
                                line_number=i,
                                context=line,
                                language=language,
                                metadata={"import_line": line},
                            )
                            dependencies["imports"].append(dep_info)

                # Look for function calls and variable usage (simplified)
                call_pattern = r"(\w+)\s*\("
                calls = re.findall(call_pattern, line)
                for call in calls:
                    if call not in ["if", "for", "while", "def", "class"]:
                        dep_info = ReferenceInfo(
                            symbol_name=call,
                            reference_type=ReferenceType.CALL,
                            file_path="",  # Will be set by caller
                            line_number=i,
                            context=line,
                            language=language,
                            metadata={"call_site": line},
                        )
                        dependencies["runtime_dependencies"].append(dep_info)

        except Exception as e:
            logger.warning(f"Error analyzing file dependencies: {e}")

        return dependencies

    def _analyze_transitive_dependencies(self, direct_deps: List[ReferenceInfo], language: str) -> List[ReferenceInfo]:
        """Analyze transitive dependencies (simplified implementation)."""
        # This is a placeholder for transitive dependency analysis
        # In a full implementation, you would recursively analyze dependencies
        return []

    def _is_critical_reference(self, reference: ReferenceInfo) -> bool:
        """Determine if a reference is critical for refactoring safety."""
        # References in test files, configuration files, or public APIs might be critical
        critical_patterns = ["test_", "spec_", "config", "api", "public"]
        critical_file_patterns = ["test_", "spec_", "__init__.py"]

        for pattern in critical_patterns:
            if pattern in reference.symbol_name.lower():
                return True

        for pattern in critical_file_patterns:
            if pattern in reference.file_path.lower():
                return True

        # Method calls in inheritance contexts are critical
        if reference.reference_type == ReferenceType.CALL and "class" in reference.context.lower():
            return True

        return False

    def _calculate_refactoring_impact(self, references: List[ReferenceInfo]) -> float:
        """Calculate the impact score of refactoring (0.0 = low, 1.0 = high)."""
        if not references:
            return 0.0

        impact = 0.0

        # More references = higher impact
        impact = min(len(references) / 10.0, 0.4)  # Max 0.4 for reference count

        # References across multiple files = higher impact
        unique_files = len(set(r.file_path for r in references))
        impact += min(unique_files / 5.0, 0.3)  # Max 0.3 for file spread

        # Critical references = higher impact
        critical_count = sum(1 for r in references if self._is_critical_reference(r))
        impact += min(critical_count / 3.0, 0.3)  # Max 0.3 for critical refs

        return min(impact, 1.0)

    def _generate_refactoring_recommendations(self, safety_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for safe refactoring."""
        recommendations = []

        if not safety_analysis["is_safe_to_rename"]:
            recommendations.append("Consider creating a backup before refactoring")

        if safety_analysis["impact_score"] > 0.7:
            recommendations.append("High impact refactoring - consider incremental changes")

        if len(safety_analysis["affected_files"]) > 5:
            recommendations.append("Consider refactoring in smaller batches")

        if safety_analysis["reference_count"] > 50:
            recommendations.append("High number of references - ensure comprehensive testing")

        if not recommendations:
            recommendations.append("Refactoring appears safe - proceed with normal caution")

        return recommendations

    def _extract_container_symbol(self, line_text: str, language: str) -> Optional[str]:
        """Extract the container symbol (class/function) from a line."""
        # This is a simplified implementation
        # In a full implementation, you would use AST analysis
        try:
            if language == "python":
                # Look for class or function definitions in the broader context
                # This would require analyzing more than just the current line
                pass
            elif language in ["javascript", "typescript"]:
                # Similar analysis for JS/TS
                pass
        except Exception:
            pass

        return None

    def _determine_reference_scope(self, line_text: str, language: str) -> SymbolScope:
        """Determine the scope of a reference."""
        # This is a simplified implementation
        # In a full implementation, you would use AST analysis
        if language in ["javascript", "typescript"]:
            if "this." in line_text:
                return SymbolScope.CLASS
            elif "var " in line_text or "let " in line_text or "const " in line_text:
                return SymbolScope.LOCAL

        return SymbolScope.GLOBAL

    def _is_exemption_from_unused_check(self, symbol: SymbolInfo) -> bool:
        """Check if a symbol should be exempt from unused symbol analysis."""
        # Common patterns that should not be flagged as unused
        exempt_patterns = [
            "__init__",  # Python constructors
            "__str__", "__repr__",  # String representation methods
            "main",  # Entry points
            "setUp", "tearDown",  # Test setup methods
            "test_", "spec_",  # Test functions
        ]

        for pattern in exempt_patterns:
            if pattern in symbol.name.lower():
                return True

        # Don't flag private methods as unused (they might be used internally)
        if symbol.is_private:
            return True

        return False

    def clear_cache(self):
        """Clear all reference analysis caches."""
        self._reference_cache.clear()
        self._dependency_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "reference_cache_size": len(self._reference_cache),
            "dependency_cache_size": len(self._dependency_cache),
        }


class AdvancedSymbolOperations:
    """Advanced symbol operations with semantic understanding and scope awareness."""

    def __init__(
        self,
        ripgrep_integration: Optional[RipgrepIntegration] = None,
        ast_engine: Optional[LLMAstReasoningEngine] = None,
        search_infrastructure: Optional[EnhancedSearchInfrastructure] = None,
    ):
        """Initialize advanced symbol operations."""
        self.ripgrep = ripgrep_integration or RipgrepIntegration()
        self.ast_engine = ast_engine or LLMAstReasoningEngine()
        self.search_infra = search_infrastructure or EnhancedSearchInfrastructure()

        # Cache for symbol analysis results
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._scope_cache: Dict[str, ResolvedScope] = {}

        # Language-specific patterns
        self._language_patterns = {
            "python": {
                "function": r"def\s+(\w+)\s*\(",
                "class": r"class\s+(\w+)\s*[\(:]",
                "method": r"def\s+(\w+)\s*\(",
                "variable": r"(\w+)\s*=",
                "import": r"import\s+(\w+)",
                "from_import": r"from\s+\w+\s+import\s+(\w+)",
            },
            "javascript": {
                "function": r"function\s+(\w+)\s*\(",
                "class": r"class\s+(\w+)\s*[\{]",
                "method": r"(\w+)\s*\([^)]*\)\s*[\{=]",
                "variable": r"(?:const|let|var)\s+(\w+)\s*=",
                "import": r"import\s+.*?\s+from\s+['\"](\w+)['\"]",
            },
            "typescript": {
                "function": r"function\s+(\w+)\s*\(",
                "class": r"class\s+(\w+)\s*[\{]",
                "method": r"(\w+)\s*\([^)]*\)\s*[\{=:]",
                "variable": r"(?:const|let|var)\s+(\w+)\s*[:=]",
                "interface": r"interface\s+(\w+)\s*[\{]",
                "type": r"type\s+(\w+)\s*=",
            },
        }

    def find_symbol(self, symbol_name: str, symbol_type: Optional[SymbolType] = None, scope: Optional[str] = None) -> SymbolInfo:
        """
        Find a symbol by name with semantic understanding and scope awareness.

        Args:
            symbol_name: Name of the symbol to find
            symbol_type: Optional type filter for the symbol
            scope: Optional scope constraint

        Returns:
            SymbolInfo object with comprehensive symbol information
        """
        # Create search context
        context = SymbolSearchContext(symbol_name=symbol_name, symbol_type=symbol_type, scope=scope, use_semantic_search=True)

        # Check cache first
        cache_key = self._generate_symbol_cache_key(context)
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key]

        # Perform multi-stage symbol search
        symbol_info = self._perform_symbol_search(context)

        # Cache result
        self._symbol_cache[cache_key] = symbol_info

        return symbol_info

    def find_symbols_by_pattern(self, pattern: str, language: str = "python") -> List[SymbolInfo]:
        """
        Find symbols matching a pattern with intelligent matching.

        Args:
            pattern: Search pattern (can include regex)
            language: Programming language to search in

        Returns:
            List of matching SymbolInfo objects
        """
        symbols = []

        # Use ripgrep for fast pattern matching
        try:
            # Search without file type filter (due to ripgrep integration issue)
            # and filter results manually based on file extension
            options = SearchOptions()
            results = self.ripgrep.search_files(pattern=pattern, path=".", options=options)

            # Filter results by language and convert to SymbolInfo objects
            for result in results.results:
                # Check file extension matches language
                file_ext = result.file_path.split('.')[-1] if '.' in result.file_path else ''
                should_include = False

                if language == "python" and file_ext == "py":
                    should_include = True
                elif language == "javascript" and file_ext in ["js", "jsx", "mjs"]:
                    should_include = True
                elif language == "typescript" and file_ext in ["ts", "tsx"]:
                    should_include = True

                if should_include:
                    symbol_info = self._convert_search_result_to_symbol(result, "unknown", language)
                    if symbol_info:
                        symbols.append(symbol_info)

        except Exception as e:
            logger.warning(f"Error in pattern search: {e}")
        # Sort by confidence score
        symbols.sort(key=lambda x: x.confidence_score, reverse=True)

        return symbols

    def resolve_symbol_scope(self, symbol_name: str, context: str) -> ResolvedScope:
        """
        Resolve symbol scope with contextual understanding.

        Args:
            symbol_name: Name of the symbol to resolve
            context: Context string for scope resolution

        Returns:
            ResolvedScope object with scope information
        """
        # Check cache first
        cache_key = f"{symbol_name}:{context}"
        if cache_key in self._scope_cache:
            return self._scope_cache[cache_key]

        # Parse context to understand scope
        context_info = self._parse_context(context)

        # Find all occurrences of the symbol
        all_occurrences = self.find_symbols_by_pattern(symbol_name, context_info.get("language", "python"))

        # Filter by context relevance
        relevant_symbols = self._filter_symbols_by_context(all_occurrences, context_info)

        # Determine best match
        best_match = self._determine_best_scope_match(relevant_symbols, context_info)

        # Create resolved scope
        resolved_scope = ResolvedScope(
            symbol_name=symbol_name,
            resolved_path=best_match.file_path if best_match else "",
            scope_level=best_match.scope if best_match else SymbolScope.GLOBAL,
            context=context_info,
            confidence=best_match.confidence_score if best_match else 0.0,
            alternative_matches=relevant_symbols,
        )

        # Cache result
        self._scope_cache[cache_key] = resolved_scope

        return resolved_scope

    def get_symbol_references(self, symbol_info: SymbolInfo) -> List[SymbolInfo]:
        """
        Find all references to a given symbol.

        Args:
            symbol_info: SymbolInfo object to find references for

        Returns:
            List of SymbolInfo objects representing references
        """
        references = []

        # Use enhanced search infrastructure
        search_context = SearchContext(
            query=symbol_info.name, file_types=[symbol_info.language], max_results=symbol_info.metadata.get("max_references", 100)
        )

        # Search for references
        results, _ = self.search_infra.search(search_context)

        # Convert results to symbol references
        for result in results:
            if result.file_path != symbol_info.file_path or result.line_number != symbol_info.line_number:
                reference = SymbolInfo(
                    name=symbol_info.name,
                    symbol_type=SymbolType.VARIABLE,  # References are typically variables
                    file_path=result.file_path,
                    line_number=result.line_number,
                    scope=SymbolScope.GLOBAL,
                    confidence_score=result.combined_score,
                    language=symbol_info.language,
                    metadata={"reference_type": "usage", "context": result.line_content},
                )
                references.append(reference)

        return references

    def analyze_symbol_relationships(self, symbol_info: SymbolInfo) -> Dict[str, List[SymbolInfo]]:
        """
        Analyze relationships between symbols (dependencies, usages, etc.).

        Args:
            symbol_info: SymbolInfo object to analyze

        Returns:
            Dictionary mapping relationship types to lists of related symbols
        """
        relationships: Dict[str, List[SymbolInfo]] = {
            "dependencies": [],
            "dependents": [],
            "related": [],
            "overrides": [],
            "implements": [],
        }

        # Find dependencies (symbols this one uses)
        if symbol_info.file_path:
            dependencies = self._find_symbol_dependencies(symbol_info)
            relationships["dependencies"] = dependencies

        # Find dependents (symbols that use this one)
        dependents = self.get_symbol_references(symbol_info)
        relationships["dependents"] = dependents

        # Find related symbols (semantic similarity)
        related = self._find_semantically_related_symbols(symbol_info)
        relationships["related"] = related

        return relationships

    def _perform_symbol_search(self, context: SymbolSearchContext) -> SymbolInfo:
        """Perform the actual symbol search with multiple strategies."""
        # Strategy 1: Exact pattern matching
        exact_matches = self._exact_symbol_search(context)

        # Strategy 2: Semantic search if enabled
        semantic_matches = []
        if context.use_semantic_search:
            semantic_matches = self._semantic_symbol_search(context)

        # Combine and rank results
        all_matches = exact_matches + semantic_matches

        if not all_matches:
            # Return empty SymbolInfo if no matches found
            return SymbolInfo(name=context.symbol_name, symbol_type=SymbolType.VARIABLE, file_path="", line_number=0, confidence_score=0.0)

        # Return best match
        best_match = max(all_matches, key=lambda x: x.confidence_score)
        return best_match

    def _exact_symbol_search(self, context: SymbolSearchContext) -> List[SymbolInfo]:
        """Perform exact symbol search using language patterns."""
        matches = []

        # Get language patterns
        lang_patterns = self._language_patterns.get(context.language, {})

        # Search for symbol name in patterns
        for symbol_type_str, pattern in lang_patterns.items():
            if context.symbol_type and symbol_type_str != context.symbol_type.value:
                continue

            # Create exact match pattern
            exact_pattern = pattern.replace(r"(\w+)", re.escape(context.symbol_name))

            try:
                from .ripgrep_integration import SearchOptions

                options = SearchOptions(file_types=[context.language])
                results = self.ripgrep.search_files(pattern=exact_pattern, path=context.file_path or ".", options=options)

                for result in results.results:
                    symbol_info = self._convert_search_result_to_symbol(result, symbol_type_str, context.language)
                    if symbol_info:
                        matches.append(symbol_info)

            except Exception as e:
                logger.warning(f"Error in exact search: {e}")

        return matches

    def _semantic_symbol_search(self, context: SymbolSearchContext) -> List[SymbolInfo]:
        """Perform semantic symbol search using AST analysis."""
        matches = []

        try:
            # Use enhanced search infrastructure for semantic search
            search_context = SearchContext(
                query=context.symbol_name, file_types=[context.language], strategy=SearchStrategy.SEMANTIC, max_results=20
            )

            results, _ = self.search_infra.search(search_context)

            # Convert results to symbol information
            for result in results:
                symbol_info = SymbolInfo(
                    name=context.symbol_name,
                    symbol_type=self._infer_symbol_type(result.line_content),
                    file_path=result.file_path,
                    line_number=result.line_number,
                    confidence_score=result.combined_score * 0.8,  # Lower confidence for semantic
                    language=context.language,
                    metadata={"semantic_match": True, "context": result.line_content},
                )
                matches.append(symbol_info)

        except Exception as e:
            logger.warning(f"Error in semantic search: {e}")

        return matches

    def _convert_search_result_to_symbol(self, result: SearchResult, symbol_type_str: str, language: str) -> Optional[SymbolInfo]:
        """Convert SearchResult to SymbolInfo."""
        try:
            # Extract symbol name from line using regex
            line_text = result.line_text.strip()

            # Use language-specific patterns to extract symbol name and type
            symbol_type = SymbolType.VARIABLE  # Default type
            is_private = False
            is_protected = False

            if language == "python":
                if "class " in line_text:
                    # Extract class name
                    class_match = re.search(r"class\s+(\w+)", line_text)
                    symbol_name = class_match.group(1) if class_match else line_text
                    symbol_type = SymbolType.CLASS
                elif "def " in line_text:
                    # Extract function name and check if private
                    func_match = re.search(r"def\s+(\w+)", line_text)
                    symbol_name = func_match.group(1) if func_match else line_text
                    symbol_type = SymbolType.FUNCTION
                    # Check for private method (starts with underscore)
                    if symbol_name.startswith("_"):
                        is_private = True
                    # Check for async function
                    if "async def" in line_text:
                        symbol_type = SymbolType.ASYNC_FUNCTION
                elif "=" in line_text and not line_text.startswith("#"):
                    # Extract variable name
                    var_match = re.search(r"(\w+)\s*=", line_text)
                    symbol_name = var_match.group(1) if var_match else line_text
                    symbol_type = SymbolType.VARIABLE
                    # Check for private variable
                    if symbol_name.startswith("_"):
                        is_private = True
                else:
                    symbol_name = line_text
            elif language == "javascript":
                if "class " in line_text:
                    class_match = re.search(r"class\s+(\w+)", line_text)
                    symbol_name = class_match.group(1) if class_match else line_text
                    symbol_type = SymbolType.CLASS
                elif "function " in line_text:
                    func_match = re.search(r"function\s+(\w+)", line_text)
                    symbol_name = func_match.group(1) if func_match else line_text
                    symbol_type = SymbolType.FUNCTION
                elif "const " in line_text or "let " in line_text or "var " in line_text:
                    var_match = re.search(r"(?:const|let|var)\s+(\w+)", line_text)
                    symbol_name = var_match.group(1) if var_match else line_text
                    symbol_type = SymbolType.VARIABLE
                else:
                    symbol_name = line_text
            else:
                symbol_name = line_text

            # Create SymbolInfo
            symbol_info = SymbolInfo(
                name=symbol_name,
                symbol_type=symbol_type,
                file_path=result.file_path,
                line_number=result.line_number,
                column_number=0,  # SearchResult doesn't have column_number
                confidence_score=1.0,
                language=language,
                is_private=is_private,
                is_protected=is_protected,
                metadata={"match_type": "exact", "context": result.line_text, "language": language},
            )

            return symbol_info

        except Exception as e:
            logger.warning(f"Error converting search result: {e}")
            return None

    def _string_to_symbol_type(self, type_str: str) -> SymbolType:
        """Convert string to SymbolType enum."""
        type_mapping = {
            "function": SymbolType.FUNCTION,
            "class": SymbolType.CLASS,
            "method": SymbolType.METHOD,
            "variable": SymbolType.VARIABLE,
            "import": SymbolType.IMPORT,
            "from_import": SymbolType.IMPORT,
            "interface": SymbolType.INTERFACE,
            "type": SymbolType.TYPE_ALIAS,
        }
        return type_mapping.get(type_str, SymbolType.VARIABLE)

    def _infer_symbol_type(self, line_content: str) -> SymbolType:
        """Infer symbol type from line content."""
        line_lower = line_content.lower().strip()

        if "def " in line_lower:
            if "async def" in line_lower:
                return SymbolType.ASYNC_FUNCTION
            return SymbolType.FUNCTION
        elif "class " in line_lower:
            return SymbolType.CLASS
        elif "import " in line_lower:
            return SymbolType.IMPORT
        elif "interface " in line_lower:
            return SymbolType.INTERFACE
        elif "type " in line_lower and "=" in line_lower:
            return SymbolType.TYPE_ALIAS
        else:
            return SymbolType.VARIABLE

    def _generate_symbol_cache_key(self, context: SymbolSearchContext) -> str:
        """Generate cache key for symbol search."""
        key_parts = [
            context.symbol_name,
            context.symbol_type.value if context.symbol_type else "any",
            context.scope or "global",
            context.file_path or "any",
            context.language,
        ]
        return "|".join(key_parts)

    def _parse_context(self, context: str) -> Dict[str, Any]:
        """Parse context string to extract useful information."""
        context_info = {
            "language": "python",  # Default
            "scope": "global",
            "imports": [],
            "classes": [],
            "functions": [],
        }

        # Simple language detection
        if "import " in context and "from " in context:
            context_info["language"] = "python"
        elif "function" in context and "class" in context:
            context_info["language"] = "javascript"
        elif "interface" in context or "type " in context:
            context_info["language"] = "typescript"

        return context_info

    def _filter_symbols_by_context(self, symbols: List[SymbolInfo], context_info: Dict[str, Any]) -> List[SymbolInfo]:
        """Filter symbols based on context relevance."""
        filtered = []

        for symbol in symbols:
            # Language filter
            if symbol.language != context_info["language"]:
                continue

            # Add context-specific filtering logic here
            filtered.append(symbol)

        return filtered

    def _determine_best_scope_match(self, symbols: List[SymbolInfo], context_info: Dict[str, Any]) -> Optional[SymbolInfo]:
        """Determine the best scope match from a list of symbols."""
        if not symbols:
            return None

        # Simple scoring system
        scored_symbols = []
        for symbol in symbols:
            score = symbol.confidence_score

            # Boost score for exact name matches
            if symbol.name == context_info.get("target_name", ""):
                score += 0.2

            # Boost score for appropriate scope
            if context_info.get("scope") == "class" and symbol.scope == SymbolScope.CLASS:
                score += 0.1

            scored_symbols.append((symbol, score))

        # Return symbol with highest score
        if scored_symbols:
            return max(scored_symbols, key=lambda x: x[1])[0]

        return symbols[0] if symbols else None

    def _find_symbol_dependencies(self, symbol_info: SymbolInfo) -> List[SymbolInfo]:
        """Find symbols that the given symbol depends on."""
        dependencies = []

        # This is a simplified implementation
        # In a full implementation, you would parse the file and analyze AST
        try:
            # Read the file containing the symbol
            with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple dependency detection (would be enhanced with AST analysis)
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                if i == symbol_info.line_number:
                    continue  # Skip the symbol definition itself

                # Look for potential dependencies
                for pattern in self._language_patterns.get(symbol_info.language, {}).values():
                    matches = re.findall(pattern, line)
                    for match in matches:
                        if match != symbol_info.name:
                            dep_symbol = SymbolInfo(
                                name=match,
                                symbol_type=SymbolType.VARIABLE,
                                file_path=symbol_info.file_path,
                                line_number=i,
                                confidence_score=0.5,
                                language=symbol_info.language,
                                metadata={"dependency_type": "usage"},
                            )
                            dependencies.append(dep_symbol)

        except Exception as e:
            logger.warning(f"Error finding dependencies: {e}")

        return dependencies

    def _find_semantically_related_symbols(self, symbol_info: SymbolInfo) -> List[SymbolInfo]:
        """Find symbols that are semantically related to the given symbol."""
        related = []

        try:
            # Use semantic search to find related symbols
            search_context = SearchContext(
                query=f"related to {symbol_info.name}", file_types=[symbol_info.language], strategy=SearchStrategy.SEMANTIC, max_results=10
            )

            results, _ = self.search_infra.search(search_context)

            for result in results:
                if result.file_path != symbol_info.file_path:
                    related_symbol = SymbolInfo(
                        name=result.line_content.split()[0] or "unknown",  # Extract first word as symbol name
                        symbol_type=self._infer_symbol_type(result.line_content),
                        file_path=result.file_path,
                        line_number=result.line_number,
                        confidence_score=result.combined_score * 0.6,
                        language=symbol_info.language,
                        metadata={"semantic_relation": True},
                    )
                    related.append(related_symbol)

        except Exception as e:
            logger.warning(f"Error finding semantic relations: {e}")

        return related

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for symbol operations."""
        return {
            "cache_size": len(self._symbol_cache),
            "scope_cache_size": len(self._scope_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "average_search_time": self._calculate_average_search_time(),
            "supported_languages": list(self._language_patterns.keys()),
        }

    def clear_cache(self):
        """Clear all caches."""
        self._symbol_cache.clear()
        self._scope_cache.clear()

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This would be implemented with actual hit/miss tracking
        return 0.85  # Placeholder

    def _calculate_average_search_time(self) -> float:
        """Calculate average search time (simplified)."""
        # This would be implemented with actual timing data
        return 0.05  # Placeholder
