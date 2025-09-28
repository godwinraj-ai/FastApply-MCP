"""
Deep Semantic Analysis Module

Advanced code understanding capabilities including intent analysis,
runtime behavior analysis, design pattern detection, and quality assessment.
"""

import ast
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CodeIntent(Enum):
    """Classification of code intent and purpose."""

    DATA_PROCESSING = "data_processing"
    BUSINESS_LOGIC = "business_logic"
    USER_INTERFACE = "user_interface"
    SYSTEM_CONFIGURATION = "system_configuration"
    ERROR_HANDLING = "error_handling"
    DATA_VALIDATION = "data_validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE_OPERATION = "database_operation"
    API_ENDPOINT = "api_endpoint"
    UTILITY_FUNCTION = "utility_function"
    CALCULATION = "calculation"
    TRANSFORMATION = "transformation"
    STATE_MANAGEMENT = "state_management"
    CACHE_OPERATION = "cache_operation"
    LOGGING = "logging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class DesignPattern(Enum):
    """Common design patterns that can be detected."""

    SINGLETON = "singleton"
    FACTORY = "factory"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    COMMAND = "command"
    ADAPTER = "adapter"
    DECORATOR = "decorator"
    FACADE = "facade"
    TEMPLATE_METHOD = "template_method"
    ITERATOR = "iterator"
    COMPOSITE = "composite"
    PROXY = "proxy"
    BUILDER = "builder"
    ABSTRACT_FACTORY = "abstract_factory"
    STATE = "state"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"
    VISITOR = "visitor"
    MEDIATOR = "mediator"
    MOMENTO = "momento"
    FLYWEIGHT = "flyweight"
    UNKNOWN = "unknown"


class CodeSmell(Enum):
    """Common code smells and anti-patterns."""

    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    DUPLICATE_CODE = "duplicate_code"
    LONG_PARAMETER_LIST = "long_parameter_list"
    FEATURE_ENVY = "feature_envy"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"
    LAZY_CLASS = "lazy_class"
    DATA_CLASS = "data_class"
    REFUSED_BEQUEST = "refused_bequest"
    SWITCH_STATEMENTS = "switch_statements"
    TEMPORARY_FIELD = "temporary_field"
    MESSAGE_CHAINS = "message_chains"
    MIDDLE_MAN = "middle_man"
    SPECULATIVE_GENERALITY = "speculative_generality"
    ALTERNATIVE_CLASSES = "alternative_classes"
    INCOMPLETE_LIBRARY_CLASS = "incomplete_library_class"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    NULL_CHECKS = "null_checks"
    COMPLEX_CONDITIONALS = "complex_conditionals"
    MAGIC_NUMBERS = "magic_numbers"
    GOD_OBJECT = "god_object"


@dataclass
class IntentAnalysis:
    """Analysis of code intent and purpose."""

    primary_intent: CodeIntent
    confidence: float
    supporting_evidence: List[str]
    contextual_clues: Dict[str, Any]
    alternative_intents: List[Tuple[CodeIntent, float]]


@dataclass
class BehaviorAnalysis:
    """Analysis of runtime behavior patterns."""

    execution_flow: List[str]
    side_effects: List[str]
    state_changes: List[str]
    resource_usage: Dict[str, Any]
    exception_handling: Dict[str, Any]
    performance_characteristics: Dict[str, Any]


@dataclass
class PatternAnalysis:
    """Analysis of design patterns and anti-patterns."""

    detected_patterns: List[Tuple[DesignPattern, float]]
    code_smells: List[Tuple[CodeSmell, float, str]]
    pattern_suggestions: List[str]
    refactoring_opportunities: List[str]


@dataclass
class QualityAssessment:
    """Comprehensive code quality assessment."""

    complexity_metrics: Dict[str, float]
    maintainability_score: float
    readability_score: float
    testability_score: float
    security_score: float
    overall_quality: float
    improvement_recommendations: List[str]


@dataclass
class SemanticAnalysis:
    """Complete semantic analysis result."""

    code_hash: str
    language: str
    intent_analysis: IntentAnalysis
    behavior_analysis: BehaviorAnalysis
    pattern_analysis: PatternAnalysis
    quality_assessment: QualityAssessment
    confidence_score: float
    analysis_timestamp: str


class AnalysisContext:
    """Context for semantic analysis operations."""

    def __init__(
        self,
        project_path: Optional[str] = None,
        language: str = "python",
        analysis_depth: str = "comprehensive",
        include_patterns: bool = True,
        include_quality: bool = True,
    ):
        self.project_path = project_path
        self.language = language
        self.analysis_depth = analysis_depth
        self.include_patterns = include_patterns
        self.include_quality = include_quality
        self.cached_results: Dict[str, SemanticAnalysis] = {}


class DeepSemanticAnalyzer:
    """Main class for deep semantic analysis of code."""

    def __init__(self):
        self.intent_patterns = self._load_intent_patterns()
        self.pattern_signatures = self._load_pattern_signatures()
        self.quality_thresholds = self._load_quality_thresholds()

    def analyze_semantics(self, code: str, language: str = "python", context: Optional[AnalysisContext] = None) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis of code.

        Args:
            code: The source code to analyze
            language: Programming language of the code
            context: Analysis context for configuration

        Returns:
            Complete semantic analysis result
        """
        context = context or AnalysisContext(language=language)

        # Generate code hash for caching
        code_hash = self._generate_code_hash(code)

        # Check cache
        if code_hash in context.cached_results:
            return context.cached_results[code_hash]

        # Perform analysis components
        intent_analysis = self.understand_code_intent(code, language, context)
        behavior_analysis = self.analyze_runtime_behavior(code, language)
        pattern_analysis = self.identify_design_patterns(code, language) if context.include_patterns else PatternAnalysis([], [], [], [])
        quality_assessment = (
            self.assess_code_quality(code, language) if context.include_quality else QualityAssessment({}, 0.5, 0.5, 0.5, 0.5, 0.5, [])
        )

        # Calculate overall confidence
        confidence_score = self._calculate_confidence(
            intent_analysis.confidence,
            behavior_analysis.execution_flow,
            pattern_analysis.detected_patterns,
            quality_assessment.overall_quality,
        )

        # Create comprehensive result
        result = SemanticAnalysis(
            code_hash=code_hash,
            language=language,
            intent_analysis=intent_analysis,
            behavior_analysis=behavior_analysis,
            pattern_analysis=pattern_analysis,
            quality_assessment=quality_assessment,
            confidence_score=confidence_score,
            analysis_timestamp=self._get_timestamp(),
        )

        # Cache result
        context.cached_results[code_hash] = result

        return result

    def understand_code_intent(self, code: str, language: str = "python", context: Optional[AnalysisContext] = None) -> IntentAnalysis:
        """
        Analyze the intent and purpose of code.

        Args:
            code: Source code to analyze
            language: Programming language
            context: Analysis context

        Returns:
            Intent analysis with confidence and evidence
        """
        # Extract linguistic clues
        keywords = self._extract_keywords(code)
        function_names = self._extract_function_names(code, language)
        variable_names = self._extract_variable_names(code, language)

        # Calculate intent probabilities
        intent_scores = self._calculate_intent_scores(keywords, function_names, variable_names, language)

        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = primary_intent[1]

        # Gather supporting evidence
        evidence = self._gather_intent_evidence(primary_intent[0], keywords, function_names, variable_names, code)

        # Analyze contextual clues
        contextual_clues = self._analyze_contextual_clues(code, language)

        # Alternative intents with scores
        alternative_intents = [(intent, score) for intent, score in intent_scores.items() if intent != primary_intent[0] and score > 0.3]

        return IntentAnalysis(
            primary_intent=primary_intent[0],
            confidence=confidence,
            supporting_evidence=evidence,
            contextual_clues=contextual_clues,
            alternative_intents=alternative_intents,
        )

    def analyze_runtime_behavior(self, code: str, language: str = "python") -> BehaviorAnalysis:
        """
        Analyze runtime behavior patterns of code.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Behavior analysis with execution patterns and characteristics
        """
        execution_flow = self._analyze_execution_flow(code, language)
        side_effects = self._identify_side_effects(code, language)
        state_changes = self._analyze_state_changes(code, language)
        resource_usage = self._estimate_resource_usage(code, language)
        exception_handling = self._analyze_exception_handling(code, language)
        performance_characteristics = self._analyze_performance_characteristics(code, language)

        return BehaviorAnalysis(
            execution_flow=execution_flow,
            side_effects=side_effects,
            state_changes=state_changes,
            resource_usage=resource_usage,
            exception_handling=exception_handling,
            performance_characteristics=performance_characteristics,
        )

    def identify_design_patterns(self, code: str, language: str = "python") -> PatternAnalysis:
        """
        Identify design patterns and anti-patterns in code.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Pattern analysis with detected patterns and code smells
        """
        detected_patterns = self._detect_design_patterns(code, language)
        code_smells = self._detect_code_smells(code, language)
        pattern_suggestions = self._generate_pattern_suggestions(detected_patterns, code_smells)
        refactoring_opportunities = self._identify_refactoring_opportunities(code_smells, detected_patterns)

        return PatternAnalysis(
            detected_patterns=detected_patterns,
            code_smells=code_smells,
            pattern_suggestions=pattern_suggestions,
            refactoring_opportunities=refactoring_opportunities,
        )

    def assess_code_quality(self, code: str, language: str = "python") -> QualityAssessment:
        """
        Assess overall code quality across multiple dimensions.

        Args:
            code: Source code to assess
            language: Programming language

        Returns:
            Comprehensive quality assessment
        """
        complexity_metrics = self._calculate_complexity_metrics(code, language)
        maintainability_score = self._calculate_maintainability_score(code, complexity_metrics)
        readability_score = self._calculate_readability_score(code, language)
        testability_score = self._calculate_testability_score(code, complexity_metrics)
        security_score = self._calculate_security_score(code, language)

        overall_quality = self._calculate_overall_quality(maintainability_score, readability_score, testability_score, security_score)

        improvement_recommendations = self._generate_improvement_recommendations(
            complexity_metrics, maintainability_score, readability_score, testability_score, security_score
        )

        return QualityAssessment(
            complexity_metrics=complexity_metrics,
            maintainability_score=maintainability_score,
            readability_score=readability_score,
            testability_score=testability_score,
            security_score=security_score,
            overall_quality=overall_quality,
            improvement_recommendations=improvement_recommendations,
        )

    # Private helper methods for pattern loading
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns."""
        return {
            CodeIntent.DATA_PROCESSING.value: ["process", "transform", "parse", "validate", "format"],
            CodeIntent.BUSINESS_LOGIC.value: ["business", "logic", "calculate", "compute", "rule"],
            CodeIntent.USER_INTERFACE.value: ["ui", "view", "render", "display", "component"],
            CodeIntent.ERROR_HANDLING.value: ["error", "exception", "handle", "catch", "raise"],
            CodeIntent.AUTHENTICATION.value: ["auth", "login", "verify", "credential", "session"],
            CodeIntent.DATABASE_OPERATION.value: ["db", "query", "insert", "update", "delete"],
            CodeIntent.API_ENDPOINT.value: ["api", "endpoint", "route", "handler", "controller"],
        }

    def _load_pattern_signatures(self) -> Dict[str, List[str]]:
        """Load design pattern signatures."""
        return {
            DesignPattern.SINGLETON.value: ["_instance", "getInstance", "__new__"],
            DesignPattern.FACTORY.value: ["create", "factory", "build"],
            DesignPattern.OBSERVER.value: ["observer", "notify", "subscribe", "listener"],
            DesignPattern.STRATEGY.value: ["strategy", "algorithm", "context"],
        }

    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality assessment thresholds."""
        return {
            "max_method_length": 50,
            "max_class_length": 300,
            "max_parameters": 7,
            "max_complexity": 10,
            "min_maintainability": 65,
            "min_readability": 70,
            "min_testability": 60,
        }

    # Additional helper methods for analysis components
    def _generate_code_hash(self, code: str) -> str:
        """Generate hash for code caching."""
        import hashlib

        return hashlib.md5(code.encode()).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime

        return datetime.datetime.now().isoformat()

    def _extract_keywords(self, code: str) -> List[str]:
        """Extract meaningful keywords from code."""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
        return [word.lower() for word in words if len(word) > 2]

    def _extract_function_names(self, code: str, language: str) -> List[str]:
        """Extract function names from code."""
        if language == "python":
            try:
                tree = ast.parse(code)
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                return functions
            except Exception:
                return []
        return []

    def _extract_variable_names(self, code: str, language: str) -> List[str]:
        """Extract variable names from code."""
        if language == "python":
            try:
                tree = ast.parse(code)
                variables = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        variables.append(node.id)
                return variables
            except Exception:
                return []
        return []

    def _calculate_intent_scores(
        self, keywords: List[str], function_names: List[str], variable_names: List[str], language: str
    ) -> Dict[CodeIntent, float]:
        """Calculate probability scores for each intent."""
        scores = {intent: 0.0 for intent in CodeIntent}

        all_terms = keywords + function_names + variable_names

        for intent, patterns in self.intent_patterns.items():
            matches = sum(1 for term in all_terms if any(pattern in term.lower() for pattern in patterns))
            if matches > 0:
                scores[CodeIntent(intent)] = min(matches / len(all_terms), 1.0)

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for intent in scores:
                scores[intent] = scores[intent] / total_score

        return scores

    def _gather_intent_evidence(
        self, intent: CodeIntent, keywords: List[str], function_names: List[str], variable_names: List[str], code: str
    ) -> List[str]:
        """Gather evidence supporting the determined intent."""
        evidence = []

        # Check for intent-specific keywords
        if intent.value in self.intent_patterns:
            patterns = self.intent_patterns[intent.value]
            matches = [kw for kw in keywords if any(pattern in kw for pattern in patterns)]
            if matches:
                evidence.append(f"Found relevant keywords: {', '.join(matches[:3])}")

        # Check function names
        relevant_functions = [fn for fn in function_names if any(pattern in fn.lower() for pattern in patterns)]
        if relevant_functions:
            evidence.append(f"Functions with intent-relevant names: {', '.join(relevant_functions[:2])}")

        # Check code structure clues
        if intent == CodeIntent.ERROR_HANDLING:
            if "try:" in code or "except" in code:
                evidence.append("Contains exception handling structure")

        return evidence

    def _analyze_contextual_clues(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze contextual clues for intent determination."""
        clues: Dict[str, Any] = {}

        # Code structure clues
        clues["has_classes"] = "class " in code
        clues["has_functions"] = "def " in code if language == "python" else "function " in code
        clues["has_loops"] = any(keyword in code for keyword in ["for ", "while "])
        clues["has_conditionals"] = any(keyword in code for keyword in ["if ", "else", "elif "])
        clues["has_imports"] = "import " in code

        # Comment analysis
        comments = re.findall(r"#.*$", code, re.MULTILINE) if language == "python" else []
        clues["has_comments"] = len(comments) > 0
        clues["comment_ratio"] = float(len(comments)) / max(len(code.split("\n")), 1)

        return clues

    def _analyze_execution_flow(self, code: str, language: str) -> List[str]:
        """Analyze execution flow patterns."""
        flow_patterns = []

        if language == "python":
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        flow_patterns.append("conditional_branch")
                    elif isinstance(node, (ast.For, ast.While)):
                        flow_patterns.append("loop")
                    elif isinstance(node, ast.FunctionDef):
                        flow_patterns.append("function_call")
                    elif isinstance(node, ast.Try):
                        flow_patterns.append("exception_handling")
            except Exception:
                flow_patterns.append("unparseable")

        return flow_patterns if flow_patterns else ["linear"]

    def _identify_side_effects(self, code: str, language: str) -> List[str]:
        """Identify potential side effects."""
        side_effects = []

        # File operations
        file_operations = ["open(", "write(", "read(", "delete("]
        for op in file_operations:
            if op in code:
                side_effects.append("file_operation")

        # Network operations
        network_operations = ["http", "request", "socket", "connect"]
        for op in network_operations:
            if op.lower() in code.lower():
                side_effects.append("network_operation")

        # Database operations
        db_operations = ["query", "insert", "update", "delete", "execute"]
        for op in db_operations:
            if op.lower() in code.lower():
                side_effects.append("database_operation")

        return side_effects if side_effects else ["no_side_effects"]

    def _analyze_state_changes(self, code: str, language: str) -> List[str]:
        """Analyze state changes in code."""
        state_changes = []

        if language == "python":
            try:
                tree = ast.parse(code)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                state_changes.append(f"variable_assignment:{target.id}")
                    elif isinstance(node, ast.AugAssign):
                        if isinstance(node.target, ast.Name):
                            state_changes.append(f"variable_modification:{node.target.id}")
            except Exception:
                pass

        return state_changes if state_changes else ["no_state_changes"]

    def _estimate_resource_usage(self, code: str, language: str) -> Dict[str, Any]:
        """Estimate resource usage patterns."""
        usage = {
            "memory_usage": "low",
            "cpu_usage": "low",
            "io_operations": 0,
            "network_calls": 0,
        }

        # Count I/O operations
        usage["io_operations"] = code.count("open(") + code.count("write(") + code.count("read(")

        # Count network calls
        usage["network_calls"] = len(re.findall(r"(http|request|socket|connect)", code.lower()))

        # Estimate memory usage
        large_data_structures = code.count("[") + code.count("{") + code.count("dict(") + code.count("list(")
        if large_data_structures > 10:
            usage["memory_usage"] = "high"
        elif large_data_structures > 5:
            usage["memory_usage"] = "medium"

        # Estimate CPU usage
        loops = code.count("for ") + code.count("while ")
        if loops > 5:
            usage["cpu_usage"] = "high"
        elif loops > 2:
            usage["cpu_usage"] = "medium"

        return usage

    def _analyze_exception_handling(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze exception handling patterns."""
        handling = {
            "has_try_blocks": False,
            "has_specific_exceptions": False,
            "has_generic_exceptions": False,
            "has_finally_blocks": False,
            "exception_count": 0,
        }

        if language == "python":
            handling["has_try_blocks"] = "try:" in code
            handling["has_finally_blocks"] = "finally:" in code

            # Count exception types
            exception_types = ["ValueError", "TypeError", "IndexError", "KeyError", "AttributeError"]
            for exc_type in exception_types:
                if exc_type in code:
                    handling["has_specific_exceptions"] = True
                    handling["exception_count"] += code.count(exc_type)

            # Generic exceptions
            if "Exception" in code:
                handling["has_generic_exceptions"] = True

        return handling

    def _analyze_performance_characteristics(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        # NOTE: Explicit typed structure to avoid implicit bool/int confusion in later float usage
        perf: Dict[str, Any] = {
            "time_complexity": "unknown",
            "space_complexity": "unknown",
            "has_recursion": False,  # remains boolean
            "nested_loop_depth": 0,  # integer depth
            "has_optimization_opportunities": False,
        }

        # Check for recursion
        perf["has_recursion"] = "def " in code and any(func_name in code for func_name in self._extract_function_names(code, language))

        # Calculate nested loop depth
        lines = code.split("\n")
        current_depth = 0
        max_depth = 0
        for line in lines:
            if any(keyword in line for keyword in ["for ", "while "]):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif "pass" in line or line.strip() == "":
                current_depth = max(0, current_depth - 1)
        perf["nested_loop_depth"] = max_depth

        # Estimate complexity
        if max_depth >= 3:
            perf["time_complexity"] = "high"
        elif max_depth >= 2:
            perf["time_complexity"] = "medium"
        elif max_depth == 1:
            perf["time_complexity"] = "low"

        return perf

    def _detect_design_patterns(self, code: str, language: str) -> List[Tuple[DesignPattern, float]]:
        """Detect design patterns in code."""
        detected = []

        if language == "python":
            try:
                ast.parse(code)

                # Singleton pattern detection
                singleton_indicators = ["_instance", "getInstance", "__new__"]
                for indicator in singleton_indicators:
                    if indicator in code:
                        detected.append((DesignPattern.SINGLETON, 0.8))
                        break

                # Factory pattern detection
                factory_indicators = ["create", "factory", "build"]
                for indicator in factory_indicators:
                    if any(indicator in func_name.lower() for func_name in self._extract_function_names(code, language)):
                        detected.append((DesignPattern.FACTORY, 0.7))
                        break

                # Observer pattern detection
                observer_indicators = ["observer", "notify", "subscribe", "listener"]
                for indicator in observer_indicators:
                    if indicator.lower() in code.lower():
                        detected.append((DesignPattern.OBSERVER, 0.7))
                        break

            except Exception:
                pass

        return detected if detected else [(DesignPattern.UNKNOWN, 0.0)]

    def _detect_code_smells(self, code: str, language: str) -> List[Tuple[CodeSmell, float, str]]:
        """Detect code smells and anti-patterns."""
        smells = []

        # Long method smell
        line_count = len(code.split("\n"))
        if line_count > self.quality_thresholds["max_method_length"]:
            smells.append((CodeSmell.LONG_METHOD, 0.9, f"Method is {line_count} lines long"))

        # Complex conditionals
        if code.count("if ") + code.count("elif ") > 3:
            smells.append((CodeSmell.COMPLEX_CONDITIONALS, 0.7, "Too many conditional branches"))

        # Magic numbers
        magic_numbers = re.findall(r"\b\d+\b", code)
        if len(magic_numbers) > 5:
            smells.append((CodeSmell.MAGIC_NUMBERS, 0.6, "Multiple magic numbers found"))

        # Duplicate code (simplified check)
        lines = code.split("\n")
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.7:
            smells.append((CodeSmell.DUPLICATE_CODE, 0.5, "Potential code duplication"))

        return smells

    def _generate_pattern_suggestions(
        self, detected_patterns: List[Tuple[DesignPattern, float]], code_smells: List[Tuple[CodeSmell, float, str]]
    ) -> List[str]:
        """Generate pattern improvement suggestions."""
        suggestions = []

        for pattern, confidence in detected_patterns:
            if pattern == DesignPattern.SINGLETON and confidence > 0.7:
                suggestions.append("Consider thread-safe implementation for singleton pattern")
            elif pattern == DesignPattern.FACTORY and confidence > 0.7:
                suggestions.append("Consider using abstract factory for better extensibility")

        for smell, confidence, description in code_smells:
            if smell == CodeSmell.LONG_METHOD and confidence > 0.7:
                suggestions.append("Break down long method into smaller, focused functions")
            elif smell == CodeSmell.COMPLEX_CONDITIONALS and confidence > 0.7:
                suggestions.append("Use strategy pattern to simplify complex conditionals")

        return suggestions

    def _identify_refactoring_opportunities(
        self, code_smells: List[Tuple[CodeSmell, float, str]], detected_patterns: List[Tuple[DesignPattern, float]]
    ) -> List[str]:
        """Identify refactoring opportunities."""
        opportunities = []

        # Group related code smells
        severity_scores = sum(confidence for _, confidence, _ in code_smells)

        if severity_scores > 2.0:
            opportunities.append("Major refactoring needed: multiple code smells detected")
        elif severity_scores > 1.0:
            opportunities.append("Moderate refactoring: several improvement areas identified")

        # Pattern-specific opportunities
        for pattern, confidence in detected_patterns:
            if confidence > 0.8:
                opportunities.append(f"Enhance {pattern.value} pattern implementation")

        return opportunities

    def _calculate_complexity_metrics(self, code: str, language: str) -> Dict[str, float]:
        """Calculate cyclomatic and cognitive complexity metrics."""
        # Ensure metrics is explicitly typed so mypy knows all numeric values are floats
        metrics: Dict[str, float] = {}

        if language == "python":
            try:
                tree = ast.parse(code)

                # Cyclomatic complexity
                decision_points = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        decision_points += 1
                    elif isinstance(node, ast.BoolOp):
                        decision_points += len(node.values) - 1

                metrics["cyclomatic_complexity"] = float(decision_points + 1)

                # Cognitive complexity (simplified)
                metrics["cognitive_complexity"] = float(self._calculate_cognitive_complexity(code))

                # Lines of code
                metrics["lines_of_code"] = float(len(code.split("\n")))

                # Number of statements
                statements = 0
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Assign, ast.Expr, ast.Return, ast.Raise)):
                        statements += 1
                metrics["statements"] = float(statements)

            except Exception:
                metrics["cyclomatic_complexity"] = 1.0
                metrics["cognitive_complexity"] = 1.0
                metrics["lines_of_code"] = float(len(code.split("\n")))
                metrics["statements"] = 1.0

        return metrics

    def _calculate_cognitive_complexity(self, code: str) -> float:
        """Calculate cognitive complexity (simplified version)."""
        complexity = 0

        # Nesting levels
        nesting = 0
        for line in code.split("\n"):
            if any(keyword in line for keyword in ["if ", "for ", "while ", "try:", "with "]):
                nesting += 1
                complexity += nesting
            elif line.strip() in ["pass", ""] or line.startswith("return"):
                nesting = max(0, nesting - 1)

        return complexity

    def _calculate_maintainability_score(self, code: str, complexity_metrics: Dict[str, float]) -> float:
        """Calculate maintainability index."""
        # Halstead volume (simplified)
        operators = len(re.findall(r"[\+\-\*\/\=\!\>\<\&\|\%]", code))
        operands = len(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code))
        volume = (operators + operands) * 0.001  # Simplified calculation

        # Maintainability index (simplified)
        loc = complexity_metrics.get("lines_of_code", 1)
        complexity = complexity_metrics.get("cyclomatic_complexity", 1)

        mi = max(0, 171 - 5.2 * volume - 0.23 * complexity - 16.2 * math.log(loc) if loc > 0 else 100)
        return min(100, mi) / 100.0

    def _calculate_readability_score(self, code: str, language: str) -> float:
        """Calculate readability score."""
        score = 0.5  # Base score

        # Comment ratio
        comments = len(re.findall(r"#.*$", code, re.MULTILINE))
        lines = len(code.split("\n"))
        comment_ratio = comments / max(lines, 1)
        score += min(comment_ratio * 2, 0.3)  # Up to 0.3 for comments

        # Function and variable naming
        function_names = self._extract_function_names(code, language)
        good_names = sum(1 for name in function_names if len(name) > 3 and name.islower())
        if function_names:
            score += min(good_names / len(function_names) * 0.2, 0.2)

        # Code structure
        if "class " in code:
            score += 0.1  # Object-oriented structure

        # Whitespace and formatting
        if code.count("\n\n") > 0:  # Has blank lines for separation
            score += 0.1

        return min(score, 1.0)

    def _calculate_testability_score(self, code: str, complexity_metrics: Dict[str, float]) -> float:
        """Calculate testability score."""
        score = 1.0

        # Deduct for complexity
        complexity = complexity_metrics.get("cyclomatic_complexity", 1)
        if complexity > 10:
            score -= 0.3
        elif complexity > 5:
            score -= 0.1

        # Deduct for dependencies
        imports = code.count("import ")
        if imports > 10:
            score -= 0.2
        elif imports > 5:
            score -= 0.1

        # Deduct for external dependencies
        external_deps = len(re.findall(r"from\s+\w+\s+import", code))
        if external_deps > 5:
            score -= 0.2

        return max(0.0, score)

    def _calculate_security_score(self, code: str, language: str) -> float:
        """Calculate security score."""
        score = 1.0

        # Check for security issues
        security_issues = []

        # Hardcoded secrets
        if any(keyword in code.lower() for keyword in ["password", "secret", "key", "token"]):
            security_issues.append("potential_hardcoded_secrets")

        # SQL injection risks
        if "execute(" in code and "%" in code:
            security_issues.append("potential_sql_injection")

        # File path operations
        if "open(" in code and "+" in code:
            security_issues.append("potential_path_injection")

        # Deduct for security issues
        score -= len(security_issues) * 0.3

        return max(0.0, score)

    def _calculate_overall_quality(self, maintainability: float, readability: float, testability: float, security: float) -> float:
        """Calculate overall quality score."""
        return maintainability * 0.3 + readability * 0.2 + testability * 0.3 + security * 0.2

    def _generate_improvement_recommendations(
        self,
        complexity_metrics: Dict[str, float],
        maintainability_score: float,
        readability_score: float,
        testability_score: float,
        security_score: float,
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Maintainability recommendations
        if maintainability_score < 0.7:
            recommendations.append("Reduce method complexity and improve code structure")

        # Readability recommendations
        if readability_score < 0.7:
            recommendations.append("Add more comments and improve variable naming")

        # Testability recommendations
        if testability_score < 0.7:
            recommendations.append("Reduce dependencies and improve function isolation")

        # Security recommendations
        if security_score < 0.7:
            recommendations.append("Address potential security vulnerabilities")

        # Complexity recommendations
        complexity = complexity_metrics.get("cyclomatic_complexity", 1)
        if complexity > 10:
            recommendations.append("Break down complex functions into smaller units")

        return recommendations

    def _calculate_confidence(
        self,
        intent_confidence: float,
        execution_flow: List[str],
        detected_patterns: List[Tuple[DesignPattern, float]],
        overall_quality: float,
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence_factors = [
            intent_confidence,
            min(len(execution_flow) / 5, 1.0),  # More flow patterns = higher confidence
            max([conf for _, conf in detected_patterns], default=0.5),
            overall_quality,
        ]

        return sum(confidence_factors) / len(confidence_factors)
