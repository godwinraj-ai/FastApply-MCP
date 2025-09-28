"""
LLM-Based AST Rule Intelligence System

Follows LLM reasoning patterns from @cctools.mdc instead of traditional software architecture.
Implements collective consciousness through Qdrant learning and direct CLI integration.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import structlog

# Optional YAML import for advanced rule generation
try:
    import yaml  # type: ignore[import-untyped]

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

    # Create a minimal yaml fallback for basic functionality
    class YamlFallback:
        @staticmethod
        def dump(data: Any, **kwargs: Any) -> str:
            return json.dumps(data, indent=2)

        @staticmethod
        def safe_dump(data: Any, **kwargs: Any) -> str:
            return json.dumps(data, indent=2)

        @staticmethod
        def safe_load(data: Any) -> Any:
            return json.loads(data) if isinstance(data, str) else data

        @staticmethod
        def load(data: Any) -> Any:
            return json.loads(data) if isinstance(data, str) else data

    # Alias for compatibility
    yaml = YamlFallback()  # type: ignore[assignment]

# Direct CLI integration - import only from ast_search_official.py
try:
    from .ast_search_official import dump_syntax_tree, find_code_by_rule, is_ast_grep_available, run_ast_grep, test_match_code_rule

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

    # Fallback functions for development with matching signatures
    def run_ast_grep(args: List[str], cwd: Optional[str] = None, input_text: Optional[str] = None) -> str:  # type: ignore[unused-ignore]
        return ""

    def test_match_code_rule(code: str, rule_yaml: str) -> str:  # type: ignore[unused-ignore]
        return "[]"


def is_available() -> bool:
    """Check if LLM-based AST intelligence system is available."""
    return CLI_AVAILABLE


# Qdrant integration for collective consciousness
try:
    from mcp__remote_sse_server import qdrant_find, qdrant_store

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

    async def qdrant_store(*args, **kwargs):
        pass

    async def qdrant_find(*args, **kwargs):
        return []


logger = structlog.get_logger(__name__)


@dataclass
class Intent:
    """Represents user intent from natural language query"""

    primary_goal: str
    constraints: List[str]
    code_elements: List[str]
    language_hint: Optional[str] = None
    complexity: str = "simple"


@dataclass
class RuleComponents:
    """Broken down components for rule construction"""

    atomic_rules: List[Dict]
    relational_rules: List[Dict]
    composite_rules: List[Dict]
    meta_variables: List[str]
    example_snippet: str


@dataclass
class Experience:
    """Learning experience stored in Qdrant"""

    query: str
    rule: Dict
    success: bool
    feedback: str
    timestamp: datetime
    language: str


class LLMAstReasoningEngine:
    """LLM-based reasoning engine for ast-grep rule generation"""

    def __init__(self):
        self.cli_functions = {
            "run_ast_grep": run_ast_grep,
            "test_match_code_rule": test_match_code_rule,
            "find_code_by_rule": find_code_by_rule,
            "dump_syntax_tree": dump_syntax_tree,
        }

        # Pattern library following cctools.mdc 36-function approach
        self.pattern_templates = self._initialize_pattern_library()

        # Learning state
        self.learning_cache = {}
        self.success_patterns = []
        self.failure_patterns = []

        logger.info("LLM AST Reasoning Engine initialized", cli_available=CLI_AVAILABLE, qdrant_available=QDRANT_AVAILABLE)

    def _initialize_pattern_library(self) -> Dict:
        """Initialize 36-function pattern library from cctools.mdc"""
        return {
            # Atomic Rules
            "atomic": {
                "pattern": {"description": "Match by code pattern with metavariables", "template": lambda code: {"pattern": code}},
                "kind": {"description": "Match by AST node type", "template": lambda node_type: {"kind": node_type}},
                "regex": {"description": "Match by text regex", "template": lambda pattern: {"regex": pattern}},
                "nthChild": {"description": "Match by position", "template": lambda position: {"nthChild": position}},
                "range": {
                    "description": "Match by character range",
                    "template": lambda start, end: {"range": {"start": start, "end": end}},
                },
            },
            # Relational Rules (always with stopBy: end per cctools.mdc)
            "relational": {
                "inside": {"description": "Target inside parent node", "template": lambda rule: {"inside": {**rule, "stopBy": "end"}}},
                "has": {"description": "Target has descendant", "template": lambda rule: {"has": {**rule, "stopBy": "end"}}},
                "precedes": {"description": "Target appears before", "template": lambda rule: {"precedes": rule}},
                "follows": {"description": "Target appears after", "template": lambda rule: {"follows": rule}},
            },
            # Composite Rules
            "composite": {
                "all": {"description": "All rules must match (AND)", "template": lambda rules: {"all": rules}},
                "any": {"description": "Any rule must match (OR)", "template": lambda rules: {"any": rules}},
                "not": {"description": "Rule must not match (NOT)", "template": lambda rule: {"not": rule}},
                "matches": {"description": "Match utility rule", "template": lambda rule_id: {"matches": rule_id}},
            },
            # Meta-variable patterns
            "metavars": {"single": "$VAR", "unnamed": "$$VAR", "multi": "$$$VAR", "non_capturing": "_VAR"},
        }

    async def reason_and_generate_rule(self, query: str, language: str = "python") -> Dict:
        """
        Main LLM reasoning loop following cctools.mdc 7-step process:
        1. Understand query → 2. Break down → 3. Create example →
        4. Write rule → 5. Test → 6. Debug → 7. Iterate
        """
        try:
            # Step 1: Check Qdrant for similar past experiences
            past_experiences = await self._retrieve_similar_experiences(query, language)

            # Step 2: Understand query with context from past experiences
            intent = self._understand_query(query, past_experiences)

            # Step 3: Break down query into components using learned patterns
            components = self._breakdown_query(intent, past_experiences)

            # Step 4: Generate example code snippet with memory
            example = self._generate_example(components, past_experiences)

            # Step 5: Construct initial rule using CLI functions
            rule = self._construct_rule(components, example, language)

            # Step 6: Test and debug iteratively using LLM logic
            final_rule = await self._test_and_debug_rule(rule, example, language)

            # Step 7: Store experience in collective memory
            await self._store_experience(query, final_rule, True, "Success", language)

            logger.info("LLM reasoning completed", query=query, rule_type=type(final_rule).__name__, learned_from=len(past_experiences))

            return final_rule

        except Exception as e:
            logger.error("LLM reasoning failed", error=str(e), query=query)
            # Store failure experience
            await self._store_experience(query, {}, False, str(e), language)
            raise

    def _understand_query(self, query: str, past_experiences: List[Experience]) -> Intent:
        """Step 1: Understand natural language query with context"""
        query_lower = query.lower()

        # Extract code elements from query
        code_elements = []
        if "function" in query_lower:
            code_elements.append("function")
        if "class" in query_lower:
            code_elements.append("class")
        if "variable" in query_lower:
            code_elements.append("variable")
        if "import" in query_lower:
            code_elements.append("import")
        if "call" in query_lower:
            code_elements.append("call_expression")

        # Determine language hint
        language_hint = None
        for lang in ["python", "javascript", "typescript", "java", "cpp"]:
            if lang in query_lower:
                language_hint = lang
                break

        # Determine complexity based on query structure
        complexity = "simple"
        if any(word in query_lower for word in ["complex", "nested", "multiple", "relationship"]):
            complexity = "complex"
        elif any(word in query_lower for word in ["pattern", "structure", "hierarchy"]):
            complexity = "moderate"

        # Extract constraints
        constraints = []
        if "not" in query_lower:
            constraints.append("negation")
        if "inside" in query_lower or "within" in query_lower:
            constraints.append("contextual")
        if "before" in query_lower or "after" in query_lower:
            constraints.append("sequential")

        # Learn from past experiences
        if past_experiences:
            successful_intents = [exp for exp in past_experiences if exp.success]
            if successful_intents:
                # Use most successful past intent as reference
                reference_intent = successful_intents[0]
                if reference_intent.feedback:
                    constraints.extend(reference_intent.feedback.split(","))

        return Intent(
            primary_goal=query_lower.split()[0],  # Simple heuristic
            constraints=constraints,
            code_elements=code_elements,
            language_hint=language_hint,
            complexity=complexity,
        )

    def _breakdown_query(self, intent: Intent, past_experiences: List[Experience]) -> RuleComponents:
        """Step 2: Break down query into rule components using learned patterns"""

        # Initialize empty components
        atomic_rules = []
        relational_rules = []
        composite_rules = []
        meta_variables = []

        # Generate atomic rules based on code elements
        for element in intent.code_elements:
            if element == "function":
                atomic_rules.append({"kind": "function_declaration"})
                meta_variables.append("$FUNC")
            elif element == "class":
                atomic_rules.append({"kind": "class_definition"})
                meta_variables.append("$CLASS")
            elif element == "call_expression":
                atomic_rules.append({"kind": "call_expression"})
                meta_variables.append("$CALL")
            elif element == "import":
                atomic_rules.append({"kind": "import_statement"})
                meta_variables.append("$IMPORT")

        # Add relational rules based on constraints
        if "contextual" in intent.constraints:
            # Add 'inside' or 'has' rules with stopBy: end (per cctools.mdc)
            if atomic_rules:
                relational_rules.append(self.pattern_templates["relational"]["inside"]["template"](atomic_rules[0]))

        if "sequential" in intent.constraints:
            # Add 'precedes' or 'follows' rules
            if len(atomic_rules) >= 2:
                relational_rules.append(self.pattern_templates["relational"]["precedes"]["template"](atomic_rules[1]))

        # Add composite rules for complex queries
        if intent.complexity == "complex" or len(atomic_rules) > 1:
            composite_rules.append(self.pattern_templates["composite"]["all"]["template"](atomic_rules))
        elif intent.complexity == "moderate":
            composite_rules.append(self.pattern_templates["composite"]["any"]["template"](atomic_rules))

        # Generate example snippet
        # Coerce optional language hint to default 'python'
        language_for_example = intent.language_hint or "python"
        example_snippet = self._generate_example_from_components(atomic_rules, relational_rules, language_for_example)

        # Learn from past experiences
        for experience in past_experiences:
            if experience.success:
                # Incorporate successful patterns
                if isinstance(experience.rule, dict):
                    if "atomic" in experience.rule and isinstance(experience.rule.get("atomic"), list):
                        atomic_rules.extend(experience.rule["atomic"])  # type: ignore[arg-type]
                    if "relational" in experience.rule and isinstance(experience.rule.get("relational"), list):
                        relational_rules.extend(experience.rule["relational"])  # type: ignore[arg-type]

        return RuleComponents(
            atomic_rules=atomic_rules,
            relational_rules=relational_rules,
            composite_rules=composite_rules,
            meta_variables=meta_variables,
            example_snippet=example_snippet,
        )

    def _generate_example_from_components(
        self,
        atomic_rules: List[Dict[str, Any]],
        relational_rules: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Generate example code snippet from rule components"""

        if language == "python":
            if any(rule.get("kind") == "function_declaration" for rule in atomic_rules):
                return "def example_function($PARAM): \n    return $RESULT"
            elif any(rule.get("kind") == "class_definition" for rule in atomic_rules):
                return "class ExampleClass:\n    def method(self):\n        pass"
            elif any(rule.get("kind") == "call_expression" for rule in atomic_rules):
                return "example_function($ARG)"
            else:
                return "$VARIABLE = $VALUE"

        elif language == "javascript":
            if any(rule.get("kind") == "function_declaration" for rule in atomic_rules):
                return "function exampleFunction($PARAM) {\n    return $RESULT;\n}"
            elif any(rule.get("kind") == "class_definition" for rule in atomic_rules):
                return "class ExampleClass {\n    method() {\n        // implementation\n    }\n}"
            else:
                return "const $VARIABLE = $VALUE;"

        else:
            # Generic example
            return "$EXAMPLE_PATTERN"

    def _generate_example(self, components: RuleComponents, past_experiences: List[Experience]) -> str:
        """Step 3: Generate example code snippet with memory"""

        # Start with base example from components
        example = components.example_snippet

        # Enhance with learned patterns from past experiences
        for experience in past_experiences:
            if experience.success and "example" in experience.rule:
                # Use successful example patterns
                learned_example = experience.rule["example"]
                if len(learned_example) > len(example):
                    example = learned_example  # Prefer more detailed examples

        # Ensure example includes meta-variables
        for metavar in components.meta_variables:
            if metavar not in example:
                example = example.replace("$EXAMPLE", metavar)

        return example

    def _construct_rule(self, components: RuleComponents, example: str, language: str) -> Dict[str, Any]:
        """Step 4: Construct ast-grep rule using CLI functions"""
        # Explicit typing to help mypy understand nested rule shape
        rule: Dict[str, Any] = {"id": "generated-rule", "language": language, "rule": {}, "example": example}

        # Start with the most specific atomic rule
        if components.atomic_rules:
            if len(components.atomic_rules) == 1:
                rule["rule"] = components.atomic_rules[0]
            else:
                # Use composite rule for multiple atomic rules
                rule["rule"] = self.pattern_templates["composite"]["all"]["template"](components.atomic_rules)

        # Add relational rules
        if components.relational_rules:
            if "all" not in rule["rule"]:
                rule["rule"] = self.pattern_templates["composite"]["all"]["template"]([rule["rule"], *components.relational_rules])
            else:
                rule["rule"]["all"].extend(components.relational_rules)

        # Add composite rules
        if components.composite_rules:
            if len(rule["rule"]) == 0:
                rule["rule"] = components.composite_rules[0]
            elif "all" not in rule["rule"]:
                rule["rule"] = self.pattern_templates["composite"]["all"]["template"]([rule["rule"], *components.composite_rules])

        return rule

    async def _test_and_debug_rule(self, rule: Dict, example: str, language: str) -> Dict:
        """Step 5-6: Test and debug rule using LLM logic patterns"""

        if not CLI_AVAILABLE:
            logger.warning("CLI not available, skipping testing")
            return rule

        max_iterations = 5
        current_rule = rule.copy()

        for iteration in range(max_iterations):
            try:
                # Test rule against example using CLI function
                rule_yaml = yaml.dump(current_rule["rule"])
                test_result = test_match_code_rule(example, rule_yaml)

                # Parse test result
                try:
                    matches = json.loads(test_result)
                    successful = len(matches) > 0
                except (json.JSONDecodeError, Exception):
                    successful = False

                if successful:
                    logger.info("Rule test successful", iteration=iteration + 1)
                    return current_rule

                # Debug using LLM logic patterns from cctools.mdc
                current_rule = self._debug_with_llm_logic(current_rule, example, iteration)

            except Exception as e:
                logger.warning("Rule testing failed", iteration=iteration + 1, error=str(e))
                current_rule = self._debug_with_llm_logic(current_rule, example, iteration)

        logger.warning("Max debugging iterations reached, returning current rule")
        return current_rule

    def _debug_with_llm_logic(self, rule: Dict[str, Any], example: str, iteration: int) -> Dict[str, Any]:
        """Debug rule using LLM logic patterns from cctools.mdc"""

        debugged_rule = rule.copy()

        # LLM Debug Pattern 1: Add stopBy: end to relational rules (per cctools.mdc tip)
        if iteration == 0:
            for key in ["inside", "has"]:
                if key in debugged_rule["rule"] and "stopBy" not in debugged_rule["rule"][key]:
                    debugged_rule["rule"][key]["stopBy"] = "end"
                    logger.debug("Added stopBy: end to relational rule", rule=key)

        # LLM Debug Pattern 2: Simplify by removing complex constraints
        elif iteration == 1:
            if "all" in debugged_rule["rule"] and len(debugged_rule["rule"]["all"]) > 2:
                # Reduce to simplest working combination
                debugged_rule["rule"]["all"] = debugged_rule["rule"]["all"][:2]
                logger.debug("Simplified composite rule")

        # LLM Debug Pattern 3: Use kind instead of pattern for ambiguous cases
        elif iteration == 2:
            if "pattern" in debugged_rule["rule"]:
                pattern = debugged_rule["rule"]["pattern"]
                # Try to infer kind from pattern
                if "def " in pattern or "function " in pattern:
                    debugged_rule["rule"] = {"kind": "function_declaration"}
                elif "class " in pattern:
                    debugged_rule["rule"] = {"kind": "class_definition"}
                logger.debug("Switched from pattern to kind")

        # LLM Debug Pattern 4: Remove relational constraints and focus on atomic
        elif iteration == 3:
            for key in ["inside", "has", "precedes", "follows"]:
                if key in debugged_rule["rule"]:
                    del debugged_rule["rule"][key]
                    logger.debug("Removed relational constraint", rule=key)

        # LLM Debug Pattern 5: Return to basic pattern matching
        else:
            debugged_rule["rule"] = {"pattern": "$EXAMPLE"}
            logger.debug("Reset to basic pattern")

        return debugged_rule

    async def _retrieve_similar_experiences(self, query: str, language: str) -> List[Experience]:
        """Retrieve similar past experiences from Qdrant"""
        if not QDRANT_AVAILABLE:
            return []

        try:
            # Search for similar queries in Qdrant
            results = await qdrant_find(f"ast-grep rule query: {query} language:{language}")

            experiences = []
            for result in results:
                try:
                    if isinstance(result, dict) and "information" in result:
                        experience_data = json.loads(result["information"])
                        experiences.append(Experience(**experience_data))
                except Exception:
                    continue

            logger.debug("Retrieved experiences from Qdrant", count=len(experiences))
            return experiences

        except Exception as e:
            logger.warning("Failed to retrieve experiences from Qdrant", error=str(e))
            return []

    async def _store_experience(self, query: str, rule: Dict[str, Any], success: bool, feedback: str, language: str) -> None:
        """Store experience in Qdrant for collective learning"""
        if not QDRANT_AVAILABLE:
            return

        try:
            experience = Experience(query=query, rule=rule, success=success, feedback=feedback, timestamp=datetime.now(), language=language)

            # Store experience in Qdrant with metadata
            metadata = {
                "type": "ast_experience",
                "success": success,
                "language": language,
                "timestamp": experience.timestamp.isoformat(),
                "rule_complexity": len(str(rule)),
            }

            await qdrant_store(json.dumps(experience.__dict__), metadata)

            # Update local cache
            cache_key = f"{query}:{language}"
            self.learning_cache[cache_key] = experience

            if success:
                self.success_patterns.append(rule)
            else:
                self.failure_patterns.append(rule)

            logger.debug("Stored experience in Qdrant", success=success, language=language, cache_size=len(self.learning_cache))

        except Exception as e:
            logger.warning("Failed to store experience in Qdrant", error=str(e))


class DynamicASTToolGenerator:
    """Generate AST tools dynamically based on examples and LLM reasoning"""

    def __init__(self, reasoning_engine: LLMAstReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self.generated_tools: Dict[str, Dict[str, Any]] = {}

    async def generate_tool_from_example(self, example_code: str, description: str, language: str = "python") -> Dict[str, Any]:
        """Generate AST tool dynamically from example using LLM reasoning"""

        # Create query from example and description
        query = f"Find code similar to: {description}"

        # Use LLM reasoning to generate rule
        rule = await self.reasoning_engine.reason_and_generate_rule(query, language)

        # Create tool specification
        tool_spec = {
            "name": f"ast_{description.replace(' ', '_').lower()}",
            "description": description,
            "rule": rule,
            "example": example_code,
            "language": language,
            "generated_at": datetime.now().isoformat(),
        }

        # Store generated tool
        tool_name = cast(str, tool_spec["name"])
        self.generated_tools[tool_name] = tool_spec

        logger.info("Generated dynamic AST tool", tool_name=tool_name, rule_complexity=len(str(rule)))

        return tool_spec

    async def execute_generated_tool(self, tool_name: str, search_path: str = ".") -> str:
        """Execute generated tool using direct CLI functions"""

        if tool_name not in self.generated_tools:
            raise ValueError(f"Tool {tool_name} not found")

        tool = self.generated_tools[tool_name]
        rule = tool["rule"]

        # Convert rule to YAML for CLI execution
        rule_yaml = yaml.dump(rule["rule"])

        # Execute using direct CLI function
        result = find_code_by_rule(rule_yaml, search_path, "json")

        logger.debug("Executed generated tool", tool_name=tool_name, result_length=len(result))

        return result


# Global instance for easy access
llm_ast_engine = LLMAstReasoningEngine()
ast_tool_generator = DynamicASTToolGenerator(llm_ast_engine)


async def generate_ast_rule(query: str, language: str = "python") -> Dict[str, Any]:
    """Convenience function for rule generation"""
    return await llm_ast_engine.reason_and_generate_rule(query, language)


async def generate_ast_tool(example_code: str, description: str, language: str = "python") -> Dict[str, Any]:
    """Convenience function for tool generation"""
    return await ast_tool_generator.generate_tool_from_example(example_code, description, language)


def get_cli_status() -> Dict:
    """Check CLI and Qdrant availability"""
    return {
        "cli_available": CLI_AVAILABLE and is_ast_grep_available(),
        "qdrant_available": QDRANT_AVAILABLE,
        "generated_tools_count": len(ast_tool_generator.generated_tools),
        "learning_cache_size": len(llm_ast_engine.learning_cache),
        "success_patterns_count": len(llm_ast_engine.success_patterns),
        "failure_patterns_count": len(llm_ast_engine.failure_patterns),
    }
