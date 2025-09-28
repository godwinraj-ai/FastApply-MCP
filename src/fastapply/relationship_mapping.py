"""
Relationship Mapping Module

Advanced relationship analysis for understanding code architecture,
dependencies, and structural patterns across codebases.
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

# Handle optional dependencies gracefully
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class RelationshipType(Enum):
    """Types of relationships between code components."""
    INHERITS_FROM = "inherits_from"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    CALLS = "calls"
    IMPORTS = "imports"
    INSTANTIATES = "instantiates"
    REFERENCES = "references"
    CONTAINS = "contains"
    OVERRIDES = "overrides"
    EXTENDS = "extends"
    COMPOSES = "composes"
    AGGREGATES = "aggregates"
    ASSOCIATES = "associates"


class DependencyType(Enum):
    """Types of dependencies between modules."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CYCLICAL = "cyclical"
    TRANSITIVE = "transitive"
    WEAK = "weak"
    STRONG = "strong"


class CouplingType(Enum):
    """Types of coupling between modules."""
    CONTENT = "content"  # One module depends on internal implementation of another
    COMMON = "common"  # Modules share global data
    EXTERNAL = "external"  # Module depends on interface of another
    CONTROL = "control"  # One module controls flow of another
    STAMP = "stamp"  # Modules share data structure
    DATA = "data"  # Modules communicate via parameters
    TEMPORAL = "temporal"  # Timing dependencies


@dataclass
class CodeComponent:
    """Represents a code component (class, function, module, etc.)."""
    name: str
    type: str  # "class", "function", "module", "variable"
    file_path: str
    line_number: int
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Represents a relationship between two code components."""
    source: CodeComponent
    target: CodeComponent
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dependency:
    """Represents a dependency between modules."""
    source_module: str
    target_module: str
    dependency_type: DependencyType
    path: List[str] = field(default_factory=list)  # For indirect dependencies
    strength: float = 1.0


@dataclass
class CouplingAnalysis:
    """Analysis of coupling between modules."""
    source_module: str
    target_module: str
    coupling_types: List[CouplingType]
    coupling_score: float  # 0.0 to 1.0, higher = more coupled
    shared_elements: List[str]
    impact_score: float  # Impact of changes on dependent modules


@dataclass
class CohesionAnalysis:
    """Analysis of cohesion within a module."""
    module_name: str
    cohesion_score: float  # 0.0 to 1.0, higher = more cohesive
    responsibility_count: int
    related_elements: List[str]
    cohesion_type: str  # "functional", "sequential", "communicational", "procedural"


@dataclass
class CircularDependency:
    """Represents a circular dependency between modules."""
    cycle_path: List[str]
    dependency_types: List[DependencyType]
    impact_score: float
    resolution_suggestions: List[str]


@dataclass
class RelationshipMap:
    """Complete relationship map of a codebase."""
    components: Dict[str, CodeComponent] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    coupling_analysis: List[CouplingAnalysis] = field(default_factory=list)
    cohesion_analysis: List[CohesionAnalysis] = field(default_factory=list)
    circular_dependencies: List[CircularDependency] = field(default_factory=list)
    graph: Any = field(default_factory=lambda: None if not NETWORKX_AVAILABLE else nx.DiGraph())


@dataclass
class DependencyAnalysis:
    """Analysis of dependencies between modules."""
    direct_dependencies: List[Dependency]
    indirect_dependencies: List[Dependency]
    circular_dependencies: List[CircularDependency]
    transitive_dependencies: List[Dependency]
    dependency_score: float  # Overall dependency complexity score
    dependencies: List[Dependency] = field(default_factory=list)
    dependency_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    critical_paths: List[List[str]] = field(default_factory=list)


@dataclass
class ControlFlowNode:
    """Represents a node in control flow graph."""
    id: str
    type: str  # "entry", "exit", "condition", "loop", "statement", "function_call"
    label: str
    line_number: int
    condition: Optional[str] = None


@dataclass
class ControlFlowEdge:
    """Represents an edge in control flow graph."""
    source: str
    target: str
    type: str  # "true_branch", "false_branch", "unconditional", "exception"
    condition: Optional[str] = None


@dataclass
class ControlFlowGraph:
    """Represents control flow graph of code."""
    nodes: List[ControlFlowNode] = field(default_factory=list)
    edges: List[ControlFlowEdge] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    cyclomatic_complexity: int = 1
    entry_points: List[str] = field(default_factory=list)
    exit_points: List[str] = field(default_factory=list)

    def add_node(self, node_id: str, **kwargs):
        """Add a node to the CFG."""
        self.nodes.append(ControlFlowNode(
            id=node_id,
            type=kwargs.get('type', 'statement'),
            label=kwargs.get('label', node_id),
            line_number=kwargs.get('line_number', 0),
            condition=kwargs.get('condition')
        ))


@dataclass
class DataFlowNode:
    """Represents a variable or data element."""
    name: str
    type: str
    scope: str
    definitions: List[int] = field(default_factory=list)  # Line numbers where defined
    uses: List[int] = field(default_factory=list)  # Line numbers where used
    mutations: List[int] = field(default_factory=list)  # Line numbers where modified


@dataclass
class DataFlowEdge:
    """Represents data flow dependency."""
    source: str  # Variable name
    target: str  # Variable name
    type: str  # "definition", "use", "kill"
    line_number: int


@dataclass
class DataFlowGraph:
    """Represents data flow graph of code."""
    variables: List[DataFlowNode] = field(default_factory=list)
    data_dependencies: List[DataFlowEdge] = field(default_factory=list)
    flow_paths: List[List[str]] = field(default_factory=list)
    critical_variables: List[str] = field(default_factory=list)
    data_flow_complexity: float = 0.0

    def add_variable(self, variable: str):
        """Add a variable to the DFG."""
        self.variables.append(DataFlowNode(
            name=variable,
            type='unknown',
            scope='global'
        ))
        self.critical_variables.append(variable)


@dataclass
class ArchitecturalPatterns:
    """Analysis of architectural patterns detected."""
    patterns: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    pattern_locations: Dict[str, List[str]] = field(default_factory=dict)
    architectural_style: str = ""
    layering_violations: List[str] = field(default_factory=list)
    pattern_consistency: float = 0.0
    detected_patterns: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


class CodeStructure:
    """Represents the structure of analyzed code."""
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.files: Dict[str, str] = {}
        self.modules: Dict[str, List[CodeComponent]] = defaultdict(list)
        self.imports: Dict[str, Set[str]] = defaultdict(set)


class RelationshipMapper:
    """Main class for mapping code relationships and architecture."""

    def __init__(self) -> None:
        self.component_cache: Dict[str, CodeComponent] = {}
        self.relationship_cache: Dict[str, List[Relationship]] = defaultdict(list)

    def understand_relationships(self,
                               code: str,
                               context: str = "",
                               project_path: Optional[str] = None) -> RelationshipMap:
        """
        Analyze and map relationships in code.

        Args:
            code: Source code to analyze
            context: Analysis context or file path
            project_path: Root path of the project

        Returns:
            Complete relationship map
        """
        # Parse code structure
        code_structure = self._parse_code_structure(code, context, project_path)

        # Extract components
        components = self._extract_components(code_structure)

        # Build relationship graph
        relationship_map = self._build_relationship_graph(components, code_structure)

        # Analyze dependencies
        dependencies = self._analyze_dependencies(relationship_map, code_structure)

        # Analyze coupling and cohesion
        coupling_analysis = self._analyze_coupling(relationship_map, dependencies)
        cohesion_analysis = self._analyze_cohesion(relationship_map, components)

        # Detect circular dependencies
        circular_dependencies = self._detect_circular_dependencies(dependencies)

        # Build final relationship map
        final_map = RelationshipMap(
            components={c.name: c for c in components},
            relationships=relationship_map.relationships,
            dependencies=dependencies,
            coupling_analysis=coupling_analysis,
            cohesion_analysis=cohesion_analysis,
            circular_dependencies=circular_dependencies,
            graph=relationship_map.graph
        )

        return final_map

    def analyze_dependencies(self,
                            code_structure: CodeStructure) -> DependencyAnalysis:
        """
        Analyze dependencies between modules.

        Args:
            code_structure: Parsed code structure

        Returns:
            Complete dependency analysis
        """
        dependencies = []

        # Direct dependencies from imports
        for module, imports in code_structure.imports.items():
            for imported_module in imports:
                dependency = Dependency(
                    source_module=module,
                    target_module=imported_module,
                    dependency_type=DependencyType.DIRECT
                )
                dependencies.append(dependency)

        # Find indirect dependencies
        indirect_deps = self._find_indirect_dependencies(dependencies)
        dependencies.extend(indirect_deps)

        # Find transitive dependencies
        transitive_deps = self._find_transitive_dependencies(dependencies)
        dependencies.extend(transitive_deps)

        return DependencyAnalysis(
            direct_dependencies=dependencies,
            indirect_dependencies=indirect_deps,
            circular_dependencies=[],
            transitive_dependencies=transitive_deps,
            dependency_score=0.0,
            dependencies=dependencies,
            dependency_matrix=self._build_dependency_matrix(dependencies),
            critical_paths=self._identify_critical_paths(dependencies)
        )

    def map_control_flow(self, code: str, language: str = "python") -> ControlFlowGraph:
        """
        Map control flow within code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Control flow graph
        """
        cfg = ControlFlowGraph()

        if language == "python":
            try:
                tree = ast.parse(code)
                self._build_cfg_from_ast(tree, cfg)
            except Exception:
                # Fallback to simple control flow analysis
                self._build_simple_cfg(code, cfg)

        return cfg

    def analyze_data_flow(self, code: str, language: str = "python") -> DataFlowGraph:
        """
        Analyze data flow within code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Data flow graph
        """
        dfg = DataFlowGraph()

        if language == "python":
            try:
                tree = ast.parse(code)
                self._build_dfg_from_ast(tree, dfg)
            except Exception:
                # Fallback to simple data flow analysis
                self._build_simple_dfg(code, dfg)

        return dfg

    def detect_architectural_patterns(self,
                                    relationship_map: RelationshipMap) -> ArchitecturalPatterns:
        """
        Detect architectural patterns from relationship analysis.

        Args:
            relationship_map: Complete relationship map

        Returns:
            Detected architectural patterns
        """
        patterns = ArchitecturalPatterns()

        # Layered architecture detection
        if self._detect_layered_architecture(relationship_map):
            patterns.detected_patterns.append("layered")

        # MVC architecture detection
        if self._detect_mvc_architecture(relationship_map):
            patterns.detected_patterns.append("mvc")

        # Microservices detection
        if self._detect_microservices_architecture(relationship_map):
            patterns.detected_patterns.append("microservices")

        # Repository pattern detection
        if self._detect_repository_pattern(relationship_map):
            patterns.detected_patterns.append("repository")

        # Anti-patterns detection
        anti_patterns = self._detect_anti_patterns(relationship_map)
        patterns.anti_patterns.extend(anti_patterns)

        # Architectural insights
        patterns.insights = self._generate_architectural_insights(relationship_map)

        return patterns

    # Private helper methods

    @lru_cache(maxsize=256)
    def _parse_code_structure(self, code: str, context: str, project_path: Optional[str]) -> CodeStructure:
        """Parse code and extract structural information."""
        structure = CodeStructure(project_path or "")

        # Store code
        if context:
            structure.files[context] = code

        # Extract imports
        if context:
            structure.imports[context] = self._extract_imports(code)

        # Extract components by parsing
        try:
            tree = ast.parse(code)
            module_name = self._get_module_name(context)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    component = CodeComponent(
                        name=node.name,
                        type="class",
                        file_path=context,
                        line_number=node.lineno,
                        language="python",
                        metadata={"bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]}
                    )
                    structure.modules[module_name].append(component)

                elif isinstance(node, ast.FunctionDef):
                    component = CodeComponent(
                        name=node.name,
                        type="function",
                        file_path=context,
                        line_number=node.lineno,
                        language="python",
                        metadata={"args": [arg.arg for arg in node.args.args]}
                    )
                    structure.modules[module_name].append(component)

        except Exception:
            pass

        return structure

    def _extract_components(self, code_structure: CodeStructure) -> List[CodeComponent]:
        """Extract all code components from structure."""
        components = []

        for module, module_components in code_structure.modules.items():
            components.extend(module_components)

        return components

    def _build_relationship_graph(self,
                                  components: List[CodeComponent],
                                  code_structure: CodeStructure) -> RelationshipMap:
        """Build graph of relationships between components."""
        relationship_map = RelationshipMap()
        relationship_map.components = {c.name: c for c in components}

        # Build networkx graph
        graph = nx.DiGraph()
        for component in components:
            graph.add_node(component.name, component=component)

        relationships: List[Relationship] = []

        # Analyze inheritance relationships
        inheritance_rels = self._analyze_inheritance_relationships(components)
        relationships.extend(inheritance_rels)

        # Analyze call relationships
        call_rels = self._analyze_call_relationships(components, code_structure)
        relationships.extend(call_rels)

        # Analyze import relationships
        import_rels = self._analyze_import_relationships(components, code_structure)
        relationships.extend(import_rels)

        # Add relationships to graph
        for rel in relationships:
            graph.add_edge(rel.source.name, rel.target.name,
                      relationship_type=rel.relationship_type.value,
                      strength=rel.strength)

        relationship_map.relationships = relationships
        relationship_map.graph = graph

        return relationship_map

    def _analyze_inheritance_relationships(self,
                                          components: List[CodeComponent]) -> List[Relationship]:
        """Analyze inheritance relationships between components."""
        relationships: List[Relationship] = []

        for component in components:
            if component.type == "class":
                bases = component.metadata.get("bases", [])
                for base in bases:
                    # Find base component
                    base_component = next((c for c in components if c.name == base), None)
                    if base_component:
                        relationship = Relationship(
                            source=component,
                            target=base_component,
                            relationship_type=RelationshipType.INHERITS_FROM,
                            strength=0.9
                        )
                        relationships.append(relationship)

        return relationships

    def _analyze_call_relationships(self,
                                  components: List[CodeComponent],
                                  code_structure: CodeStructure) -> List[Relationship]:
        """Analyze function/method call relationships."""
        relationships: List[Relationship] = []

        for component in components:
            if component.type == "function":
                # This would require more sophisticated analysis of function bodies
                # For now, we'll create placeholder relationships
                pass

        return relationships

    def _analyze_import_relationships(self,
                                     components: List[CodeComponent],
                                     code_structure: CodeStructure) -> List[Relationship]:
        """Analyze import-based relationships."""
        relationships: List[Relationship] = []

        for module, imports in code_structure.imports.items():
            for imported_module in imports:
                # Find components in each module
                source_components = [c for c in components if module in c.file_path]
                target_components = [c for c in components if imported_module in c.file_path]

                for source in source_components:
                    for target in target_components:
                        relationship = Relationship(
                            source=source,
                            target=target,
                            relationship_type=RelationshipType.IMPORTS,
                            strength=0.7
                        )
                        relationships.append(relationship)

        return relationships

    def _analyze_dependencies(self,
                             relationship_map: RelationshipMap,
                             code_structure: CodeStructure) -> List[Dependency]:
        """Analyze dependencies between modules."""
        dependencies = []

        # Extract modules from relationship map
        modules = set()
        for component in relationship_map.components.values():
            module = self._get_module_name(component.file_path)
            if module:
                modules.add(module)

        # Build dependency matrix
        for relationship in relationship_map.relationships:
            if relationship.relationship_type in [RelationshipType.IMPORTS, RelationshipType.DEPENDS_ON]:
                source_module = self._get_module_name(relationship.source.file_path)
                target_module = self._get_module_name(relationship.target.file_path)

                if source_module and target_module and source_module != target_module:
                    dependency = Dependency(
                        source_module=source_module,
                        target_module=target_module,
                        dependency_type=DependencyType.DIRECT,
                        strength=relationship.strength
                    )
                    dependencies.append(dependency)

        return dependencies

    def _analyze_coupling(self,
                         relationship_map: RelationshipMap,
                         dependencies: List[Dependency]) -> List[CouplingAnalysis]:
        """Analyze coupling between modules."""
        coupling_analysis = []

        # Group dependencies by module pair
        module_pairs = defaultdict(list)
        for dep in dependencies:
            key = (dep.source_module, dep.target_module)
            module_pairs[key].append(dep)

        # Analyze coupling for each pair
        for (source, target), deps in module_pairs.items():
            coupling_types = self._determine_coupling_types(deps)
            coupling_score = self._calculate_coupling_score(deps)
            shared_elements = self._find_shared_elements(relationship_map, source, target)
            impact_score = self._calculate_impact_score(deps)

            analysis = CouplingAnalysis(
                source_module=source,
                target_module=target,
                coupling_types=coupling_types,
                coupling_score=coupling_score,
                shared_elements=shared_elements,
                impact_score=impact_score
            )
            coupling_analysis.append(analysis)

        return coupling_analysis

    def _analyze_cohesion(self,
                         relationship_map: RelationshipMap,
                         components: List[CodeComponent]) -> List[CohesionAnalysis]:
        """Analyze cohesion within modules."""
        cohesion_analysis = []

        # Group components by module
        module_components = defaultdict(list)
        for component in components:
            module = self._get_module_name(component.file_path)
            if module:
                module_components[module].append(component)

        # Analyze cohesion for each module
        for module, module_comps in module_components.items():
            cohesion_score = self._calculate_cohesion_score(module_comps, relationship_map)
            responsibility_count = len(module_comps)
            related_elements = self._find_related_elements(module_comps, relationship_map)
            cohesion_type = self._determine_cohesion_type(module_comps, relationship_map)

            analysis = CohesionAnalysis(
                module_name=module,
                cohesion_score=cohesion_score,
                responsibility_count=responsibility_count,
                related_elements=related_elements,
                cohesion_type=cohesion_type
            )
            cohesion_analysis.append(analysis)

        return cohesion_analysis

    def _detect_circular_dependencies(self,
                                     dependencies: List[Dependency]) -> List[CircularDependency]:
        """Detect circular dependencies between modules."""
        circular_deps = []

        # Build dependency graph
        graph = nx.DiGraph()
        for dep in dependencies:
            graph.add_edge(dep.source_module, dep.target_module)

        # Find cycles
        try:
            cycles = list(nx.simple_cycles(graph))
        except Exception:
            cycles = []

        for cycle in cycles:
            impact_score = self._calculate_cycle_impact(cycle, dependencies)
            resolution_suggestions = self._generate_cycle_resolution_suggestions(cycle)

            circular_dep = CircularDependency(
                cycle_path=cycle,
                dependency_types=[DependencyType.CYCLICAL],
                impact_score=impact_score,
                resolution_suggestions=resolution_suggestions
            )
            circular_deps.append(circular_dep)

        return circular_deps

    def _find_indirect_dependencies(self,
                                   dependencies: List[Dependency]) -> List[Dependency]:
        """Find indirect dependencies through transitive analysis."""
        indirect_deps = []

        # Build graph for path analysis
        graph = nx.DiGraph()
        for dep in dependencies:
            graph.add_edge(dep.source_module, dep.target_module)

        # Find all pairs shortest paths
        try:
            all_pairs = dict(nx.all_pairs_shortest_path(graph))
        except Exception:
            all_pairs = {}

        for source in all_pairs:
            for target in all_pairs[source]:
                if source != target and len(all_pairs[source][target]) > 2:
                    # This is an indirect dependency
                    path = all_pairs[source][target]
                    indirect_dep = Dependency(
                        source_module=source,
                        target_module=target,
                        dependency_type=DependencyType.INDIRECT,
                        path=path,
                        strength=0.5  # Weaker than direct
                    )
                    indirect_deps.append(indirect_dep)

        return indirect_deps

    def _find_transitive_dependencies(self,
                                     dependencies: List[Dependency]) -> List[Dependency]:
        """Find transitive dependencies."""
        transitive_deps = []

        # Build dependency graph
        graph = nx.DiGraph()
        for dep in dependencies:
            graph.add_edge(dep.source_module, dep.target_module)

        # Find transitive closure
        transitive_closure = nx.transitive_closure(graph)

        # Add transitive dependencies not in original graph
        for edge in transitive_closure.edges():
            if not graph.has_edge(edge[0], edge[1]):
                transitive_dep = Dependency(
                    source_module=edge[0],
                    target_module=edge[1],
                    dependency_type=DependencyType.TRANSITIVE,
                    strength=0.3  # Weakest dependency type
                )
                transitive_deps.append(transitive_dep)

        return transitive_deps

    def _determine_coupling_types(self, dependencies: List[Dependency]) -> List[CouplingType]:
        """Determine types of coupling between modules."""
        coupling_types = []

        # Based on dependency types and patterns
        for dep in dependencies:
            if dep.dependency_type == DependencyType.DIRECT:
                coupling_types.append(CouplingType.EXTERNAL)
            elif dep.dependency_type == DependencyType.CYCLICAL:
                coupling_types.append(CouplingType.COMMON)

        return list(set(coupling_types))

    def _calculate_coupling_score(self, dependencies: List[Dependency]) -> float:
        """Calculate coupling score between modules."""
        if not dependencies:
            return 0.0

        # Base score on number and strength of dependencies
        total_strength = sum(dep.strength for dep in dependencies)
        normalized_score = min(total_strength / 2.0, 1.0)  # Normalize to 0-1

        return normalized_score

    def _find_shared_elements(self,
                             relationship_map: RelationshipMap,
                             source_module: str,
                             target_module: str) -> List[str]:
        """Find shared elements between modules."""
        shared_elements = []

        # Find components in each module
        source_components = [c for c in relationship_map.components.values()
                           if self._get_module_name(c.file_path) == source_module]
        target_components = [c for c in relationship_map.components.values()
                           if self._get_module_name(c.file_path) == target_module]

        # Find shared relationships
        for rel in relationship_map.relationships:
            if (rel.source in source_components and rel.target in target_components) or \
               (rel.source in target_components and rel.target in source_components):
                shared_elements.append(f"{rel.source.name} -> {rel.target.name}")

        return shared_elements

    def _calculate_impact_score(self, dependencies: List[Dependency]) -> float:
        """Calculate impact score of dependencies."""
        if not dependencies:
            return 0.0

        # Higher impact for stronger dependencies
        total_strength = sum(dep.strength for dep in dependencies)
        return min(total_strength, 1.0)

    def _calculate_cohesion_score(self,
                                 module_components: List[CodeComponent],
                                 relationship_map: RelationshipMap) -> float:
        """Calculate cohesion score for a module."""
        if len(module_components) <= 1:
            return 1.0

        # Count internal relationships
        internal_relationships = 0
        total_relationships = 0

        for rel in relationship_map.relationships:
            if rel.source in module_components:
                total_relationships += 1
                if rel.target in module_components:
                    internal_relationships += 1

        # Cohesion = internal relationships / total relationships
        if total_relationships == 0:
            return 0.5  # Neutral score

        return internal_relationships / total_relationships

    def _find_related_elements(self,
                             module_components: List[CodeComponent],
                             relationship_map: RelationshipMap) -> List[str]:
        """Find related elements within a module."""
        related_elements = []

        for component in module_components:
            # Find relationships to other components in the same module
            for rel in relationship_map.relationships:
                if rel.source == component and rel.target in module_components:
                    related_elements.append(f"{component.name} -> {rel.target.name}")

        return related_elements

    def _determine_cohesion_type(self,
                               module_components: List[CodeComponent],
                               relationship_map: RelationshipMap) -> str:
        """Determine the type of cohesion."""
        # Simplified cohesion type detection
        function_count = sum(1 for c in module_components if c.type == "function")
        class_count = sum(1 for c in module_components if c.type == "class")

        if function_count > class_count:
            return "functional"
        elif class_count > 0:
            return "sequential"
        else:
            return "procedural"

    def _calculate_cycle_impact(self, cycle: List[str], dependencies: List[Dependency]) -> float:
        """Calculate impact score of a circular dependency."""
        # Longer cycles generally have higher impact
        base_impact = len(cycle) * 0.2

        # Consider dependency strengths
        cycle_deps = [d for d in dependencies
                      if d.source_module in cycle and d.target_module in cycle]
        strength_impact = sum(d.strength for d in cycle_deps) * 0.1

        return min(base_impact + strength_impact, 1.0)

    def _generate_cycle_resolution_suggestions(self, cycle: List[str]) -> List[str]:
        """Generate suggestions for resolving circular dependencies."""
        suggestions = []

        if len(cycle) == 2:
            suggestions.append("Extract shared interface to break direct dependency")
        elif len(cycle) == 3:
            suggestions.append("Apply dependency inversion principle")
        else:
            suggestions.append("Consider refactoring into separate layers")

        suggestions.append("Use dependency injection to reduce coupling")
        suggestions.append("Introduce abstraction layer between circular dependencies")

        return suggestions

    def _detect_layered_architecture(self, relationship_map: RelationshipMap) -> bool:
        """Detect if the codebase follows layered architecture."""
        # Simplified detection based on import patterns
        layers = self._identify_layers(relationship_map)
        return len(layers) >= 3

    def _detect_mvc_architecture(self, relationship_map: RelationshipMap) -> bool:
        """Detect if the codebase follows MVC architecture."""
        # Look for model, view, controller patterns
        has_models = any("model" in c.name.lower() for c in relationship_map.components.values())
        has_views = any("view" in c.name.lower() for c in relationship_map.components.values())
        has_controllers = any("controller" in c.name.lower() or "handler" in c.name.lower()
                             for c in relationship_map.components.values())

        return has_models and has_views and has_controllers

    def _detect_microservices_architecture(self, relationship_map: RelationshipMap) -> bool:
        """Detect if the codebase follows microservices architecture."""
        # Look for service boundaries and loose coupling
        modules = set()
        for component in relationship_map.components.values():
            module = self._get_module_name(component.file_path)
            if module:
                modules.add(module)

        # Check for low coupling between modules
        coupling_count = len([r for r in relationship_map.relationships
                             if r.relationship_type == RelationshipType.IMPORTS])
        coupling_ratio = coupling_count / max(len(modules), 1)

        return coupling_ratio < 2.0 and len(modules) > 3

    def _detect_repository_pattern(self, relationship_map: RelationshipMap) -> bool:
        """Detect if the codebase uses repository pattern."""
        # Look for repository classes and data access patterns
        repositories = [c for c in relationship_map.components.values()
                       if "repository" in c.name.lower()]
        return len(repositories) > 0

    def _detect_anti_patterns(self, relationship_map: RelationshipMap) -> List[str]:
        """Detect architectural anti-patterns."""
        anti_patterns = []

        # God object detection
        large_components = [c for c in relationship_map.components.values()
                          if c.metadata.get("size", 0) > 1000]
        if large_components:
            anti_patterns.append("god_object")

        # Spaghetti code detection
        circular_deps = relationship_map.circular_dependencies
        if len(circular_deps) > len(relationship_map.components) * 0.1:
            anti_patterns.append("spaghetti_code")

        return anti_patterns

    def _generate_architectural_insights(self,
                                       relationship_map: RelationshipMap) -> List[str]:
        """Generate architectural insights from relationship analysis."""
        insights = []

        # Component count insight
        total_components = len(relationship_map.components)
        if total_components > 100:
            insights.append("Large codebase with many components - consider modularization")

        # Relationship density insight
        total_relationships = len(relationship_map.relationships)
        relationship_density = total_relationships / max(total_components, 1)
        if relationship_density > 5:
            insights.append("High relationship density - may indicate tight coupling")

        # Circular dependency insight
        if relationship_map.circular_dependencies:
            insights.append("Circular dependencies detected - refactoring recommended")

        return insights

    def _extract_imports(self, code: str) -> Set[str]:
        """Extract imports from code."""
        imports = set()

        # Python import patterns
        import_patterns = [
            r'import\s+(\w+(?:\.\w+)*)',
            r'from\s+(\w+(?:\.\w+)*)\s+import',
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            imports.update(matches)

        return imports

    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        if not file_path:
            return ""

        # Remove file extension
        module = file_path.replace(".py", "").replace(".js", "").replace(".ts", "")

        # Extract just the module name without path
        module = module.split("/")[-1].split("\\")[-1]

        return module

    def _identify_layers(self, relationship_map: RelationshipMap) -> List[Set[str]]:
        """Identify architectural layers."""
        # Simplified layer identification based on naming conventions
        layers = defaultdict(set)

        for component in relationship_map.components.values():
            name_lower = component.name.lower()

            if any(keyword in name_lower for keyword in ["ui", "view", "present"]):
                layers["presentation"].add(component.name)
            elif any(keyword in name_lower for keyword in ["controller", "handler", "service"]):
                layers["business"].add(component.name)
            elif any(keyword in name_lower for keyword in ["model", "entity", "data"]):
                layers["data"].add(component.name)
            else:
                layers["common"].add(component.name)

        return [layer for layer in layers.values() if layer]

    def _build_cfg_from_ast(self, tree: ast.AST, cfg: 'ControlFlowGraph'):
        """Build control flow graph from AST."""
        # Simplified CFG building
        # This would require more sophisticated implementation
        pass

    def _build_simple_cfg(self, code: str, cfg: 'ControlFlowGraph'):
        """Build simple control flow graph."""
        # Very simplified CFG building
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if "if " in line:
                cfg.add_node(f"line_{i}")
            elif "for " in line:
                cfg.add_node(f"line_{i}")
            elif "while " in line:
                cfg.add_node(f"line_{i}")

    def _build_dfg_from_ast(self, tree: ast.AST, dfg: 'DataFlowGraph'):
        """Build data flow graph from AST."""
        # Simplified DFG building
        # This would require more sophisticated implementation
        pass

    def _build_simple_dfg(self, code: str, dfg: 'DataFlowGraph'):
        """Build simple data flow graph."""
        # Very simplified DFG building
        # Extract variable assignments
        assignments = re.findall(r'(\w+)\s*=', code)

        for var in assignments:
            dfg.add_variable(var)

    def _build_dependency_matrix(self, dependencies: List[Dependency]) -> Dict[str, Dict[str, float]]:
        """Build dependency matrix."""
        matrix: Dict[str, Dict[str, float]] = defaultdict(dict)

        for dep in dependencies:
            matrix[dep.source_module][dep.target_module] = dep.strength

        return dict(matrix)

    def _identify_critical_paths(self, dependencies: List[Dependency]) -> List[List[str]]:
        """Identify critical dependency paths."""
        # Simplified critical path identification
        critical_paths = []

        # Find dependencies with high impact
        high_impact_deps = [d for d in dependencies if d.strength > 0.8]

        for dep in high_impact_deps:
            critical_paths.append([dep.source_module, dep.target_module])

        return critical_paths




