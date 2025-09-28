"""
Safe Refactoring Tools for FastApply

Implements safe refactoring operations including symbol renaming and code extraction/movement
with comprehensive safety validation and rollback capabilities.

Phase 5 Implementation - Safe Refactoring Execution Tools
"""

import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from .enhanced_search import EnhancedSearchInfrastructure, EnhancedSearchResult, SearchContext, SearchStrategy
from .symbol_operations import ReferenceInfo, ReferenceType, SymbolInfo, SymbolScope, SymbolType

logger = structlog.get_logger(__name__)


class RefactoringResult(Enum):
    """Result of a refactoring operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RenameOperation:
    """Represents a symbol rename operation."""

    old_name: str
    new_name: str
    symbol_type: str
    file_path: str
    scope: Optional[str] = None
    references: List[ReferenceInfo] = field(default_factory=list)


@dataclass
class ExtractionOperation:
    """Represents a code extraction operation."""

    source_range: Tuple[int, int]  # (start_line, end_line)
    target_name: str
    target_file: str
    extraction_type: str  # "function", "method", "class"
    dependencies: List[str] = field(default_factory=list)


@dataclass
class MoveOperation:
    """Represents a symbol movement operation."""

    symbol_name: str
    source_file: str
    target_file: str
    symbol_type: str
    scope: Optional[str] = None


@dataclass
class RollbackPlan:
    """Rollback plan for refactoring operations."""

    original_files: Dict[str, str]  # file_path -> original_content
    backup_files: Dict[str, str]  # file_path -> backup_path
    operation_log: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImpactAnalysis:
    """Analysis of refactoring impact."""

    affected_files: Set[str]
    affected_symbols: Set[str]
    external_dependencies: Set[str]
    test_impact: bool
    breaking_changes: bool
    risk_score: float  # 0.0 (low) to 1.0 (high)


class SafeSymbolRenaming:
    """
    Safe symbol renaming with comprehensive reference updating and rollback capability.

    Implements Story 5.1: Symbol Renaming from the implementation specification.
    """

    def __init__(self, search_engine: Optional[EnhancedSearchInfrastructure] = None):
        """Initialize the safe symbol renaming system."""
        self.search_engine = search_engine or EnhancedSearchInfrastructure()
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.operation_lock = threading.Lock()

    def rename_symbol_safely(
        self, old_name: str, new_name: str, symbol_type: str = "function", scope: Optional[str] = None, project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely rename a symbol with automatic reference updating.

        Args:
            old_name: Current symbol name
            new_name: New symbol name
            symbol_type: Type of symbol (function, class, variable, etc.)
            scope: Optional scope/context for the symbol
            project_path: Path to the project root

        Returns:
            Dictionary with rename operation results
        """
        try:
            logger.info(f"Starting safe rename: {old_name} -> {new_name}")

            # Validate rename safety
            impact_analysis = self.analyze_rename_impact(old_name, new_name, symbol_type, scope, project_path)

            if not self._validate_rename_safety(impact_analysis):
                return {
                    "status": RefactoringResult.FAILED.value,
                    "error": "Rename operation deemed unsafe",
                    "impact_analysis": impact_analysis,
                }

            # Find symbol definition and references
            symbol_info = self._find_symbol_definition(old_name, symbol_type, scope, project_path)
            if not symbol_info:
                return {"status": RefactoringResult.FAILED.value, "error": f"Symbol '{old_name}' not found"}

            references = self._find_all_references(symbol_info, project_path)

            # Create rollback plan
            rollback_plan = self._create_rename_rollback_plan(symbol_info, references)

            # Execute rename operation
            with self.operation_lock:
                result = self._execute_symbol_rename(symbol_info, new_name, references, rollback_plan)

                if result["status"] == RefactoringResult.SUCCESS.value:
                    # Store rollback plan for potential undo
                    operation_id = f"rename_{old_name}_{int(time.time())}"
                    self.rollback_plans[operation_id] = rollback_plan
                    result["operation_id"] = operation_id

                return result

        except Exception as e:
            logger.error(f"Error during safe rename: {e}")
            return {"status": RefactoringResult.FAILED.value, "error": str(e)}

    def analyze_rename_impact(
        self, old_name: str, new_name: str, symbol_type: str, scope: Optional[str] = None, project_path: Optional[str] = None
    ) -> ImpactAnalysis:
        """
        Analyze the impact of a rename operation.

        Args:
            old_name: Current symbol name
            new_name: New symbol name
            symbol_type: Type of symbol
            scope: Optional scope
            project_path: Project root path

        Returns:
            ImpactAnalysis object with detailed impact assessment
        """
        try:
            # Find symbol definition
            symbol_info = self._find_symbol_definition(old_name, symbol_type, scope, project_path)
            if not symbol_info:
                return ImpactAnalysis(
                    affected_files=set(),
                    affected_symbols=set(),
                    external_dependencies=set(),
                    test_impact=False,
                    breaking_changes=True,
                    risk_score=1.0,
                )

            # Find all references
            references = self._find_all_references(symbol_info, project_path)

            # Analyze impact
            affected_files = {ref.file_path for ref in references}
            affected_files.add(symbol_info.file_path)

            # Check for name conflicts
            name_conflicts = self._check_name_conflicts(new_name, symbol_type, project_path)

            # Check external API usage
            external_apis = self._identify_external_apis(references)

            # Check test impact
            test_files = [f for f in affected_files if any(pattern in f.lower() for pattern in ["test_", "spec_"])]

            # Calculate risk score
            risk_score = self._calculate_rename_risk_score(len(references), len(external_apis), len(test_files), bool(name_conflicts))

            return ImpactAnalysis(
                affected_files=affected_files,
                affected_symbols={ref.symbol_name for ref in references},
                external_dependencies=external_apis,
                test_impact=bool(test_files),
                breaking_changes=bool(external_apis or name_conflicts),
                risk_score=risk_score,
            )

        except Exception as e:
            logger.error(f"Error analyzing rename impact: {e}")
            return ImpactAnalysis(
                affected_files=set(),
                affected_symbols=set(),
                external_dependencies=set(),
                test_impact=True,
                breaking_changes=True,
                risk_score=1.0,
            )

    def generate_rollback_plan(self, symbol_info: SymbolInfo, new_name: str, references: List[ReferenceInfo]) -> RollbackPlan:
        """
        Generate a rollback plan for a rename operation.

        Args:
            symbol_info: Information about the symbol being renamed
            new_name: The new name for the symbol
            references: List of all references to the symbol

        Returns:
            RollbackPlan with complete rollback information
        """
        original_files = {}
        backup_files = {}
        operation_log = []

        # Backup original files
        all_files = {symbol_info.file_path}
        all_files.update(ref.file_path for ref in references)

        for file_path in all_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    original_files[file_path] = f.read()

                # Create backup file
                backup_path = self._create_backup_file(file_path)
                backup_files[file_path] = backup_path
                operation_log.append(f"Backed up {file_path} to {backup_path}")

        return RollbackPlan(original_files=original_files, backup_files=backup_files, operation_log=operation_log)

    def validate_rename_safety(self, impact_analysis: ImpactAnalysis) -> bool:
        """
        Validate if a rename operation is safe to execute.

        Args:
            impact_analysis: Impact analysis results

        Returns:
            True if rename is safe, False otherwise
        """
        # High risk threshold
        if impact_analysis.risk_score > 0.8:
            return False

        # Check for breaking changes in production
        if impact_analysis.breaking_changes and impact_analysis.risk_score > 0.6:
            return False

        # Check for too many affected files
        if len(impact_analysis.affected_files) > 50:
            return False

        return True

    def execute_rollback(self, operation_id: str) -> Dict[str, Any]:
        """
        Execute a rollback operation.

        Args:
            operation_id: ID of the operation to rollback

        Returns:
            Dictionary with rollback results
        """
        if operation_id not in self.rollback_plans:
            return {"status": RefactoringResult.FAILED.value, "error": f"No rollback plan found for operation {operation_id}"}

        rollback_plan = self.rollback_plans[operation_id]

        try:
            logger.info(f"Executing rollback for operation {operation_id}")

            # Restore original files
            for file_path, original_content in rollback_plan.original_files.items():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(original_content)

            # Clean up backup files
            for backup_path in rollback_plan.backup_files.values():
                if os.path.exists(backup_path):
                    os.remove(backup_path)

            # Remove rollback plan
            del self.rollback_plans[operation_id]

            return {
                "status": RefactoringResult.SUCCESS.value,
                "files_restored": len(rollback_plan.original_files),
                "operation_log": rollback_plan.operation_log,
            }

        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return {"status": RefactoringResult.FAILED.value, "error": str(e)}

    def _find_symbol_definition(
        self, symbol_name: str, symbol_type: str, scope: Optional[str] = None, project_path: Optional[str] = None
    ) -> Optional[SymbolInfo]:
        """Find the definition of a symbol."""
        # Map symbol types to Python keywords
        keyword = "def"
        if symbol_type.lower() == "class":
            keyword = "class"
        elif symbol_type.lower() == "function":
            keyword = "def"

        # Use enhanced search to find symbol definition
        search_context = SearchContext(query=f"{keyword} {symbol_name}", path=project_path or ".", strategy=SearchStrategy.EXACT)

        results = self.search_engine.search(search_context)

        # Handle the union return type
        search_results: List[EnhancedSearchResult] = []
        if isinstance(results, tuple):
            search_results = results[0]
        elif isinstance(results, list):
            search_results = results

        for result in search_results:
            if symbol_name in result.line_content:
                # Try to extract the symbol type from the context
                actual_symbol_type = SymbolType.FUNCTION
                if "class" in result.line_content.lower():
                    actual_symbol_type = SymbolType.CLASS
                elif "def" in result.line_content.lower():
                    actual_symbol_type = SymbolType.FUNCTION

                # Determine scope
                actual_scope = SymbolScope.GLOBAL
                if scope:
                    try:
                        actual_scope = SymbolScope(scope)
                    except ValueError:
                        actual_scope = SymbolScope.GLOBAL

                return SymbolInfo(
                    name=symbol_name,
                    symbol_type=actual_symbol_type,
                    file_path=result.file_path,
                    line_number=result.line_number,
                    scope=actual_scope,
                )

        return None

    def _find_all_references(self, symbol_info: SymbolInfo, project_path: Optional[str] = None) -> List[ReferenceInfo]:
        """Find all references to a symbol."""
        search_context = SearchContext(query=symbol_info.name, path=project_path or ".", strategy=SearchStrategy.EXACT)

        results = self.search_engine.search(search_context)

        references: List[ReferenceInfo] = []

        # Handle the union return type
        search_results: List[EnhancedSearchResult] = []
        if isinstance(results, tuple):
            search_results = results[0]
        elif isinstance(results, list):
            search_results = results

        for result in search_results:
            if result.file_path != symbol_info.file_path or result.line_number != symbol_info.line_number:
                references.append(
                    ReferenceInfo(
                        symbol_name=symbol_info.name,
                        reference_type=ReferenceType.REFERENCE,
                        file_path=result.file_path,
                        line_number=result.line_number,
                    )
                )

        return references

    def _execute_symbol_rename(
        self, symbol_info: SymbolInfo, new_name: str, references: List[ReferenceInfo], rollback_plan: RollbackPlan
    ) -> Dict[str, Any]:
        """Execute the actual symbol rename operation."""
        try:
            # Rename symbol definition
            self._rename_symbol_in_file(symbol_info, new_name, rollback_plan)

            # Update all references
            updated_references = 0
            for reference in references:
                if self._rename_reference_in_file(reference, new_name, rollback_plan):
                    updated_references += 1

            return {
                "status": RefactoringResult.SUCCESS.value,
                "symbol_renamed": symbol_info.name,
                "new_name": new_name,
                "references_updated": updated_references,
                "files_affected": len(set([symbol_info.file_path] + [ref.file_path for ref in references])),
            }

        except Exception as e:
            # Attempt rollback on failure
            self._execute_partial_rollback(rollback_plan)
            return {"status": RefactoringResult.FAILED.value, "error": str(e), "rollback_attempted": True}

    def _rename_symbol_in_file(self, symbol_info: SymbolInfo, new_name: str, rollback_plan: RollbackPlan) -> None:
        """Rename the symbol definition in its file."""
        with open(symbol_info.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace symbol definition
        lines = content.split("\n")
        if 0 <= symbol_info.line_number - 1 < len(lines):
            line = lines[symbol_info.line_number - 1]
            updated_line = line.replace(symbol_info.name, new_name, 1)
            lines[symbol_info.line_number - 1] = updated_line

        # Write back
        with open(symbol_info.file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        rollback_plan.operation_log.append(f"Renamed symbol definition in {symbol_info.file_path}")

    def _rename_reference_in_file(self, reference: ReferenceInfo, new_name: str, rollback_plan: RollbackPlan) -> bool:
        """Rename a reference in a file."""
        try:
            with open(reference.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace reference
            lines = content.split("\n")
            if 0 <= reference.line_number - 1 < len(lines):
                line = lines[reference.line_number - 1]
                updated_line = line.replace(reference.symbol_name, new_name, 1)
                lines[reference.line_number - 1] = updated_line

                with open(reference.file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                rollback_plan.operation_log.append(f"Updated reference in {reference.file_path}")
                return True

        except Exception as e:
            logger.warning(f"Failed to update reference in {reference.file_path}: {e}")

        return False

    def _create_backup_file(self, file_path: str) -> str:
        """Create a backup file."""
        timestamp = int(time.time())
        backup_dir = Path(tempfile.gettempdir()) / "fastapply_backups"
        backup_dir.mkdir(exist_ok=True)

        file_name = Path(file_path).name
        backup_path = backup_dir / f"{file_name}_{timestamp}.backup"

        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def _check_name_conflicts(self, new_name: str, symbol_type: str, project_path: Optional[str] = None) -> List[str]:
        """Check for name conflicts with the new name."""
        conflicts: List[str] = []

        search_context = SearchContext(query=new_name, path=project_path or ".", strategy=SearchStrategy.EXACT)

        results = self.search_engine.search(search_context)

        # Handle the union return type
        if isinstance(results, list):
            search_results: List[EnhancedSearchResult] = results
            for result in search_results:
                if symbol_type in result.symbol_type:
                    conflicts.append(f"{result.file_path}:{result.line_number}")

        return conflicts

    def _identify_external_apis(self, references: List[ReferenceInfo]) -> Set[str]:
        """Identify external API dependencies."""
        external_apis = set()

        for ref in references:
            # Simple heuristic: files outside the main source tree
            if any(pattern in ref.file_path.lower() for pattern in ["api", "public", "interface"]):
                external_apis.add(ref.file_path)

        return external_apis

    def _calculate_rename_risk_score(
        self, reference_count: int, external_api_count: int, test_file_count: int, has_conflicts: bool
    ) -> float:
        """Calculate risk score for rename operation."""
        risk_score = 0.0

        # Base risk from reference count
        risk_score += min(reference_count / 100.0, 0.5)

        # External API impact
        risk_score += min(external_api_count * 0.3, 0.3)

        # Test impact
        risk_score += min(test_file_count * 0.1, 0.2)

        # Name conflicts
        if has_conflicts:
            risk_score += 0.5

        return min(risk_score, 1.0)

    def _validate_rename_safety(self, impact_analysis: ImpactAnalysis) -> bool:
        """Validate if rename operation is safe."""
        return self.validate_rename_safety(impact_analysis)

    def _create_rename_rollback_plan(self, symbol_info: SymbolInfo, references: List[ReferenceInfo]) -> RollbackPlan:
        """Create rollback plan for rename operation."""
        return self.generate_rollback_plan(symbol_info, symbol_info.name, references)

    def _execute_partial_rollback(self, rollback_plan: RollbackPlan) -> None:
        """Execute partial rollback on failure."""
        logger.warning("Executing partial rollback due to operation failure")

        for file_path, original_content in rollback_plan.original_files.items():
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
            except Exception as e:
                logger.error(f"Failed to rollback {file_path}: {e}")


class CodeExtractionAndMovement:
    """
    Safe code extraction and movement operations.

    Implements Story 5.2: Code Extraction & Movement from the implementation specification.
    """

    def __init__(self, search_engine: Optional[EnhancedSearchInfrastructure] = None):
        """Initialize the code extraction and movement system."""
        self.search_engine = search_engine or EnhancedSearchInfrastructure()
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.operation_lock = threading.Lock()

    def extract_function_safely(
        self, source_range: Tuple[int, int], function_name: str, target_file: str, source_file: str, project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Safely extract a function/method to a new location.

        Args:
            source_range: (start_line, end_line) of code to extract
            function_name: Name for the extracted function
            target_file: File to extract the function to
            source_file: File to extract the function from
            project_path: Project root path

        Returns:
            Dictionary with extraction results
        """
        try:
            logger.info(f"Extracting function from {source_file} to {target_file}")

            # Analyze extraction safety
            safety_analysis = self.analyze_extraction_safety(source_range, source_file, project_path)

            if not safety_analysis["is_safe"]:
                return {"status": RefactoringResult.FAILED.value, "error": "Extraction deemed unsafe", "safety_analysis": safety_analysis}

            # Create rollback plan
            rollback_plan = self._create_extraction_rollback_plan(source_file, target_file)

            # Execute extraction
            with self.operation_lock:
                result = self._execute_function_extraction(source_range, function_name, target_file, source_file, rollback_plan)

                if result["status"] == RefactoringResult.SUCCESS.value:
                    operation_id = f"extract_{function_name}_{int(time.time())}"
                    self.rollback_plans[operation_id] = rollback_plan
                    result["operation_id"] = operation_id

                return result

        except Exception as e:
            logger.error(f"Error during function extraction: {e}")
            return {"status": RefactoringResult.FAILED.value, "error": str(e)}

    def move_symbol_safely(
        self,
        symbol_name: str,
        source_file: str,
        target_file: str,
        symbol_type: str = "function",
        scope: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Safely move a symbol between files.

        Args:
            symbol_name: Name of the symbol to move
            source_file: Current file containing the symbol
            target_file: Target file to move the symbol to
            symbol_type: Type of symbol
            scope: Optional scope
            project_path: Project root path

        Returns:
            Dictionary with move results
        """
        try:
            logger.info(f"Moving symbol {symbol_name} from {source_file} to {target_file}")

            # Analyze movement safety
            safety_analysis = self.analyze_movement_safety(symbol_name, source_file, target_file, project_path=project_path)

            if not safety_analysis["is_safe"]:
                return {"status": RefactoringResult.FAILED.value, "error": "Movement deemed unsafe", "safety_analysis": safety_analysis}

            # Find symbol definition
            symbol_info = self._find_symbol_definition(symbol_name, symbol_type, source_file)
            if not symbol_info:
                return {"status": RefactoringResult.FAILED.value, "error": f"Symbol '{symbol_name}' not found in {source_file}"}

            # Create rollback plan
            rollback_plan = self._create_movement_rollback_plan(source_file, target_file)

            # Execute movement
            with self.operation_lock:
                result = self._execute_symbol_movement(symbol_info, target_file, rollback_plan)

                if result["status"] == RefactoringResult.SUCCESS.value:
                    operation_id = f"move_{symbol_name}_{int(time.time())}"
                    self.rollback_plans[operation_id] = rollback_plan
                    result["operation_id"] = operation_id

                return result

        except Exception as e:
            logger.error(f"Error during symbol movement: {e}")
            return {"status": RefactoringResult.FAILED.value, "error": str(e)}

    def analyze_extraction_safety(
        self, source_range: Tuple[int, int], source_file: str, project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the safety of a code extraction operation.

        Args:
            source_range: (start_line, end_line) of code to extract
            source_file: File containing the code to extract
            project_path: Project root path

        Returns:
            Dictionary with safety analysis results
        """
        try:
            if not os.path.exists(source_file):
                return {"is_safe": False, "error": "Source file does not exist"}

            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            start_line, end_line = source_range

            if start_line < 1 or end_line > len(lines):
                return {"is_safe": False, "error": "Invalid source range"}

            # Extract the code to analyze
            code_to_extract = "\n".join(lines[start_line - 1 : end_line])

            # Check for dependencies
            dependencies = self._analyze_code_dependencies(code_to_extract, source_file)

            # Check for external references
            external_refs = self._analyze_external_references(code_to_extract, source_file)

            # Calculate safety score
            safety_score = self._calculate_extraction_safety_score(dependencies, external_refs)

            return {
                "is_safe": safety_score > 0.7,
                "safety_score": safety_score,
                "dependencies": dependencies,
                "external_references": external_refs,
                "risk_factors": self._identify_extraction_risks(code_to_extract),
            }

        except Exception as e:
            logger.error(f"Error analyzing extraction safety: {e}")
            return {"is_safe": False, "error": str(e)}

    def analyze_movement_safety(
        self, symbol_name: str, source_file: str, target_file: str, symbol_type: str = "function", project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the safety of a symbol movement operation.

        Args:
            symbol_name: Name of the symbol to move
            source_file: Current file containing the symbol
            target_file: Target file to move the symbol to
            symbol_type: Type of symbol ('function' or 'class')
            project_path: Project root path

        Returns:
            Dictionary with safety analysis results
        """
        try:
            # Check if files exist
            if not os.path.exists(source_file):
                return {"is_safe": False, "error": "Source file does not exist"}

            # Check if target file exists or can be created
            target_dir = os.path.dirname(target_file)
            if not os.path.exists(target_dir):
                return {"is_safe": False, "error": "Target directory does not exist"}

            # Find symbol definition
            symbol_info = self._find_symbol_definition(symbol_name, symbol_type, source_file)
            if not symbol_info:
                return {"is_safe": False, "error": f"Symbol '{symbol_name}' not found in source file"}

            # Analyze dependencies
            dependencies = self._analyze_symbol_dependencies(symbol_info)

            # Check for circular dependencies
            circular_deps = self._check_circular_dependencies(source_file, target_file)

            # Calculate safety score
            safety_score = self._calculate_movement_safety_score(dependencies, circular_deps)

            return {
                "is_safe": safety_score > 0.7,
                "safety_score": safety_score,
                "dependencies": dependencies,
                "circular_dependencies": circular_deps,
                "risk_factors": self._identify_movement_risks(symbol_info, target_file),
            }

        except Exception as e:
            logger.error(f"Error analyzing movement safety: {e}")
            return {"is_safe": False, "error": str(e)}

    def manage_import_dependencies(self, symbol_move: MoveOperation, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Manage import/export dependencies for symbol movement.

        Args:
            symbol_move: MoveOperation details
            project_path: Project root path

        Returns:
            Dictionary with dependency management results
        """
        try:
            results: Dict[str, Any] = {"imports_added": [], "imports_removed": [], "exports_added": [], "exports_removed": [], "errors": []}

            # Analyze required imports for target file
            required_imports = self._analyze_required_imports(symbol_move)

            # Add imports to target file
            for import_stmt in required_imports:
                if self._add_import_to_file(symbol_move.target_file, import_stmt):
                    results["imports_added"].append(import_stmt)

            # Remove unused imports from source file
            unused_imports = self._analyze_unused_imports(symbol_move.source_file, symbol_move.symbol_name)
            for import_stmt in unused_imports:
                if self._remove_import_from_file(symbol_move.source_file, import_stmt):
                    results["imports_removed"].append(import_stmt)

            return results

        except Exception as e:
            logger.error(f"Error managing import dependencies: {e}")
            return {"errors": [str(e)]}

    def _execute_function_extraction(
        self, source_range: Tuple[int, int], function_name: str, target_file: str, source_file: str, rollback_plan: RollbackPlan
    ) -> Dict[str, Any]:
        """Execute the actual function extraction."""
        try:
            # Read source file
            with open(source_file, "r", encoding="utf-8") as f:
                source_content = f.read()

            # Extract code
            lines = source_content.split("\n")
            start_line, end_line = source_range
            extracted_code = "\n".join(lines[start_line - 1 : end_line])

            # Remove code from source
            remaining_lines = lines[: start_line - 1] + lines[end_line:]

            # Write back to source
            with open(source_file, "w", encoding="utf-8") as f:
                f.write("\n".join(remaining_lines))

            # Add function to target
            target_content = ""
            if os.path.exists(target_file):
                with open(target_file, "r", encoding="utf-8") as f:
                    target_content = f.read()

            # Add extracted function
            function_code = f"\ndef {function_name}():\n    {extracted_code}\n"
            target_content += function_code

            with open(target_file, "w", encoding="utf-8") as f:
                f.write(target_content)

            rollback_plan.operation_log.append(f"Extracted function from {source_file} to {target_file}")

            return {
                "status": RefactoringResult.SUCCESS.value,
                "function_name": function_name,
                "source_file": source_file,
                "target_file": target_file,
                "lines_extracted": end_line - start_line + 1,
            }

        except Exception as e:
            self._execute_partial_rollback(rollback_plan)
            return {"status": RefactoringResult.FAILED.value, "error": str(e), "rollback_attempted": True}

    def _execute_symbol_movement(self, symbol_info: SymbolInfo, target_file: str, rollback_plan: RollbackPlan) -> Dict[str, Any]:
        """Execute the actual symbol movement."""
        try:
            # Read source file
            with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                source_content = f.read()

            # Extract symbol definition
            lines = source_content.split("\n")
            symbol_lines = lines[symbol_info.line_number - 1 : symbol_info.line_number]

            # Remove symbol from source
            remaining_lines = lines[: symbol_info.line_number - 1] + lines[symbol_info.line_number :]

            # Write back to source
            with open(symbol_info.file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(remaining_lines))

            # Add symbol to target
            target_content = ""
            if os.path.exists(target_file):
                with open(target_file, "r", encoding="utf-8") as f:
                    target_content = f.read()

            target_content += "\n" + "\n".join(symbol_lines) + "\n"

            with open(target_file, "w", encoding="utf-8") as f:
                f.write(target_content)

            rollback_plan.operation_log.append(f"Moved symbol from {symbol_info.file_path} to {target_file}")

            return {
                "status": RefactoringResult.SUCCESS.value,
                "symbol_name": symbol_info.name,
                "source_file": symbol_info.file_path,
                "target_file": target_file,
                "lines_moved": len(symbol_lines),
            }

        except Exception as e:
            self._execute_partial_rollback(rollback_plan)
            return {"status": RefactoringResult.FAILED.value, "error": str(e), "rollback_attempted": True}

    def _analyze_code_dependencies(self, code: str, source_file: str) -> List[str]:
        """Analyze dependencies in code to be extracted."""
        dependencies = []

        # Simple dependency detection
        import_pattern = r"import\s+(\w+)|from\s+(\w+)\s+import"
        matches = re.findall(import_pattern, code)

        for match in matches:
            for module in match:
                if module:
                    dependencies.append(module)

        return dependencies

    def _analyze_external_references(self, code: str, source_file: str) -> List[str]:
        """Analyze external references in code to be extracted."""
        external_refs = []

        # Look for references to variables/functions outside the extraction range
        var_pattern = r"\b([a-zA-Z_]\w*)\b"
        matches = re.findall(var_pattern, code)

        # Filter out keywords and common built-ins
        built_ins = {"def", "class", "if", "else", "for", "while", "return", "import", "from"}
        for match in matches:
            if match not in built_ins and len(match) > 2:
                external_refs.append(match)

        return external_refs

    def _calculate_extraction_safety_score(self, dependencies: List[str], external_refs: List[str]) -> float:
        """Calculate safety score for extraction operation."""
        safety_score = 1.0

        # Reduce score for dependencies
        safety_score -= len(dependencies) * 0.1

        # Reduce score for external references
        safety_score -= len(external_refs) * 0.05

        return max(safety_score, 0.0)

    def _identify_extraction_risks(self, code: str) -> List[str]:
        """Identify risks in extraction operation."""
        risks = []

        if "global " in code:
            risks.append("Uses global variables")

        if "nonlocal " in code:
            risks.append("Uses nonlocal variables")

        if "yield " in code:
            risks.append("Contains generator function")

        if "async def " in code:
            risks.append("Contains async function")

        return risks

    def _find_symbol_definition(self, symbol_name: str, symbol_type: str, source_file: str) -> Optional[SymbolInfo]:
        """Find symbol definition in source file."""
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Map symbol types to Python keywords
                keyword = "def"
                if symbol_type.lower() == "class":
                    keyword = "class"
                elif symbol_type.lower() == "function":
                    keyword = "def"

                if f"{keyword} {symbol_name}" in line:
                    # Try to determine the symbol type
                    actual_symbol_type = SymbolType.FUNCTION
                    if "class" in line.lower():
                        actual_symbol_type = SymbolType.CLASS
                    elif "def" in line.lower():
                        actual_symbol_type = SymbolType.FUNCTION

                    return SymbolInfo(
                        name=symbol_name,
                        symbol_type=actual_symbol_type,
                        file_path=source_file,
                        line_number=i + 1,
                    )

        except Exception as e:
            logger.error(f"Error finding symbol definition: {e}")

        return None

    def _analyze_symbol_dependencies(self, symbol_info: SymbolInfo) -> List[str]:
        """Analyze dependencies for a symbol."""
        try:
            with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            symbol_lines = lines[symbol_info.line_number - 1 : symbol_info.line_number]
            symbol_code = "\n".join(symbol_lines)

            return self._analyze_code_dependencies(symbol_code, symbol_info.file_path)

        except Exception as e:
            logger.error(f"Error analyzing symbol dependencies: {e}")
            return []

    def _check_circular_dependencies(self, source_file: str, target_file: str) -> List[str]:
        """Check for circular dependencies between files."""
        circular_deps = []

        # Simple check - could be enhanced with full dependency graph
        try:
            source_imports = self._get_file_imports(source_file)
            target_imports = self._get_file_imports(target_file)

            # Remove .py extension from basenames for comparison
            source_basename = os.path.basename(source_file).replace('.py', '')
            target_basename = os.path.basename(target_file).replace('.py', '')

            if target_basename in source_imports:
                circular_deps.append(f"{source_file} imports {target_file}")

            if source_basename in target_imports:
                circular_deps.append(f"{target_file} imports {source_file}")

        except Exception as e:
            logger.error(f"Error checking circular dependencies: {e}")

        return circular_deps

    def _get_file_imports(self, file_path: str) -> List[str]:
        """Get imports from a file."""
        imports = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            import_pattern = r"import\s+([a-zA-Z_][a-zA-Z0-9_]*)|from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import"
            matches = re.findall(import_pattern, content)

            for match in matches:
                for module in match:
                    if module:
                        imports.append(module)

        except Exception as e:
            logger.error(f"Error getting file imports: {e}")

        return imports

    def _calculate_movement_safety_score(self, dependencies: List[str], circular_deps: List[str]) -> float:
        """Calculate safety score for movement operation."""
        safety_score = 1.0

        # Reduce score for dependencies
        safety_score -= len(dependencies) * 0.1

        # Reduce score for circular dependencies
        safety_score -= len(circular_deps) * 0.3

        return max(safety_score, 0.0)

    def _identify_movement_risks(self, symbol_info: SymbolInfo, target_file: str) -> List[str]:
        """Identify risks in movement operation."""
        risks = []

        try:
            with open(symbol_info.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            symbol_lines = lines[symbol_info.line_number - 1 : symbol_info.line_number]
            symbol_code = "\n".join(symbol_lines)

            if "__" in symbol_info.name:
                risks.append("Private/dunder method - movement may break encapsulation")

            if "property" in symbol_code:
                risks.append("Property decorator - movement may affect class structure")

            if "staticmethod" in symbol_code or "classmethod" in symbol_code:
                risks.append("Static/class method - movement may affect class structure")

        except Exception as e:
            logger.error(f"Error identifying movement risks: {e}")

        return risks

    def _analyze_required_imports(self, symbol_move: MoveOperation) -> List[str]:
        """Analyze required imports for target file."""
        # This is a simplified version - could be enhanced with full dependency analysis
        return []

    def _add_import_to_file(self, target_file: str, import_stmt: str) -> bool:
        """Add import statement to target file."""
        try:
            if not os.path.exists(target_file):
                return False

            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()

            if import_stmt in content:
                return True  # Already imported

            # Add import at the top
            lines = content.split("\n")
            insert_pos = 0

            # Find end of existing imports
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_pos = i + 1
                elif line.strip() and not line.startswith("#") and insert_pos > 0:
                    break

            lines.insert(insert_pos, import_stmt)

            with open(target_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            logger.error(f"Error adding import to file: {e}")
            return False

    def _analyze_unused_imports(self, source_file: str, symbol_name: str) -> List[str]:
        """Analyze unused imports in source file after symbol removal."""
        # This is a simplified version - could be enhanced with full usage analysis
        return []

    def _remove_import_from_file(self, source_file: str, import_stmt: str) -> bool:
        """Remove import statement from source file."""
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()

            if import_stmt not in content:
                return True  # Already removed

            # Remove the import line
            lines = content.split("\n")
            lines = [line for line in lines if line.strip() != import_stmt.strip()]

            with open(source_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return True

        except Exception as e:
            logger.error(f"Error removing import from file: {e}")
            return False

    def _create_extraction_rollback_plan(self, source_file: str, target_file: str) -> RollbackPlan:
        """Create rollback plan for extraction operation."""
        original_files = {}
        backup_files = {}

        # Backup source file
        if os.path.exists(source_file):
            with open(source_file, "r", encoding="utf-8") as f:
                original_files[source_file] = f.read()
            backup_files[source_file] = self._create_backup_file(source_file)

        # Backup target file
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                original_files[target_file] = f.read()
            backup_files[target_file] = self._create_backup_file(target_file)

        return RollbackPlan(original_files=original_files, backup_files=backup_files, operation_log=[])

    def _create_movement_rollback_plan(self, source_file: str, target_file: str) -> RollbackPlan:
        """Create rollback plan for movement operation."""
        return self._create_extraction_rollback_plan(source_file, target_file)

    def _create_backup_file(self, file_path: str) -> str:
        """Create a backup file."""
        timestamp = int(time.time())
        backup_dir = Path(tempfile.gettempdir()) / "fastapply_backups"
        backup_dir.mkdir(exist_ok=True)

        file_name = Path(file_path).name
        backup_path = backup_dir / f"{file_name}_{timestamp}.backup"

        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    def _execute_partial_rollback(self, rollback_plan: RollbackPlan) -> None:
        """Execute partial rollback on failure."""
        logger.warning("Executing partial rollback due to operation failure")

        for file_path, original_content in rollback_plan.original_files.items():
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
            except Exception as e:
                logger.error(f"Failed to rollback {file_path}: {e}")


# Factory functions for MCP integration
def create_safe_refactoring_tools() -> Tuple[SafeSymbolRenaming, CodeExtractionAndMovement]:
    """Create instances of safe refactoring tools."""
    search_engine = EnhancedSearchInfrastructure()
    return SafeSymbolRenaming(search_engine), CodeExtractionAndMovement(search_engine)
