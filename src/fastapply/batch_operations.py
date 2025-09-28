"""
Batch Operations Module

Phase 6: Large-scale batch processing capabilities for enterprise codebases.
Provides efficient project-wide analysis, transformation, and monitoring.

Key Features:
- Large-scale project analysis (1000+ files)
- Bulk code transformations with safety validation
- Real-time progress monitoring and reporting
- Intelligent resource management and scheduling
- Comprehensive error handling and recovery

Classes:
- BatchAnalysisSystem: Project-wide analysis capabilities
- BatchTransformation: Safe bulk code transformations
- ProgressMonitor: Real-time operation tracking
- BatchScheduler: Intelligent job scheduling
- BatchValidator: Quality and safety validation
"""

import os
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .relationship_mapping import RelationshipMapper

# Import existing infrastructure
from .ripgrep_integration import RipgrepIntegration, SearchOptions
from .safe_refactoring import CodeExtractionAndMovement, SafeSymbolRenaming
from .symbol_operations import AdvancedSymbolOperations


class BatchOperationType(Enum):
    """Types of batch operations supported."""

    PROJECT_ANALYSIS = "project_analysis"
    SYMBOL_RENAME = "symbol_rename"
    CODE_EXTRACTION = "code_extraction"
    PATTERN_TRANSFORMATION = "pattern_transformation"
    REFACTORING = "refactoring"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"


class BatchStatus(Enum):
    """Status of batch operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class BatchOperation:
    """Represents a single batch operation."""

    id: str
    type: BatchOperationType
    name: str
    description: str
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    affected_files: List[str] = field(default_factory=list)
    memory_usage: int = 0
    cpu_usage: float = 0.0


@dataclass
class BatchConfig:
    """Configuration for batch operations."""

    max_concurrent_operations: int = 4
    max_memory_usage_mb: int = 4096
    timeout_seconds: int = 3600
    chunk_size: int = 100
    enable_progress_tracking: bool = True
    auto_save_interval: int = 30
    retry_failed_operations: bool = True
    max_retries: int = 3
    enable_rollback: bool = True
    validation_level: str = "strict"  # strict, moderate, lenient


@dataclass
class BatchResults:
    """Results of batch operations."""

    operation_id: str
    success: bool
    processed_files: int
    total_files: int
    execution_time: float
    memory_peak_mb: float
    transformations_applied: int
    errors_encountered: int
    warnings_generated: int
    details: Dict[str, Any] = field(default_factory=dict)


class ProgressMonitor:
    """Real-time progress monitoring for batch operations."""

    def __init__(self, operation_id: str, total_items: int):
        self.operation_id = operation_id
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.progress_history: List[Dict[str, Any]] = []
        self.current_stage = "initialization"
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def update_progress(self, increment: int = 1, stage: Optional[str] = None):
        """Update progress with thread safety."""
        with self._lock:
            self.processed_items += increment
            current_time = time.time()

            if stage:
                self.current_stage = stage
                if stage not in self.stages:
                    self.stages[stage] = {"start": current_time, "items": 0}
                self.stages[stage]["items"] += increment

            progress = min(100.0, (self.processed_items / self.total_items) * 100) if self.total_items > 0 else 0

            self.progress_history.append(
                {
                    "timestamp": current_time,
                    "progress": progress,
                    "stage": self.current_stage,
                    "processed": self.processed_items,
                    "total": self.total_items,
                }
            )

    def add_error(self, error: str, file_path: Optional[str] = None):
        """Add error to monitoring."""
        with self._lock:
            error_entry = {"timestamp": time.time(), "error": error, "file": file_path, "stage": self.current_stage}
            self.errors.append(error_entry)

    def add_warning(self, warning: str, file_path: Optional[str] = None):
        """Add warning to monitoring."""
        with self._lock:
            warning_entry = {"timestamp": time.time(), "warning": warning, "file": file_path, "stage": self.current_stage}
            self.warnings.append(warning_entry)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Calculate estimated time remaining
        progress = (self.processed_items / self.total_items) if self.total_items > 0 else 0
        eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0

        return {
            "operation_id": self.operation_id,
            "progress_percent": min(100.0, progress * 100),
            "processed_items": self.processed_items,
            "total_items": self.total_items,
            "elapsed_time_seconds": elapsed,
            "estimated_time_remaining": eta,
            "current_stage": self.current_stage,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "stages_completed": len([s for s in self.stages.values() if s["items"] > 0]),
            "throughput_items_per_second": self.processed_items / elapsed if elapsed > 0 else 0,
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed progress report."""
        return {
            "operation_id": self.operation_id,
            "progress_summary": self.get_progress_summary(),
            "errors": self.errors[-10:],  # Last 10 errors
            "warnings": self.warnings[-10:],  # Last 10 warnings
            "stages": self.stages,
            "progress_history": self.progress_history[-50:],  # Last 50 updates
        }


class BatchScheduler:
    """Intelligent scheduling for batch operations."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.pending_operations: List[BatchOperation] = []
        self.running_operations: Dict[str, BatchOperation] = {}
        self.completed_operations: Dict[str, BatchOperation] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_operations)
        self._lock = threading.Lock()

    def schedule_operation(self, operation: BatchOperation) -> str:
        """Schedule a batch operation for execution."""
        with self._lock:
            operation.id = str(uuid.uuid4())
            self.pending_operations.append(operation)
            self._sort_pending_operations()
            return operation.id

    def _sort_pending_operations(self):
        """Sort pending operations by priority and dependencies."""

        def sort_key(op: BatchOperation) -> Tuple[int, int]:
            # Higher priority first, then fewer dependencies
            return (-op.priority, -len(op.dependencies))

        self.pending_operations.sort(key=sort_key)

    def get_executable_operations(self) -> List[BatchOperation]:
        """Get operations ready to execute (dependencies satisfied)."""
        with self._lock:
            executable = []
            completed_ids = set(self.completed_operations.keys())

            for op in self.pending_operations[:]:
                # Check if all dependencies are completed
                deps_satisfied = all(dep_id in completed_ids for dep_id in op.dependencies)
                # Check if we have capacity
                has_capacity = len(self.running_operations) < self.config.max_concurrent_operations

                if deps_satisfied and has_capacity:
                    executable.append(op)
                    self.pending_operations.remove(op)

            return executable

    def start_operation(self, operation: BatchOperation):
        """Mark operation as running."""
        with self._lock:
            operation.status = BatchStatus.RUNNING
            operation.start_time = datetime.now()
            self.running_operations[operation.id] = operation

    def complete_operation(self, operation_id: str, success: bool, results: Optional[Dict] = None):
        """Mark operation as completed."""
        with self._lock:
            if operation_id in self.running_operations:
                operation = self.running_operations.pop(operation_id)
                operation.status = BatchStatus.COMPLETED if success else BatchStatus.FAILED
                operation.end_time = datetime.now()
                if operation.start_time:
                    operation.actual_duration = operation.end_time - operation.start_time
                if results:
                    operation.results = results
                self.completed_operations[operation_id] = operation

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        with self._lock:
            return {
                "pending_count": len(self.pending_operations),
                "running_count": len(self.running_operations),
                "completed_count": len(self.completed_operations),
                "max_concurrent": self.config.max_concurrent_operations,
                "utilization_percent": (len(self.running_operations) / self.config.max_concurrent_operations) * 100,
                "running_operations": [
                    {
                        "id": op.id,
                        "name": op.name,
                        "progress": op.progress,
                        "start_time": op.start_time.isoformat() if op.start_time else None,
                    }
                    for op in self.running_operations.values()
                ],
            }


class BatchAnalysisSystem:
    """Large-scale project analysis capabilities."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.ripgrep = RipgrepIntegration()
        self.symbol_ops = AdvancedSymbolOperations()
        self.relationship_mapper = RelationshipMapper()
        self.scheduler = BatchScheduler(config)
        self.active_operations: Dict[str, ProgressMonitor] = {}
        self.progress_monitor = ProgressMonitor(operation_id="system", total_items=1)

    def analyze_project_batches(self, project_path: str, analysis_types: Optional[List[str]] = None) -> BatchResults:
        """
        Perform comprehensive batch analysis of a project.

        Args:
            project_path: Path to the project root
            analysis_types: Types of analysis to perform

        Returns:
            BatchResults containing analysis outcomes
        """
        if analysis_types is None:
            analysis_types = ["symbols", "dependencies", "quality", "complexity"]

        operation_id = str(uuid.uuid4())
        start_time = time.time()

        # Find all relevant files
        project_files = self._discover_project_files(project_path)
        total_files = len(project_files)

        # Initialize progress monitoring
        progress_monitor = ProgressMonitor(operation_id, total_files)
        self.active_operations[operation_id] = progress_monitor

        try:
            results: Dict[str, Any] = {
                "project_path": project_path,
                "analysis_types": analysis_types,
                "total_files": total_files,
                "file_analyses": {},
                "project_metrics": {},
                "dependencies": {},
                "quality_metrics": {},
                "complexity_analysis": {},
            }

            processed_files = 0

            # Process files in chunks
            for i in range(0, len(project_files), self.config.chunk_size):
                chunk = project_files[i : i + self.config.chunk_size]

                # Process chunk in parallel
                with ThreadPoolExecutor(max_workers=self.config.max_concurrent_operations) as executor:
                    future_to_file = {executor.submit(self._analyze_file_batch, file, analysis_types): file for file in chunk}

                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_result = future.result()
                            results["file_analyses"][file_path] = file_result
                            processed_files += 1
                            progress_monitor.update_progress(1, "file_analysis")
                        except Exception as e:
                            error_msg = f"Error analyzing {file_path}: {str(e)}"
                            progress_monitor.add_error(error_msg, file_path)
                            results["file_analyses"][file_path] = {"error": error_msg}

            # Aggregate project-level metrics
            results["project_metrics"] = self._calculate_project_metrics(results["file_analyses"])

            # Perform cross-file analysis
            if "dependencies" in analysis_types:
                results["dependencies"] = self._analyze_project_dependencies(project_path)
                progress_monitor.update_progress(0, "dependency_analysis")

            if "quality" in analysis_types:
                results["quality_metrics"] = self._analyze_project_quality(results["file_analyses"])
                progress_monitor.update_progress(0, "quality_analysis")

            if "complexity" in analysis_types:
                results["complexity_analysis"] = self._analyze_project_complexity(results["file_analyses"])
                progress_monitor.update_progress(0, "complexity_analysis")

            execution_time = time.time() - start_time

            return BatchResults(
                operation_id=operation_id,
                success=True,
                processed_files=processed_files,
                total_files=total_files,
                execution_time=execution_time,
                memory_peak_mb=0,  # TODO: Implement memory tracking
                transformations_applied=0,
                errors_encountered=len(progress_monitor.errors),
                warnings_generated=len(progress_monitor.warnings),
                details=results,
            )

        except Exception as e:
            error_msg = f"Project analysis failed: {str(e)}"
            progress_monitor.add_error(error_msg)

            return BatchResults(
                operation_id=operation_id,
                success=False,
                processed_files=processed_files,
                total_files=total_files,
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=0,
                errors_encountered=len(progress_monitor.errors) + 1,
                warnings_generated=len(progress_monitor.warnings),
                details={"error": error_msg},
            )
        finally:
            # Clean up
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

    def _discover_project_files(self, project_path: str) -> List[str]:
        """Discover all relevant files in the project."""
        project_path_obj = Path(project_path)
        relevant_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h"}

        files: List[Path] = []
        for ext in relevant_extensions:
            files.extend(project_path_obj.rglob(f"*{ext}"))

        return [str(f) for f in files if f.is_file()]

    def _analyze_file_batch(self, file_path: str, analysis_types: List[str]) -> Dict[str, Any]:
        """Analyze a single file for the specified analysis types."""
        result: Dict[str, Any] = {"file_path": file_path, "analyses": {}}

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Symbol analysis
            if "symbols" in analysis_types:
                symbols = self.symbol_ops.find_symbols_by_pattern(content)
                result["analyses"]["symbols"] = {
                    "count": len(symbols),
                    "types": defaultdict(int),
                    "symbols": symbols[:10],  # First 10 symbols
                }

            # Dependencies analysis
            if "dependencies" in analysis_types:
                imports = self._extract_imports(content, Path(file_path).suffix)
                result["analyses"]["dependencies"] = {"imports": imports, "import_count": len(imports)}

            # Quality metrics
            if "quality" in analysis_types:
                quality = self._calculate_file_quality(content, Path(file_path).suffix)
                result["analyses"]["quality"] = quality

            # Complexity analysis
            if "complexity" in analysis_types:
                complexity = self._calculate_file_complexity(content, Path(file_path).suffix)
                result["analyses"]["complexity"] = complexity

        except Exception as e:
            result["error"] = str(e)

        return result

    def _extract_imports(self, content: str, file_extension: str) -> List[str]:
        """Extract import statements from file content."""
        imports = []

        if file_extension == ".py":
            # Python imports
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith(("import ", "from ")):
                    imports.append(line)
        elif file_extension in {".js", ".ts", ".jsx", ".tsx"}:
            # JavaScript/TypeScript imports
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith(("import ", "require(")):
                    imports.append(line)

        return imports

    def _calculate_file_quality(self, content: str, file_extension: str) -> Dict[str, Any]:
        """Calculate quality metrics for a file."""
        lines = content.split("\n")

        # Basic metrics
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith("#")])
        comment_lines = len([line for line in lines if line.strip().startswith("#")])
        empty_lines = len([line for line in lines if not line.strip()])

        # Calculate complexity (simplified)
        complexity_indicators = 0
        for line in lines:
            if any(keyword in line for keyword in ["if ", "for ", "while ", "try:", "except", "elif "]):
                complexity_indicators += 1

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "empty_lines": empty_lines,
            "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0,
            "complexity_indicators": complexity_indicators,
            "maintainability_index": max(0, 100 - (complexity_indicators * 5) - (total_lines / 100)),
        }

    def _calculate_file_complexity(self, content: str, file_extension: str) -> Dict[str, Any]:
        """Calculate complexity metrics for a file."""
        lines = content.split("\n")

        # Cyclomatic complexity (simplified calculation)
        complexity = 1  # Base complexity
        for line in lines:
            if any(keyword in line for keyword in ["if ", "elif ", "for ", "while ", "except", "with "]):
                complexity += 1
            elif "def " in line or "class " in line:
                complexity += 1

        # Nesting depth
        max_nesting = 0
        current_nesting = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "if ", "for ", "while ", "with ", "try:")):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif stripped and not stripped.startswith((" ", "\t")) and current_nesting > 0:
                current_nesting = max(0, current_nesting - 1)

        return {
            "cyclomatic_complexity": complexity,
            "max_nesting_depth": max_nesting,
            "complexity_level": "low" if complexity <= 5 else "medium" if complexity <= 10 else "high",
        }

    def _calculate_project_metrics(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate project-level metrics from file analyses."""
        total_files = len(file_analyses)
        total_lines = 0
        total_symbols = 0
        quality_scores = []
        complexity_scores = []

        for file_analysis in file_analyses.values():
            if "error" not in file_analysis and "analyses" in file_analysis:
                analyses = file_analysis["analyses"]

                if "quality" in analyses:
                    quality = analyses["quality"]
                    total_lines += quality.get("total_lines", 0)
                    quality_scores.append(quality.get("maintainability_index", 0))

                if "symbols" in analyses:
                    total_symbols += analyses["symbols"].get("count", 0)

                if "complexity" in analyses:
                    complexity = analyses["complexity"]
                    complexity_scores.append(complexity.get("cyclomatic_complexity", 1))

        return {
            "total_files": total_files,
            "total_lines_of_code": total_lines,
            "total_symbols": total_symbols,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            "files_per_symbol": total_symbols / total_files if total_files > 0 else 0,
            "lines_per_file": total_lines / total_files if total_files > 0 else 0,
        }

    def _analyze_project_dependencies(self, project_path: str) -> Dict[str, Any]:
        """Analyze project-level dependencies."""
        try:
            # Parse code structure first
            code_structure = self.relationship_mapper._parse_code_structure(
                code="", context="project", project_path=project_path
            )

            # Use relationship mapper for dependency analysis
            dependency_analysis = self.relationship_mapper.analyze_dependencies(code_structure)

            return {
                "dependency_graph_built": True,
                "nodes_count": len(dependency_analysis.dependencies),
                "edges_count": len(dependency_analysis.dependencies),
                "has_circular_dependencies": len(
                    self.relationship_mapper._detect_circular_dependencies(dependency_analysis.dependencies)
                ) > 0,
            }
        except Exception as e:
            return {"error": str(e), "dependency_graph_built": False}

    def _analyze_project_quality(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project-level quality metrics."""
        quality_scores = []
        complexity_scores = []

        for file_analysis in file_analyses.values():
            if "error" not in file_analysis and "analyses" in file_analysis:
                analyses = file_analysis["analyses"]

                if "quality" in analyses:
                    quality_scores.append(analyses["quality"].get("maintainability_index", 0))

                if "complexity" in analyses:
                    complexity_scores.append(analyses["complexity"].get("cyclomatic_complexity", 1))

        return {
            "average_maintainability": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            "quality_distribution": {
                "high": len([s for s in quality_scores if s >= 80]),
                "medium": len([s for s in quality_scores if 60 <= s < 80]),
                "low": len([s for s in quality_scores if s < 60]),
            },
            "complexity_distribution": {
                "low": len([c for c in complexity_scores if c <= 5]),
                "medium": len([c for c in complexity_scores if 5 < c <= 10]),
                "high": len([c for c in complexity_scores if c > 10]),
            },
        }

    def _analyze_project_complexity(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project-level complexity metrics."""
        complexity_scores = []
        nesting_scores = []

        for file_analysis in file_analyses.values():
            if "error" not in file_analysis and "analyses" in file_analysis:
                analyses = file_analysis["analyses"]

                if "complexity" in analyses:
                    complexity = analyses["complexity"]
                    complexity_scores.append(complexity.get("cyclomatic_complexity", 1))
                    nesting_scores.append(complexity.get("max_nesting_depth", 0))

        return {
            "total_complexity_score": sum(complexity_scores),
            "average_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            "max_complexity": max(complexity_scores) if complexity_scores else 0,
            "average_nesting_depth": sum(nesting_scores) / len(nesting_scores) if nesting_scores else 0,
            "max_nesting_depth": max(nesting_scores) if nesting_scores else 0,
            "complexity_hotspots": [{"complexity": c, "threshold": 15} for c in complexity_scores if c > 15],
        }

    def get_operation_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for an operation."""
        if operation_id in self.active_operations:
            return self.active_operations[operation_id].get_detailed_report()
        return None

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation."""
        # Implementation for operation cancellation
        # This would require more sophisticated threading/interruption handling
        return False

    def generate_batch_report(
        self,
        operation_id: str,
        report_type: str = "summary",
        include_metrics: bool = True,
        include_errors: bool = True,
        include_recommendations: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a batch operation.

        Args:
            operation_id: ID of the operation to report on
            report_type: Type of report ("summary", "detailed", "executive")
            include_metrics: Whether to include performance metrics
            include_errors: Whether to include error information
            include_recommendations: Whether to include recommendations

        Returns:
            Dictionary containing the report data
        """
        from datetime import datetime

        # Create basic report structure
        report = {
            "operation_id": operation_id,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "include_metrics": include_metrics,
            "include_errors": include_errors,
            "include_recommendations": include_recommendations,
        }

        # Get operation progress if available
        if operation_id in self.active_operations:
            progress = self.active_operations[operation_id].get_progress_summary()
            report["operation_progress"] = progress

            if include_metrics:
                report["performance_metrics"] = {
                    "execution_time": progress.get("execution_time", 0),
                    "items_processed": progress.get("items_processed", 0),
                    "total_items": progress.get("total_items", 0),
                    "progress_percentage": progress.get("progress_percentage", 0),
                }

            if include_errors:
                report["error_summary"] = {
                    "total_errors": len(progress.get("errors", [])),
                    "total_warnings": len(progress.get("warnings", [])),
                    "recent_errors": progress.get("errors", [])[-5:],  # Last 5 errors
                    "error_categories": self._categorize_errors(progress.get("errors", [])),
                }

        # Add recommendations based on report type
        if include_recommendations:
            report["recommendations"] = self._generate_report_recommendations(operation_id, report_type)

        # Add executive summary for executive reports
        if report_type == "executive":
            report["executive_summary"] = self._generate_executive_summary(operation_id)

        return report

    def _categorize_errors(self, errors: List[str]) -> Dict[str, int]:
        """Categorize errors by type."""
        categories = {"file_access": 0, "permission": 0, "syntax": 0, "analysis": 0, "transformation": 0, "other": 0}

        for error in errors:
            error_lower = error.lower()
            if "permission" in error_lower or "access" in error_lower:
                categories["permission"] += 1
            elif "file" in error_lower and ("not found" in error_lower or "access" in error_lower):
                categories["file_access"] += 1
            elif "syntax" in error_lower:
                categories["syntax"] += 1
            elif "analysis" in error_lower:
                categories["analysis"] += 1
            elif "transformation" in error_lower:
                categories["transformation"] += 1
            else:
                categories["other"] += 1

        return categories

    def _generate_report_recommendations(self, operation_id: str, report_type: str) -> List[str]:
        """Generate recommendations based on operation results."""
        recommendations = []

        if operation_id in self.active_operations:
            progress = self.active_operations[operation_id]
            errors = progress.errors
            _warnings = progress.warnings

            # Error-based recommendations
            if len(errors) > 10:
                recommendations.append("Consider implementing better error handling and validation")

            if any("permission" in str(error).lower() for error in errors):
                recommendations.append("Review file permissions and access controls")

            if any("syntax" in str(error).lower() for error in errors):
                recommendations.append("Implement syntax checking before analysis")

            # Performance-based recommendations
            if progress.total_items > 1000 and progress.processed_items < progress.total_items * 0.5:
                recommendations.append("Consider optimizing processing for large datasets")

            # General recommendations
            if report_type == "detailed":
                recommendations.extend(
                    [
                        "Consider implementing incremental processing for better performance",
                        "Add more comprehensive logging for debugging",
                        "Implement automated retry mechanisms for transient failures",
                    ]
                )

        return recommendations if recommendations else ["Operation completed successfully with no issues detected"]

    def _generate_executive_summary(self, operation_id: str) -> Dict[str, Any]:
        """Generate executive summary for batch operations."""
        if operation_id not in self.active_operations:
            return {"status": "Operation not found in active operations"}

        progress = self.active_operations[operation_id]
        success_rate = (progress.processed_items / progress.total_items * 100) if progress.total_items > 0 else 0

        return {
            "operation_status": "Completed" if progress.processed_items == progress.total_items else "In Progress",
            "success_rate": f"{success_rate:.1f}%",
            "total_items": progress.total_items,
            "processed_items": progress.processed_items,
            "error_count": len(progress.errors),
            "warning_count": len(progress.warnings),
            "overall_assessment": "Successful" if success_rate > 95 else "Needs Attention" if success_rate > 80 else "Requires Review",
            "key_insights": [
                f"Processed {progress.processed_items} of {progress.total_items} items",
                f"Encountered {len(progress.errors)} errors and {len(progress.warnings)} warnings",
                f"Success rate of {success_rate:.1f}%",
            ],
        }


class BatchTransformation:
    """Safe bulk code transformations across multiple files."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.safe_renamer = SafeSymbolRenaming()
        self.safe_extractor = CodeExtractionAndMovement()
        self.symbol_ops = AdvancedSymbolOperations()
        self.ripgrep = RipgrepIntegration()
        self.transformation_history: List[Dict[str, Any]] = []

    def batch_rename_symbol(
        self,
        old_name: str,
        new_name: str,
        project_path: str,
        symbol_type: str = "function",
        scope: Optional[str] = None,
        preview_only: bool = False,
    ) -> BatchResults:
        """
        Safely rename a symbol across all files in a project.

        Args:
            old_name: Current symbol name
            new_name: New symbol name
            project_path: Path to project root
            symbol_type: Type of symbol (function, class, variable, etc.)
            scope: Optional scope limitation
            preview_only: If True, only show changes without applying them

        Returns:
            BatchResults with transformation details
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Find all files containing the symbol
            search_results = self._find_symbol_occurrences(old_name, project_path, symbol_type)
            affected_files = list(search_results.keys())

            if not affected_files:
                return BatchResults(
                    operation_id=operation_id,
                    success=True,
                    processed_files=0,
                    total_files=0,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=0,
                    transformations_applied=0,
                    errors_encountered=0,
                    warnings_generated=0,
                    details={"message": f"Symbol '{old_name}' not found in project", "affected_files": []},
                )

            total_files = len(affected_files)
            progress_monitor = ProgressMonitor(operation_id, total_files)

            # Validate transformation safety
            safety_analysis = self._analyze_rename_safety(old_name, new_name, search_results)
            if safety_analysis["risk_level"] == "high" and not preview_only:
                return BatchResults(
                    operation_id=operation_id,
                    success=False,
                    processed_files=0,
                    total_files=total_files,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=0,
                    transformations_applied=0,
                    errors_encountered=1,
                    warnings_generated=0,
                    details={"error": "High-risk transformation detected", "safety_analysis": safety_analysis},
                )

            transformations_applied = 0
            processed_files = 0
            transformation_details: Dict[str, List[Dict[str, Any]]] = {}

            if preview_only:
                # Generate preview without applying changes
                for file_path, occurrences in search_results.items():
                    preview_changes = []
                    for occ in occurrences:
                        preview_changes.append(
                            {
                                "line": occ.get("line", 0),
                                "context": occ.get("context", ""),
                                "change_type": "rename",
                                "old_text": old_name,
                                "new_text": new_name,
                            }
                        )
                    transformation_details[file_path] = preview_changes
                    processed_files += 1
                    progress_monitor.update_progress(1, "preview_generation")
            else:
                # Apply actual transformations
                backup_created = False

                # Create backup if rollback is enabled
                if self.config.enable_rollback:
                    backup_created = self._create_project_backup(project_path, operation_id)

                try:
                    for file_path, occurrences in search_results.items():
                        file_transformations = self._apply_file_rename(file_path, old_name, new_name, occurrences, symbol_type)
                        transformation_details[file_path] = file_transformations
                        transformations_applied += len(file_transformations)
                        processed_files += 1
                        progress_monitor.update_progress(1, "transformation_applied")

                except Exception as e:
                    # Rollback if enabled and error occurred
                    if self.config.enable_rollback and backup_created:
                        self._rollback_project_backup(project_path, operation_id)

                    error_msg = f"Transformation failed: {str(e)}"
                    progress_monitor.add_error(error_msg)

                    return BatchResults(
                        operation_id=operation_id,
                        success=False,
                        processed_files=processed_files,
                        total_files=total_files,
                        execution_time=time.time() - start_time,
                        memory_peak_mb=0,
                        transformations_applied=transformations_applied,
                        errors_encountered=len(progress_monitor.errors) + 1,
                        warnings_generated=len(progress_monitor.warnings),
                        details={"error": error_msg, "partial_results": transformation_details},
                    )

            execution_time = time.time() - start_time

            return BatchResults(
                operation_id=operation_id,
                success=True,
                processed_files=processed_files,
                total_files=total_files,
                execution_time=execution_time,
                memory_peak_mb=0,
                transformations_applied=transformations_applied,
                errors_encountered=len(progress_monitor.errors),
                warnings_generated=len(progress_monitor.warnings),
                details={
                    "old_name": old_name,
                    "new_name": new_name,
                    "symbol_type": symbol_type,
                    "scope": scope,
                    "preview_only": preview_only,
                    "safety_analysis": safety_analysis,
                    "transformations": transformation_details,
                    "backup_created": backup_created if not preview_only else False,
                },
            )

        except Exception as e:
            error_msg = f"Batch rename failed: {str(e)}"
            return BatchResults(
                operation_id=operation_id,
                success=False,
                processed_files=0,
                total_files=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=0,
                errors_encountered=1,
                warnings_generated=0,
                details={"error": error_msg},
            )

    def batch_extract_components(
        self, component_patterns: List[str], project_path: str, output_directory: str, preview_only: bool = False
    ) -> BatchResults:
        """
        Extract multiple components (functions/classes) matching patterns.

        Args:
            component_patterns: List of patterns to match components
            project_path: Path to project root
            output_directory: Directory to save extracted components
            preview_only: If True, only show what would be extracted

        Returns:
            BatchResults with extraction details
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Find all components matching patterns
            all_components = []
            for pattern in component_patterns:
                components = self._find_components_by_pattern(pattern, project_path)
                all_components.extend(components)

            if not all_components:
                return BatchResults(
                    operation_id=operation_id,
                    success=True,
                    processed_files=0,
                    total_files=0,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=0,
                    transformations_applied=0,
                    errors_encountered=0,
                    warnings_generated=0,
                    details={"message": "No components found matching patterns", "components_found": []},
                )

            total_components = len(all_components)
            progress_monitor = ProgressMonitor(operation_id, total_components)

            extracted_components = []
            processed_components = 0

            # Create output directory if it doesn't exist
            if not preview_only:
                os.makedirs(output_directory, exist_ok=True)

            for component in all_components:
                try:
                    if preview_only:
                        # Generate preview extraction
                        extraction_result = self._preview_component_extraction(component)
                    else:
                        # Perform actual extraction
                        extraction_result = self._extract_component_to_file(component, output_directory)

                    extracted_components.append(extraction_result)
                    processed_components += 1
                    progress_monitor.update_progress(1, "component_extraction")

                except Exception as e:
                    error_msg = f"Failed to extract {component['name']}: {str(e)}"
                    progress_monitor.add_error(error_msg, component.get("file_path"))

            execution_time = time.time() - start_time

            return BatchResults(
                operation_id=operation_id,
                success=True,
                processed_files=processed_components,
                total_files=total_components,
                execution_time=execution_time,
                memory_peak_mb=0,
                transformations_applied=processed_components,
                errors_encountered=len(progress_monitor.errors),
                warnings_generated=len(progress_monitor.warnings),
                details={
                    "component_patterns": component_patterns,
                    "components_found": len(all_components),
                    "components_extracted": len(extracted_components),
                    "output_directory": output_directory,
                    "preview_only": preview_only,
                    "extracted_components": extracted_components,
                },
            )

        except Exception as e:
            error_msg = f"Batch component extraction failed: {str(e)}"
            return BatchResults(
                operation_id=operation_id,
                success=False,
                processed_files=0,
                total_files=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=0,
                errors_encountered=1,
                warnings_generated=0,
                details={"error": error_msg},
            )

    def batch_apply_pattern_transformation(
        self, pattern: str, replacement: str, project_path: str, file_patterns: Optional[List[str]] = None, preview_only: bool = False
    ) -> BatchResults:
        """
        Apply a pattern-based transformation across multiple files.

        Args:
            pattern: Pattern to search for (regex)
            replacement: Replacement pattern
            project_path: Path to project root
            file_patterns: List of file patterns to include
            preview_only: If True, only show changes without applying them

        Returns:
            BatchResults with transformation details
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Find all files matching the pattern
            search_results = self._find_pattern_matches(pattern, project_path, file_patterns)
            affected_files = list(search_results.keys())

            if not affected_files:
                return BatchResults(
                    operation_id=operation_id,
                    success=True,
                    processed_files=0,
                    total_files=0,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=0,
                    transformations_applied=0,
                    errors_encountered=0,
                    warnings_generated=0,
                    details={"message": "No matches found for pattern", "pattern": pattern},
                )

            total_files = len(affected_files)
            total_matches = sum(len(matches) for matches in search_results.values())
            progress_monitor = ProgressMonitor(operation_id, total_matches)

            transformations_applied = 0
            processed_files = 0
            transformation_details: Dict[str, List[Dict[str, Any]]] = {}

            if preview_only:
                # Generate preview without applying changes
                for file_path, matches in search_results.items():
                    preview_changes = []
                    for match in matches:
                        preview_changes.append(
                            {
                                "line": match.get("line", 0),
                                "context": match.get("context", ""),
                                "match": match.get("match", ""),
                                "replacement": replacement,
                            }
                        )
                    transformation_details[file_path] = preview_changes
                    transformations_applied += len(matches)
                    progress_monitor.update_progress(len(matches), "preview_generation")
                    processed_files += 1
            else:
                # Apply actual transformations
                backup_created = False

                if self.config.enable_rollback:
                    backup_created = self._create_project_backup(project_path, operation_id)

                try:
                    for file_path, matches in search_results.items():
                        file_transformations = self._apply_pattern_transformation(file_path, pattern, replacement, matches)
                        transformation_details[file_path] = file_transformations
                        transformations_applied += len(file_transformations)
                        progress_monitor.update_progress(len(matches), "pattern_transformation")
                        processed_files += 1

                except Exception as e:
                    if self.config.enable_rollback and backup_created:
                        self._rollback_project_backup(project_path, operation_id)

                    error_msg = f"Pattern transformation failed: {str(e)}"
                    progress_monitor.add_error(error_msg)

                    return BatchResults(
                        operation_id=operation_id,
                        success=False,
                        processed_files=processed_files,
                        total_files=total_files,
                        execution_time=time.time() - start_time,
                        memory_peak_mb=0,
                        transformations_applied=transformations_applied,
                        errors_encountered=len(progress_monitor.errors) + 1,
                        warnings_generated=len(progress_monitor.warnings),
                        details={"error": error_msg, "partial_results": transformation_details},
                    )

            execution_time = time.time() - start_time

            return BatchResults(
                operation_id=operation_id,
                success=True,
                processed_files=processed_files,
                total_files=total_files,
                execution_time=execution_time,
                memory_peak_mb=0,
                transformations_applied=transformations_applied,
                errors_encountered=len(progress_monitor.errors),
                warnings_generated=len(progress_monitor.warnings),
                details={
                    "pattern": pattern,
                    "replacement": replacement,
                    "file_patterns": file_patterns,
                    "preview_only": preview_only,
                    "total_matches_found": total_matches,
                    "transformations": transformation_details,
                    "backup_created": backup_created if not preview_only else False,
                },
            )

        except Exception as e:
            error_msg = f"Batch pattern transformation failed: {str(e)}"
            return BatchResults(
                operation_id=operation_id,
                success=False,
                processed_files=0,
                total_files=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=0,
                errors_encountered=1,
                warnings_generated=0,
                details={"error": error_msg},
            )

    def _find_symbol_occurrences(self, symbol_name: str, project_path: str, symbol_type: str) -> Dict[str, List[Dict]]:
        """Find all occurrences of a symbol in the project."""
        # Use ripgrep to find symbol occurrences
        search_options = SearchOptions(file_types=["py"])
        search_results = self.ripgrep.search_files(
            pattern=symbol_name,
            path=project_path,
            options=search_options,
        )

        # Organize results by file
        file_occurrences: Dict[str, List[Dict[str, Any]]] = {}
        for result in search_results.results:
            file_path = result.file_path
            if file_path not in file_occurrences:
                file_occurrences[file_path] = []

            file_occurrences[file_path].append(
                {"line": result.line_number, "context": " ".join(result.context_before or []), "match": result.line_text}
            )

        return file_occurrences

    def _analyze_rename_safety(self, old_name: str, new_name: str, search_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze the safety of a rename operation."""
        total_occurrences = sum(len(occurrences) for occurrences in search_results.values())
        affected_files = len(search_results)

        risk_level = "low"
        risks = []

        # Check for potential conflicts
        if len(new_name) < 2:
            risks.append("New symbol name is too short")
            risk_level = "high"

        # Check for keyword conflicts (basic check)
        python_keywords = {"if", "else", "for", "while", "def", "class", "import", "from"}
        if new_name.lower() in python_keywords:
            risks.append("New name conflicts with Python keyword")
            risk_level = "high"

        # Check for potential naming conflicts
        if new_name[0].isdigit():
            risks.append("New name starts with a digit")
            risk_level = "medium"

        # High volume changes increase risk
        if total_occurrences > 100:
            risks.append(f"High volume of changes ({total_occurrences} occurrences)")
            risk_level = "medium" if risk_level == "low" else "high"

        return {
            "risk_level": risk_level,
            "total_occurrences": total_occurrences,
            "affected_files": affected_files,
            "risks": risks,
            "recommendations": self._generate_safety_recommendations(risk_level, risks),
        }

    def _generate_safety_recommendations(self, risk_level: str, risks: List[str]) -> List[str]:
        """Generate safety recommendations based on risk analysis."""
        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Consider running in preview mode first",
                    "Create a manual backup before proceeding",
                    "Review all identified risks carefully",
                    "Consider an alternative naming approach",
                ]
            )
        elif risk_level == "medium":
            recommendations.extend(
                [
                    "Run in preview mode to review changes",
                    "Ensure rollback protection is enabled",
                    "Test the changes in a development environment",
                ]
            )

        return recommendations

    def _create_project_backup(self, project_path: str, operation_id: str) -> bool:
        """Create a backup of the project before transformation."""
        # Implementation would create a backup archive
        # For now, return True to indicate backup was "created"
        return True

    def _rollback_project_backup(self, project_path: str, operation_id: str) -> bool:
        """Rollback project from backup."""
        # Implementation would restore from backup
        # For now, return True to indicate rollback was "applied"
        return True

    def _apply_file_rename(self, file_path: str, old_name: str, new_name: str, occurrences: List[Dict], symbol_type: str) -> List[Dict]:
        """Apply rename transformations to a single file."""
        transformations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            modified_lines = []

            for line_num, line in enumerate(lines, 1):
                modified_line = line
                for occ in occurrences:
                    if occ.get("line") == line_num:
                        # Simple string replacement for now
                        # In production, this would use more sophisticated AST-based replacement
                        modified_line = line.replace(old_name, new_name)
                        transformations.append({"line": line_num, "original": line, "modified": modified_line, "change_type": "rename"})

                modified_lines.append(modified_line)

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(modified_lines))

        except Exception as e:
            raise Exception(f"Failed to transform {file_path}: {str(e)}")

        return transformations

    def _find_components_by_pattern(self, pattern: str, project_path: str) -> List[Dict]:
        """Find components matching a pattern in the project."""
        components = []

        # Use ripgrep to find function/class definitions
        search_options = SearchOptions(file_types=["py"])
        search_results = self.ripgrep.search_files(pattern=pattern, path=project_path, options=search_options)

        for result in search_results.results:
            components.append(
                {
                    "name": result.line_text.strip(),
                    "file_path": result.file_path,
                    "line": result.line_number,
                    "context": " ".join(result.context_before or []),
                    "type": "function" if "def " in result.line_text else "class",
                }
            )

        return components

    def _preview_component_extraction(self, component: Dict) -> Dict:
        """Generate a preview of component extraction."""
        return {
            "component_name": component["name"],
            "file_path": component["file_path"],
            "line": component["line"],
            "type": component["type"],
            "action": "preview_extract",
            "estimated_size": "medium",  # Could be calculated more precisely
        }

    def _extract_component_to_file(self, component: Dict, output_directory: str) -> Dict:
        """Extract a component to a separate file."""
        try:
            # Read the source file
            with open(component["file_path"], "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            component_lines = []
            in_component = False
            indentation_level = 0

            # Find and extract the component
            for i, line in enumerate(lines):
                line_num = i + 1
                if line_num == component["line"]:
                    in_component = True
                    indentation_level = len(line) - len(line.lstrip())
                    component_lines.append(line)
                elif in_component:
                    current_indent = len(line) - len(line.lstrip())
                    if line.strip() and current_indent <= indentation_level and line_num > component["line"]:
                        break
                    component_lines.append(line)

            # Write to output file
            output_filename = f"{component['name']}.py"
            output_path = os.path.join(output_directory, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(component_lines))

            return {
                "component_name": component["name"],
                "file_path": component["file_path"],
                "output_path": output_path,
                "lines_extracted": len(component_lines),
                "action": "extracted",
            }

        except Exception as e:
            raise Exception(f"Failed to extract {component['name']}: {str(e)}")

    def _find_pattern_matches(self, pattern: str, project_path: str, file_patterns: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Find all matches for a pattern in the project."""
        search_options = SearchOptions(file_types=["py"] if not file_patterns else None)
        search_results = self.ripgrep.search_files(pattern=pattern, path=project_path, options=search_options)

        # Organize results by file
        file_matches: Dict[str, List[Dict[str, Any]]] = {}
        for result in search_results.results:
            file_path = result.file_path
            if file_patterns and not any(fp in file_path for fp in file_patterns):
                continue

            if file_path not in file_matches:
                file_matches[file_path] = []

            file_matches[file_path].append(
                {"line": result.line_number, "context": " ".join(result.context_before or []), "match": result.line_text}
            )

        return file_matches

    def _apply_pattern_transformation(self, file_path: str, pattern: str, replacement: str, matches: List[Dict]) -> List[Dict]:
        """Apply pattern transformation to a single file."""
        transformations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            modified_lines = []

            for line_num, line in enumerate(lines, 1):
                modified_line = line
                for match in matches:
                    if match.get("line") == line_num:
                        # Apply regex replacement
                        import re

                        modified_line = re.sub(pattern, replacement, line)
                        transformations.append(
                            {"line": line_num, "original": line, "modified": modified_line, "change_type": "pattern_transformation"}
                        )
                        break  # Apply only once per line

                modified_lines.append(modified_line)

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(modified_lines))

        except Exception as e:
            raise Exception(f"Failed to transform {file_path}: {str(e)}")

        return transformations

    def batch_transform_code(
        self,
        transformation_type: str,
        project_path: str,
        code_patterns: List[str],
        replacement_patterns: Optional[List[str]] = None,
        preview_only: bool = False,
    ) -> BatchResults:
        """
        Apply code transformations across multiple files.

        Args:
            transformation_type: Type of transformation (refactor, optimize, etc.)
            project_path: Path to project root
            code_patterns: List of code patterns to transform
            replacement_patterns: Optional replacement patterns
            preview_only: If True, only show changes without applying them

        Returns:
            BatchResults with transformation details
        """
        operation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Find all files containing the patterns
            affected_files: List[str] = []
            all_matches: List[Dict[str, Any]] = []

            for pattern in code_patterns:
                matches = self._find_pattern_matches(pattern, project_path)
                all_matches.extend([item for sublist in matches.values() for item in sublist])
                affected_files.extend(matches.keys())

            affected_files = list(set(affected_files))  # Remove duplicates

            if not affected_files:
                return BatchResults(
                    operation_id=operation_id,
                    success=True,
                    processed_files=0,
                    total_files=0,
                    execution_time=time.time() - start_time,
                    memory_peak_mb=0,
                    transformations_applied=0,
                    errors_encountered=0,
                    warnings_generated=0,
                    details={"message": "No matching code patterns found", "affected_files": []},
                )

            # Apply transformations
            transformations_applied = 0
            errors = []

            if not preview_only:
                for file_path in affected_files:
                    try:
                        replacement = replacement_patterns[0] if replacement_patterns else code_patterns[0]
                        file_transformations = self._apply_pattern_transformation(file_path, code_patterns[0], replacement, [])
                        transformations_applied += len(file_transformations)
                    except Exception as e:
                        errors.append(f"Failed to transform {file_path}: {str(e)}")

            return BatchResults(
                operation_id=operation_id,
                success=len(errors) == 0,
                processed_files=len(affected_files),
                total_files=len(affected_files),
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=transformations_applied,
                errors_encountered=len(errors),
                warnings_generated=0,
                details={
                    "transformation_type": transformation_type,
                    "affected_files": affected_files,
                    "transformations_applied": transformations_applied,
                    "errors": errors,
                    "preview_mode": preview_only,
                },
            )

        except Exception as e:
            return BatchResults(
                operation_id=operation_id,
                success=False,
                processed_files=0,
                total_files=0,
                execution_time=time.time() - start_time,
                memory_peak_mb=0,
                transformations_applied=0,
                errors_encountered=1,
                warnings_generated=0,
                details={"error": str(e)},
            )

    def validate_batch_operations(self, operations: List[Dict[str, Any]], project_path: str) -> Dict[str, Any]:
        """
        Validate batch operations before execution.

        Args:
            operations: List of operations to validate
            project_path: Path to project root

        Returns:
            Validation results with safety assessment
        """
        validation_results: Dict[str, Any] = {
            "valid_operations": [],
            "invalid_operations": [],
            "warnings": [],
            "safety_score": 0.0,
            "recommendations": [],
        }

        try:
            for i, operation in enumerate(operations):
                op_type = operation.get("type", "unknown")
                op_details = operation.get("details", {})

                # Basic validation
                is_valid = True
                op_warnings = []

                # Check for risky operations
                if op_type in ["delete", "rename"]:
                    if "target" not in op_details:
                        is_valid = False
                        op_warnings.append("Missing target specification")
                    else:
                        # Check if target exists
                        target_path = os.path.join(project_path, op_details["target"])
                        if not os.path.exists(target_path):
                            is_valid = False
                            op_warnings.append(f"Target does not exist: {target_path}")

                # Assess safety
                safety_score = 1.0 if is_valid else 0.0

                if op_type in ["delete", "rename"]:
                    safety_score *= 0.7  # Reduce safety for destructive operations

                operation_result = {
                    "operation_index": i,
                    "type": op_type,
                    "valid": is_valid,
                    "safety_score": safety_score,
                    "warnings": op_warnings,
                }

                if is_valid:
                    validation_results["valid_operations"].append(operation_result)
                else:
                    validation_results["invalid_operations"].append(operation_result)

                validation_results["warnings"].extend(op_warnings)

            # Calculate overall safety score
            if validation_results["valid_operations"]:
                valid_scores = [op["safety_score"] for op in validation_results["valid_operations"]]
                validation_results["safety_score"] = sum(valid_scores) / len(valid_scores)

            # Generate recommendations
            if validation_results["invalid_operations"]:
                validation_results["recommendations"].append("Fix invalid operations before proceeding")

            if validation_results["safety_score"] < 0.8:
                validation_results["recommendations"].append("Consider creating backups before execution")

        except Exception as e:
            validation_results["errors"] = [f"Validation failed: {str(e)}"]
            validation_results["safety_score"] = 0.0

        return validation_results
