"""
Improved performance monitoring system.
"""
from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List
import time
import logging
from contextlib import contextmanager
from collections import defaultdict, deque
from functools import wraps


@dataclass
class PerformanceMetrics:
    """Performance metrics for QA system."""
    entity_extraction_time: float = 0.0
    entity_matching_time: float = 0.0
    kg_retrieval_time: float = 0.0
    rag_search_time: float = 0.0
    agent_selection_time: float = 0.0
    llm_response_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with formatted times."""
        return {
            'entity_extraction': f"{self.entity_extraction_time:.3f}s",
            'entity_matching': f"{self.entity_matching_time:.3f}s",
            'kg_retrieval': f"{self.kg_retrieval_time:.3f}s",
            'rag_search': f"{self.rag_search_time:.3f}s",
            'agent_selection': f"{self.agent_selection_time:.3f}s",
            'llm_response': f"{self.llm_response_time:.3f}s",
            'total': f"{self.total_time:.3f}s"
        }
    
    def get_slowest_step(self) -> str:
        """Get the slowest step name."""
        steps = {
            '实体提取': self.entity_extraction_time,
            '实体匹配': self.entity_matching_time,
            'KG检索': self.kg_retrieval_time,
            'RAG搜索': self.rag_search_time,
            '智能体选择': self.agent_selection_time,
            'LLM响应': self.llm_response_time,
        }
        return max(steps, key=steps.get)


class PerformanceMonitor:
    """Performance monitor with time tracking and statistics."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('qa_system_performance')
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.max_history = 100
    
    @contextmanager
    def measure(self, step_name: str):
        """
        Context manager to measure execution time of a code block.
        
        Args:
            step_name: Name of the step to measure
            
        Example:
            with monitor.measure("entity_extraction"):
                entities = extractor.extract(question)
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[step_name].append(elapsed)
            
            # Limit history size
            if len(self.timings[step_name]) > self.max_history:
                self.timings[step_name].pop(0)
            
            # Log warnings if too slow
            if elapsed > 5.0:
                self.logger.warning(f"{step_name} 耗时过长: {elapsed:.3f}s")
            else:
                self.logger.debug(f"{step_name} 耗时: {elapsed:.3f}s")
    
    def timing_decorator(self, step_name: str):
        """
        Decorator to automatically measure function execution time.
        
        Args:
            step_name: Name of the step
            
        Example:
            @monitor.timing_decorator("entity_extraction")
            def extract_entities(question):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(step_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_average_time(self, step_name: str) -> float:
        """
        Get average execution time for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Average execution time in seconds
        """
        times = self.timings.get(step_name, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_statistics(self, step_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary with count, min, max, avg, total times
        """
        times = self.timings.get(step_name, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'min': f"{min(times):.3f}s",
            'max': f"{max(times):.3f}s",
            'avg': f"{sum(times) / len(times):.3f}s",
            'total': f"{sum(times):.3f}s"
        }
    
    def report(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a performance report for all steps.
        
        Returns:
            Dictionary with statistics for all steps
        """
        report = {}
        for step_name in self.timings.keys():
            report[step_name] = self.get_statistics(step_name)
        return report
    
    def reset(self) -> None:
        """Reset all timing history."""
        self.timings.clear()
        self.logger.info("性能监控数据已重置")


class TimingContext:
    """
    Context manager for tracking multiple steps in a single workflow.
    
    Example:
        with TimingContext(monitor) as ctx:
            with ctx.step("entity_extraction"):
                entities = extract(question)
            with ctx.step("entity_matching"):
                matches = match(entities)
        metrics = ctx.get_metrics()
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.metrics = PerformanceMetrics()
        self._start_time = time.time()
    
    def __enter__(self) -> 'TimingContext':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.total_time = time.time() - self._start_time
    
    @contextmanager
    def step(self, step_name: str):
        """
        Context manager for a single step that updates metrics.
        
        Args:
            step_name: Name of the step (should match PerformanceMetrics fields)
        """
        step_start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - step_start
            
            # Update corresponding metric field
            metric_map = {
                'entity_extraction': 'entity_extraction_time',
                'entity_matching': 'entity_matching_time',
                'kg_retrieval': 'kg_retrieval_time',
                'rag_search': 'rag_search_time',
                'agent_selection': 'agent_selection_time',
                'llm_response': 'llm_response_time',
            }
            
            if step_name in metric_map:
                setattr(self.metrics, metric_map[step_name], elapsed)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get collected performance metrics."""
        if self.metrics.total_time == 0.0:
            self.metrics.total_time = time.time() - self._start_time
        return self.metrics
