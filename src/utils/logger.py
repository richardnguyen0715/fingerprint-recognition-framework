"""
Logging utilities for fingerprint recognition framework.

Provides structured logging with support for experiment tracking,
metrics logging, and result persistence.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json


class ExperimentLogger:
    """
    Logger for tracking experiments and results.
    
    Combines standard Python logging with experiment-specific
    metric tracking and result persistence.
    
    Attributes:
        name: Logger name (typically experiment name)
        log_dir: Directory for log files
        logger: Python logger instance
        metrics: Dictionary storing experiment metrics
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Union[str, Path] = "logs",
        level: str = "INFO",
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the experiment logger.
        
        Args:
            name: Name of the experiment/logger
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if file_output:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            entry['step'] = step
        
        self.metrics[name].append(entry)
        self.info(f"Metric {name}: {value:.6f}" + (f" (step {step})" if step else ""))
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameter names and values
        """
        self.metrics['params'] = params
        self.info(f"Parameters: {json.dumps(params, indent=2)}")
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log final experiment results.
        
        Args:
            results: Dictionary of result names and values
        """
        self.metrics['results'] = results
        self.info("Final Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
            else:
                self.info(f"  {key}: {value}")
    
    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """
        Save all logged metrics to a JSON file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to the saved metrics file
        """
        if filename is None:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{timestamp}_metrics.json"
        
        filepath = self.log_dir / filename
        
        # Add metadata
        output = {
            'experiment_name': self.name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.info(f"Metrics saved to {filepath}")
        return filepath


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO"
) -> ExperimentLogger:
    """
    Get or create an experiment logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(name, log_dir, level)


class ProgressTracker:
    """
    Track progress of long-running operations.
    
    Provides timing estimates and progress reporting.
    """
    
    def __init__(self, total: int, logger: Optional[ExperimentLogger] = None):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            logger: Optional logger for output
        """
        self.total = total
        self.current = 0
        self.start_time = datetime.now()
        self.logger = logger
    
    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        
        if self.logger and self.current % max(1, self.total // 20) == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"Progress: {self.current}/{self.total} "
                f"({100*self.current/self.total:.1f}%) "
                f"ETA: {eta:.1f}s"
            )
    
    def finish(self) -> float:
        """
        Mark operation as complete.
        
        Returns:
            Total elapsed time in seconds
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.logger:
            self.logger.info(f"Completed {self.total} items in {elapsed:.2f}s")
        return elapsed
