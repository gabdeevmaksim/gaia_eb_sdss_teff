"""
Pipeline orchestration for eclipsing binary temperature analysis.

This module provides pipeline classes that orchestrate the data processing
and machine learning workflows.
"""

from .base import Pipeline, PipelineStep
from .data_pipeline import DataProcessingPipeline
from .ml_pipeline import MLTrainingPipeline

__all__ = [
    'Pipeline',
    'PipelineStep',
    'DataProcessingPipeline',
    'MLTrainingPipeline'
]
