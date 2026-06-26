"""
src.ml
======
The ML pipeline, split into focused modules:

    dtype_utils   — normalize pandas extension/nullable dtypes (bulletproofing)
    problem_type  — classification vs regression detection
    models        — the model registry per problem type
    preprocessing — feature-plan application, encoding, scaling, train/test split
    training      — fit + evaluate every model, compute metrics
    persistence   — save models (GridFS) + metrics (MongoDB)
    pipeline      — MLPipeline orchestrator that ties it all together

The LLM agents (feature engineering, model selection) live in `src.agents`.
"""

from src.ml.pipeline import MLPipeline

__all__ = ["MLPipeline"]
