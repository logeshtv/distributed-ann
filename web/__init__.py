"""Web module for ML Training Dashboard."""

from .app import app
from .training_service import training_service, TrainingProgress

__all__ = ['app', 'training_service', 'TrainingProgress']
