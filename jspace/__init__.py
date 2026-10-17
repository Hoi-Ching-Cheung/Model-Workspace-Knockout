"""Does chain of thought protect a model from losing its internal workspace?

Experiment code for ablating Qwen3-4B's Jacobian-lens "J-space" on math
problems of increasing difficulty, with and without chain of thought.
"""

from .ablation import Ablator, concept_scores, effective_unembed, remove_jspace
from .generation import LoopStop, build_prompt, clean_exclude_ids, generate
from .grading import extract_answer, grade
from .runner import RunConfig, run_grid

__all__ = [
    "Ablator", "LoopStop", "RunConfig", "build_prompt", "clean_exclude_ids",
    "concept_scores", "effective_unembed", "extract_answer", "generate", "grade",
    "remove_jspace", "run_grid",
]
