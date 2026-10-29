"""SolveTide: checked sparse solves and discrete implicit sensitivities."""
from .assembly import System, assemble
from .runtime import GradientResult, Runtime, SolveResult, SolverConfig

__version__ = "0.1.0"
__all__ = ["System", "assemble", "Runtime", "SolverConfig", "SolveResult", "GradientResult"]
