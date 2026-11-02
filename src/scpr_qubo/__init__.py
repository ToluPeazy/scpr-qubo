"""Exact SCPR -> QUBO encoding with Beasley pre-reduction and post-Q variable fixing."""

from .beasley import beasley
from .instance import SCPRInstance
from .oracles import brute_force_optimum
from .qubo import QUBO, build_qubo, fix_variables, solve_qubo_exact

__all__ = ["SCPRInstance", "beasley", "brute_force_optimum", "QUBO",
           "build_qubo", "fix_variables", "solve_qubo_exact"]
__version__ = "0.1.0"
