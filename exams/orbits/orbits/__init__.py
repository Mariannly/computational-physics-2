"""
Two-body simulation package for classical and relativistic orbital motion around a black hole.

This package provides tools to simulate and visualize the motion of a planet
orbiting a black hole using both classical Newtonian mechanics and relativistic corrections.

Classes:
    TwoBodySystem: Handles the physical parameters and initial conditions
    SimulationRunner: Manages the numerical integration of the orbital motion
    OrbitalAnimator: Creates animations of the orbital motion

Functions:
    classical_slope: Computes the classical gravitational acceleration
    relativistic_slope: Computes the relativistic gravitational acceleration
"""

from .orbits import (
    TwoBodySystem,
    SimulationRunner,
    OrbitalAnimator,
    classical_slope,
    relativistic_slope,
)

# Constants
from astropy.constants import c, G

__version__ = '0.1'
__author__ = 'Mariannly Marquez'
__email__ = 'mariannly.marquez@yachaytech.edu.ec'

__all__ = [
    'TwoBodySystem',
    'SimulationRunner',
    'OrbitalAnimator',
    'classical_slope',
    'relativistic_slope',
    'c',
    'G'
]