"""
Universal Geometric Obstacle Generator for Drone RL Training
=============================================================
DEPRECATED: This file is a compatibility shim.
The full implementation has moved to: training/envs/universal_generator.py

Modes:
  A: 3D Lattice Forest (Floating discrete obstacles)
  B: 3D Ant Nest (Multi-story maze with ramps)
  C: Restricted Channels (Horizontal/Vertical/Sloped tunnels)
  D: Lethal Sandwich (Cave ceilings + hanging hazards)
  E: Shooting Gallery (Dynamic obstacles)

Author: NavRL Team
"""

# Re-export everything from the new location
from envs.universal_generator import (
    ArenaMode,
    ChannelOrientation,
    SandwichSubType,
    ObstaclePrimitive,
    GapInfo,
    ArenaConfig,
    CurriculumLabels,
    ArenaResult,
    UniversalArenaGenerator,
    ArenaSpawner,
)
import sys
import os

# Add parent to path if needed
_script_dir = os.path.dirname(os.path.abspath(__file__))
_training_root = os.path.dirname(_script_dir)
if _training_root not in sys.path:
    sys.path.insert(0, _training_root)


__all__ = [
    "ArenaMode",
    "ChannelOrientation",
    "SandwichSubType",
    "ObstaclePrimitive",
    "GapInfo",
    "ArenaConfig",
    "CurriculumLabels",
    "ArenaResult",
    "UniversalArenaGenerator",
    "ArenaSpawner",
]
