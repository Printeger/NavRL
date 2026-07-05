# instinctRL package
#
# Velocity-governor for TASLAB_UAV + Livox MID360.
#
# Modules:
#   audit:            Platform-lock and actor-input contract checks
#   command_adapter:  Body-frame → world-frame velocity transform
#   governor:         Velocity governor (B0: minimal pass-through)
#   observation:      MID360 preprocessing and history buffer (instinctRL-B)
#   mid360_pattern:   Orbit RayCaster adapter for Livox MID360 rays
#   anchor:           Measurement-space anchor manager (instinctRL-C)
#   observability:    Evaluation-only range-Jacobian logger (instinctRL-D)
#   ics:              ICS-inspired command attenuation (instinctRL-E)
#
# Future (deferred):
#   rewards:          instinctRL reward terms (instinctRL-F)
