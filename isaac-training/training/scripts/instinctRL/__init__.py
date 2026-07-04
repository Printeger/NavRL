# instinctRL package
#
# Velocity-governor for TASLAB_UAV + Livox MID360.
#
# Modules:
#   audit:            Platform-lock and actor-input contract checks
#   command_adapter:  Body-frame → world-frame velocity transform
#   governor:         Velocity governor (B0: minimal pass-through)
#
# Future (deferred):
#   observation:      History buffer and MID360 preprocessing (instinctRL-B)
#   anchor:           Measurement-space anchor manager (instinctRL-C)
#   ics:              ICS-inspired command attenuation (instinctRL-E)
#   observability:    Range-Jacobian logger (instinctRL-D)
#   rewards:          instinctRL reward terms (instinctRL-F)
