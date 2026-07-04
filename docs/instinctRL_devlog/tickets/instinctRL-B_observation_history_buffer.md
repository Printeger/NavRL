# instinctRL-B: Observation / History Buffer

> **Ticket ID**: instinctRL-B  
> **Status**: ✅ Complete  
> **Date**: 2026-07-04  
> **Dependencies**: instinctRL-A  
> **Blocks**: instinctRL-C, instinctRL-E, instinctRL-F  
> **Risk**: High  
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` §Observation and History Buffer, §Tickets  
> **Resolves**: D-002 (Full MID360 preprocessing)

---

## Goal

Build the complete actor-clean observation pipeline: MID360 raw range $r_t$, valid-return mask $m_t$, staleness-weighted reliability $w_t$, timestamps, allowed IMU cues (body angular velocity + gravity direction), body-frame $v_{cmd}$, previous governor action, and fixed-size history buffer. Wire into `NavigationEnv` and update PPO feature extractor for hybrid input format.

---

## Files Modified

| File | Change Summary | Lines |
|------|---------------|:-----:|
| `cfg/train.yaml` | Added `instinctRL.observation.*` config (history_len, enable_noise, enable_dropout, tau_staleness) | +6 |
| `scripts/env.py` | Replaced danger-coded LiDAR with MID360ObservationBuilder; hybrid obs spec; v_cmd generation restored | +45 / -30 |
| `scripts/ppo.py` | Multi-channel CNN + state vector encoder + CatTensors merge for hybrid observation | +15 / -10 |
| `scripts/instinctRL/__init__.py` | Moved observation from deferred to active | ~1 |

## New Files Created

| File | Purpose | Lines |
|------|---------|:-----:|
| `scripts/instinctRL/observation.py` | `MID360ObservationBuilder` + `ObservationConfig`: raw range, mask, staleness-weighted reliability, IMU cues, history buffer | ~220 |

---

## Main Changes

### 1. Observation Pipeline (`observation.py`)

- **Raw range**: $r_i = \|p^{hit}_i - p^{lidar}\|$, true distance (NOT danger-coded)
- **Valid-return mask**: $m_t = [\text{finite} \land r > 0.01 \land r < r_{max}]$
- **Reliability weights**: $w_t = m_t \cdot \exp(-\text{age} / \tau)$, fallback to binary $w_t=m_t$ when age unavailable
- **IMU cues**: body angular velocity (3) + gravity direction in body frame (3) — no position/velocity
- **v_cmd**: body-frame velocity command from simple random generator (deferred adversarial)
- **Previous action**: stored from last governor output
- **History buffer**: rolling window ($L=4$ default), interleaved grid channels + flat state vector

### 2. Observation Spec (Hybrid Format)

```
"lidar_grid": [N, L*3, H, V]   — L frames × 3 channels (range, mask, weight)
"state_vec":  [N, L*13]        — L frames × 13 dims (IMU6 + v_cmd3 + prev3 + age1)
```

### 3. Feature Extractor

- CNN: `LazyConv2d` auto-adapts to multi-channel input
- State encoder: `Linear→ELU→LayerNorm` (64-dim)
- Merge: `CatTensors` → `MLP[192, 256]` → actor feature

---

## Method Consistency Checklist

| Check | Status |
|-------|:------:|
| No pose in actor obs | ✅ |
| No odometry in actor obs | ✅ |
| No explicit velocity in actor obs | ✅ |
| No map/SLAM in actor obs | ✅ |
| No privileged simulator state in actor obs | ✅ |
| IMU cues: body ang_vel + gravity only (no linear vel) | ✅ |
| v_cmd in body frame | ✅ |
| History buffer over allowed fields only | ✅ |
| MID360 pattern: 360°×59° FOV | ✅ |

---

## Tests Run (Code-Level)

| Test | Result |
|------|:------:|
| ObservationBuilder import | ✅ |
| Config loading (history_len=4, tau=0.5) | ✅ |
| Observation spec shape correctness | ✅ |
| PPO feature extractor input keys match | ✅ |

## Known Issues

1. **B0 smoke test**: Blocked by Isaac Sim PhysX fabric issue (pre-existing, not caused by B changes)
2. **Noise/dropout**: Deferred (D-009), config switches present but OFF by default
3. **Neighbor-consistency weights**: Deferred (D-010)
4. **Longer history ablations**: Deferred (D-011)

---

## Next Recommended Step

Proceed to **instinctRL-C**: Measurement-Space Anchor Manager.
