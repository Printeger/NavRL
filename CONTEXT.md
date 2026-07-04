# NavRL INSTINCT Domain Model

> ⚠️ **ACTIVE METHOD LOCK (instinctRL-0, 2026-07-04)**  
> The active Paper-1 implementation route is:
> **Velocity-governor with body-frame velocity commands** (`\vgov = \alpha_t\vcmd + \vcorr`)
> executed through `VelController(LeePositionController)` on TASLAB_UAV platform
> with Livox MID360 sensor.  
> Authoritative documents:  
> - `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex`  
> - `docs/paper1_vel_ctrl.tex`  
>  
> **Legacy / Future:** The terms below marked with ⚠️ describe the original
> INSTINCT vision (CTBR, CMDP/PPO-Lagrangian, GRU-required actor, full CBF-QP).
> These are **NOT active Paper-1 implementation requirements**. They remain
> documented as potential Paper-2 / future directions only.

---

This glossary defines the canonical terms for the INSTINCT system: an end-to-end quadrotor policy that performs station-keeping and safe velocity tracking from raw measurements without explicit state estimation.

---

## Core Concepts

**CTBR (Collective Thrust + Body Rates)** ⚠️ LEGACY
> The low-level control interface: a 4-dimensional action consisting of normalized collective thrust `c ∈ [0,1]` and desired body angular rates `ω_des ∈ ℝ³`. This is the most sim-to-real-robust quadrotor control interface because an onboard high-rate PID controller maps body-rate commands to motor thrust, absorbing motor dynamics and much of the sim-to-real gap. Contrast with velocity commands (current NavRL) or position waypoints (classical control).

*Note: Paper-1 uses velocity-controller-based actions, not CTBR. CTBR is a future/Paper-2 direction.*

**Station-keeping**
The task of holding position and yaw without drift, using only raw measurements (no global position estimate). Formalized as regulation of the range measurement vector `r_t` toward a frozen reference `r_star`, under null velocity command `v_cmd = 0`. This is measurement-space servoing, not pose regulation. Station-keeping is the special case of safe tracking with zero command.

**Safe tracking**
Following a non-zero velocity command `v_cmd ≠ 0` while guaranteeing collision avoidance through a hard safety constraint. The policy must track the command velocity while respecting occupancy constraints defined in measurement space.

**ICS (Inevitable Collision State)**
A state from which collision is unavoidable under any admissible control sequence, even if current clearance is positive. Formally: state `x` is an ICS if for all policies `π`, there exists future time `τ` where the vehicle enters the collision set. A vehicle moving too fast toward an obstacle may be in an ICS even when `d > d_safe`, because it cannot brake in time.

**v_safe(d) — Braking distance bound**
The maximum safe speed toward the nearest obstacle as a function of clearance `d`: `v_safe(d) = √(2·a_max·max(d - d_safe, 0))`. Derived from kinematic braking physics. If `v_parallel < v_safe(d)`, then the vehicle is NOT in an ICS w.r.t. that obstacle and a collision-avoiding control exists. This converts "never crash" into a verifiable condition tied to sensor range: `v_safe(r_max) = √(2·a_max·(r_max - d_safe))` bounds cruising speed.

**CMDP (Constrained Markov Decision Process)** ⚠️ LEGACY
> A reinforcement learning formulation with both a task objective (reward `r_t`) and a hard safety constraint (cost `c_t`): maximize cumulative reward subject to cumulative cost ≤ δ. Solved via Lagrangian relaxation with dual variable `λ` that grows when the safety budget is violated, encoding "safety dominates tracking."

*Note: Paper-1 uses ICS-inspired command attenuation, not a formal CMDP Lagrangian.*

---

## Observability

**Range Jacobian J(p)**
The sensitivity of LiDAR range measurements to translational motion: `δr ≈ -J(p)·δp`, where `J` is an `N×3` matrix with rows `-n_i^T` (surface normals in world frame). The rank of `J` determines which translational directions are observable from range measurements. Full rank (rank=3 in 3D, rank=2 in 2D) means position is locally observable; rank-deficient `J` means some motion directions produce no range change.

**Aperture degeneracy (observability failure)**
Geometric configurations where `rank(J) < 3`, meaning some translational directions lie in `ker(J)` and are unobservable from ranges. Examples:
- Straight tunnel: motion along the tunnel axis is unobservable (rank=2)
- Flat wall: motion parallel to the wall is unobservable
- Open field: all rays return `r_max`, so `J=0` and no direction is observable (full degeneracy)

This is the same phenomenon that defeats LiDAR odometry in feature-poor environments. It places a fundamental limit on drift-free station-keeping: the policy cannot correct drift it cannot observe. Performance must be reported stratified by `rank(J)` or `σ_min(J)` (smallest singular value, a continuous measure of observability).

**Anchor reward (measurement-space regulation)**
A reward term active only under null command: `r_anchor = -w_anchor · ||r_t - r_star||²`, where `r_star` is the range pattern frozen at the moment `v_cmd` became zero. This rewards "holding the same view" rather than "holding the same position," which is what station-keeping physically means without a position estimate. Strengthens drift suppression in the observable directions.

---

## Architecture & Training

**Asymmetric actor-critic**
A training architecture where the actor (deployed policy) observes only raw measurements `o_t = (r_t, IMU, v_cmd, a_prev)`, while the critics (reward-value and cost-value functions) observe privileged simulator state `(p, q, v, ω, obstacle map)` including ground-truth velocity, clearance, and ICS labels. This stabilizes value learning under partial observability without leaking privileged information into the deployed policy. At deployment, only the actor is retained.

**PPO-Lagrangian** ⚠️ LEGACY
> PPO (Proximal Policy Optimization) extended to solve the CMDP via Lagrangian duality. The Lagrange multiplier `λ ≥ 0` is adapted online:
> - Actor maximizes combined advantage: `A_total = A_reward - λ·A_cost`
> - Multiplier update: `λ ← [λ + η_λ·(J_cost - δ)]₊` (grows when cost budget is violated)
> - Two critics: `V_r` (reward value) and `V_c` (cost value)
>
> This automates the safety-performance tradeoff rather than hand-tuning a cost penalty weight.

*Note: Paper-1 uses single-critic PPO with asymmetric actor-critic architecture, not PPO-Lagrangian.*

**Recurrent actor (GRU)** ⚠️ LEGACY
> The actor must be recurrent (uses a GRU hidden state) because instantaneous velocity `v_t` is NOT observable from a single measurement `o_t`. Velocity must be inferred from the temporal evolution of range measurements `r_{0:t}` and integration of IMU accelerations. The GRU serves as the vehicle's implicit velocity estimator. Critics can be feedforward because they see privileged `v_t` directly.

*Note: Paper-1 allows feedforward actor with history buffer. GRU is an ablation option, not a requirement.*

---

## Implementation Terminology

**Privileged state**
Simulator-only information (global position, velocity, exact clearance) available for reward computation and critic training but never seen by the actor. Denoted `x_t = (p_t, q_t, v_t, ω_t) ∈ ℝ¹³`. Contrast with **observation** `o_t`, the actor's actual input.

**Cost function c_t**
The per-timestep safety violation signal used in the CMDP constraint. Consists of:
- Margin violation: `𝟙[d_t < d_safe]` (enters safety buffer)
- ICS-aware velocity penalty: `κ·[v_parallel - v_safe(d)]₊` (approaching faster than braking allows)
- Hard collision: `C_coll·𝟙[contact]` (actual impact)

Computed from privileged state during training; the actor learns to avoid it using only `o_t`.

**Training curriculum stages** ⚠️ PARTIALLY UPDATED (v1.1)
> The 5-stage progression that introduces complexity incrementally:
> - **Stage A0**: MLP actor, privileged velocity in obs, empty space, no constraint (λ=0). Verifies basic control.
> - **Stage A1**: Remove privileged velocity, add GRU. Learns velocity inference from measurement history.
> - **Stage B**: Station-keeping under aperture degeneracy (corners/walls/tunnels/open). Quantifies observability limits.
> - **Stage C**: Safe tracking in clutter, turn on Lagrangian constraint, ramp cost budget δ→0. Learns ICS avoidance.
> - **Stage D**: Domain randomization (mass/inertia/latency/noise). Sim-to-real hardening, no new behavior.
>
> Each stage initializes from the previous checkpoint.

*Note: The v1.1 handbook defines instinctRL tickets (instinctRL-0 through instinctRL-H) with specific scope. CTBR actions, CMDP constraints, and required recurrent actors are NOT in the Paper-1 route. See `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex` for the active curriculum.*

---

## Distinctions from Current NavRL

Current NavRL (RA-L 2025 paper):
- Action: velocity commands `[vx, vy, vz]` → LeePositionController → motors
- Observation: includes privileged velocity in goal frame
- Actor: feedforward MLP (no recurrence)
- Safety: reward shaping only (log-distance penalty)
- Training: single-stage, no curriculum

INSTINCT target (Paper-2 / future):
- Action: CTBR `(c, ω_des)` directly to rate controller (NOT Paper-1)
- Observation: raw LiDAR + IMU only (no velocity, no position)
- Actor: recurrent GRU (infers velocity from history) — ablation only for Paper-1
- Safety: CMDP with Lagrangian dual and cost critic (NOT Paper-1)
- Training: 5-stage curriculum with observability analysis

instinctRL active (Paper-1, v1.1 platform-locked):
- Action: body-frame velocity governor `\vgov = \alpha_t\vcmd + \vcorr` → VelController(LeePositionController)
- Actor input: MID360 range $r_t$, masks $m_t$, weights $w_t$, IMU cues, $\vcmd$, history $h_t$
- Actor output: $(\alpha_t, \vcorr)$ → deterministic $\vgov$
- Critic: asymmetric — privileged state only through critic-only branch
- Safety: ICS-inspired command attenuation $\beta_t$ (not formal CBF/CMDP)
- Platform: TASLAB_UAV + Livox MID360 (locked)
