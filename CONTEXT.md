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

**Observability proxy**
An evaluation-only approximation of range-Jacobian observability computed from MID360 ray geometry and valid-return structure when exact normals or finite-difference range perturbations are not available. It must be labeled as a proxy and must not be used as actor input or deployed safety input.
_Avoid_: exact Jacobian, deployed observability feature

**Anchor reward (measurement-space regulation)**
A reward term active only under null command: `r_anchor = -w_anchor · ||r_t - r_star||²`, where `r_star` is the range pattern frozen at the moment `v_cmd` became zero. This rewards "holding the same view" rather than "holding the same position," which is what station-keeping physically means without a position estimate. Strengthens drift suppression in the observable directions.

**Reward integration**
The Paper-1 stage that routes existing measurement-space signals into reward and logging while preserving the actor input contract. Reward integration is not the same as training convergence: it proves the reward terms are available, bounded, logged, and actor-clean, but it does not prove that a learned policy has converged.

**Command-consistency tracking proxy**
The default Paper-1 tracking reward signal used before the trainable governor path is complete. It compares the commanded body-frame velocity `v_cmd` with the final or issued body-frame command proxy `v_final`, instead of using privileged actual velocity. Actual velocity may be used only in explicitly reward-only evaluation/training variants.

**Intervention penalty**
A reward term that penalizes reliance on the ICS attenuation layer, typically through low `beta`. Its purpose is to discourage policies that repeatedly request unsafe commands and depend on the safety layer to fix them.

**Training readiness**
A narrower claim than training success. A system is training-ready when the observation, action, reward, logging, and audit paths can run without actor leakage and with test-covered semantics. It does not imply stable learning curves, convergence, or deployable policy performance.

**Long-run training stability**
The numerical property that a PPO run can continue for the declared acceptance horizon without producing non-finite actions, distribution parameters, losses, gradients, or model parameters. It is stricter than a short training smoke and weaker than convergence: a run can be numerically stable without learning a good policy.

**Learned governor**
The actor-clean policy component that transforms a commanded body-frame velocity into a governed body-frame velocity by choosing a command-preservation factor and a bounded corrective velocity. It is distinct from the controller action: the learned governor operates in body-frame command space, while the controller boundary converts the final command into the action consumed by the velocity controller.

**Command-governor task**
The active instinctRL training/evaluation task. Success is measured by actual body-frame command tracking, null-command station-keeping, clearance, collision, ICS intervention, and observability metrics; it is not measured by reaching a sampled NavRL target position.
_Avoid_: goal-navigation success, reach-goal success

**Actual-velocity tracking reward**
The default formal-training tracking reward for command-governor runs. It compares privileged simulator actual body-frame velocity to `v_cmd_b` as reward-only/critic-only information, while the actor still receives only `lidar_grid + state_vec`.
_Avoid_: command-proxy tracking as primary reward

**Legacy reach-goal diagnostic**
The old NavRL target-position arrival signal. It may remain in logs as `legacy_reach_goal` for debugging old task shells, but it is not an instinctRL success metric.
_Avoid_: success rate

**Command curriculum**
A staged source of body-frame velocity commands that starts with normal/recovery hover-heavy commands and gradually increases aggressive, oscillatory, and adversarial command modes. The actor receives only the sampled `v_cmd_b`, not generator internals.
_Avoid_: crazy command from frame zero

**Station-first command curriculum**
A command curriculum profile where recovery/null-command samples dominate early training until station-keeping is stable. It is the default formal-training profile after the 1M short diagnostic showed null-command drift.
_Avoid_: mixed command curriculum as the first convergence gate

**Null-command bias**
The learned-governor failure mode where `v_cmd = 0` and the policy outputs a persistent nonzero command without an active measurement-space anchor need. It is an output bias, not a correction. It remains an objective/curriculum failure unless observability metrics show the scene is degenerate.
_Avoid_: calling it an ICS failure, calling it tracking success

**Station correction**
A small learned-governor correction allowed under `v_cmd = 0` when the measurement-space anchor is active and anchor loss is high. A station correction is measurement-anchored: it should reduce range-anchor error and actual station drift, while `null_command_speed` still limits real vehicle motion. It is distinct from null-command bias.
_Avoid_: hard-zeroing all null-command corrections, treating any nonzero null output as bias

**Command amplification**
The learned-governor failure mode where `||v_final_b|| > ||v_cmd_b||` under a safe nonzero command. It means the policy is not merely preserving and correcting the command; it is increasing command magnitude.
_Avoid_: command preservation

**Preservation band**
The accepted magnitude-preservation interval for safe nonzero command tracking. Current A2-R3 defaults require `0.75 <= ||v_final_b|| / ||v_cmd_b|| <= 1.05` when ICS is not attenuating. Below-band slowdown and above-band amplification are both objective failures; ICS intervention is allowed to reduce speed for safety.
_Avoid_: rewarding slowdown as safety when ICS did not intervene

**Soft null-command correction prior**
A decoder-level prior that attenuates learned `v_corr` when `||v_cmd||` is near zero, but retains a small correction floor for measurement-anchored station keeping. Current A2-R3 defaults use `null_vcorr_gate_min=0.25` and `null_vcorr_gate_eps=0.25`. The reward still punishes actual null-command motion and output bias when the anchor is inactive or low-loss.
_Avoid_: relying only on reward to discover zero-output behavior, hard-zeroing observable station correction

**Hard diagnostic gate**
A pass/fail rule set over short diagnostic eval JSON, not over training reward alone. The current gate checks station drift, null-command actual speed, station anchor error, tracking RMSE, preservation band, amplification, clearance p05, collision, ICS violation, and termination reasons. `null_command_output_speed` is diagnostic-only in A2-R3 because a bounded anchor-aware station correction may legitimately output a small command. Collision rate alone is not a safety pass.
_Avoid_: calling a checkpoint ready because `eval/stats.collision == 0`

**Corrective sweep**
A small automated 128k/256k experiment set used to rank objective/config candidates before any 1M/2M or formal run. The sweep must be evaluated by hard diagnostic gates and should default to dry-run command review before execution.
_Avoid_: manual one-off 1M/2M trial-and-error

**Short diagnostic eval suite**
A two-pass handbook diagnostic evaluation: a zero-command station-keeping pass and a command-curriculum tracking pass, both under static MID360-visible geometry. It is stronger than a single rollout smoke and weaker than the full B0-B8 paper evaluation matrix.
_Avoid_: paper-level benchmark, dynamic-obstacle robustness claim

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
