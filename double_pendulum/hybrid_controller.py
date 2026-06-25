"""
Hybrid swing-up + catch controller for the double pendulum.

The diffusion policy swings the pendulum up well, but it runs at the 20 Hz
control rate, and the fully-inverted equilibrium [pi,0] is NOT stabilizable at
20 Hz on this light, weakly-actuated plant: within one 0.05 s step the pendulum
falls past the actuator's catchable basin, so the policy limit-cycles around the
top instead of holding (see rollout_tvlqr.lqr_gain / the dataset generator,
which only made the dataset's holds succeed by re-closing the LQR loop at the
integration rate dt_sim).

This controller keeps the diffusion policy for the swing-up and, once the state
first arrives near the goal with a catchable velocity, latches into an
infinite-horizon LQR "catch" that is re-closed at the integration rate dt_sim
(~500 Hz here). That fast inner loop is what actually holds the unstable
equilibrium. Verified: the same gain catches [pi,0] in the MuJoCo plant from
0.3 rad / ~4 rad/s arrivals.

Interface mirrors DiffusionController (`reset`, `action`) and adds
`substep_action(x)`, which the physics loop must call once per dt_sim sub-step so
the hold is regulated at the integration rate rather than the slow control rate:

    u = ctrl.action(x)                  # once per control step (20 Hz)
    env.data.ctrl[:] = u
    for _ in range(n_sub):
        ... apply disturbances ...
        if hasattr(ctrl, "substep_action"):
            x_sub = read_state()
            env.data.ctrl[:] = clip(ctrl.substep_action(x_sub))
        mujoco.mj_step(...)

For a plain DiffusionController (no `substep_action`) the loop is unchanged.
"""

import numpy as np

from rollout_tvlqr import lqr_gain, equilibrium_torque, wrap_to_pi, P

# Catch-gain weights. These match the [pi,0,0,0] swing-up references
# (Q_100_100_100_100__R_0p01_0p01__xgoal_3p142_0_0_0): heavy velocity penalty
# arrests an oscillating arrival, low control penalty for an aggressive catch.
DEFAULT_Q = (100.0, 100.0, 100.0, 100.0)
DEFAULT_R = (0.01, 0.01)


class LQRHoldController:
    """Infinite-horizon LQR regulator about an inverted equilibrium.

    The gain is solved at `dt` (pass the INTEGRATION step dt_sim, not the control
    step) because the inverted hold is only stabilizable at the fast rate. The
    feed-forward `u_eq` holds the goal fixed (~0 at the cardinal equilibria, but
    computed exactly so off-cardinal / mirrored goals also work).
    """

    def __init__(self, goal, dt, Q=DEFAULT_Q, R=DEFAULT_R, torque_limit=None):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.K = lqr_gain(self.goal, list(Q), list(R), dt=dt)
        self.u_eq = equilibrium_torque(self.goal)
        self.tau = P.torque_limit if torque_limit is None else float(torque_limit)

    def action(self, x):
        e = np.asarray(x, dtype=np.float64) - self.goal
        e[0], e[1] = wrap_to_pi(e[0]), wrap_to_pi(e[1])
        return np.clip(self.u_eq - self.K @ e, -self.tau, self.tau)


class HybridController:
    """Diffusion swing-up, then a latched LQR catch re-closed at the integration rate.

    Parameters
    ----------
    diff_ctrl   : a DiffusionController-like object (reset / action).
    dt_sim      : the integration step the catch loop is re-closed at (e.g.
                  env.model.opt.timestep). The LQR gain is designed for this dt.
    goal        : target equilibrium [q1,q2,dq1,dq2]; defaults to [pi,0,0,0].
    engage_ang  : summed |q1|+|q2| angle error (rad) below which the catch may
                  engage. Inside the verified catchable basin (~0.3 holds).
    engage_vel  : summed |dq1|+|dq2| velocity (rad/s) below which the catch may
                  engage -- above this the arrival is too fast to catch, so we
                  keep pumping with the policy instead of latching into a miss.
    """

    def __init__(self, diff_ctrl, dt_sim, goal=(np.pi, 0.0, 0.0, 0.0),
                 engage_ang=0.5, engage_vel=6.0, Q=DEFAULT_Q, R=DEFAULT_R,
                 torque_limit=None):
        self.diff = diff_ctrl
        self.goal = np.asarray(goal, dtype=np.float64)
        self.hold = LQRHoldController(self.goal, dt=dt_sim, Q=Q, R=R,
                                      torque_limit=torque_limit)
        self.engage_ang = float(engage_ang)
        self.engage_vel = float(engage_vel)
        self.holding = False
        self._u = np.zeros(self.hold.u_eq.shape[0], dtype=np.float64)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset(self, x0):
        self.diff.reset(x0)
        self.holding = False
        self._u = np.zeros_like(self._u)

    def _catchable(self, x):
        ang = abs(wrap_to_pi(x[0] - self.goal[0])) + abs(wrap_to_pi(x[1] - self.goal[1]))
        vel = abs(x[2]) + abs(x[3])
        return ang < self.engage_ang and vel < self.engage_vel

    # ── control-rate call (once per 20 Hz step) ──────────────────────────────
    def action(self, x):
        """Decide the mode and return this step's command. While swinging up this
        delegates to the diffusion policy (which advances its own history /
        replan); once the state first arrives catchable near the goal it latches
        into the LQR catch for the rest of the episode."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if not self.holding and self._catchable(x):
            self.holding = True
        if self.holding:
            self._u = self.hold.action(x)
        else:
            self._u = np.asarray(self.diff.action(x), dtype=np.float64).ravel()
        return self._u

    # ── integration-rate call (once per dt_sim sub-step) ─────────────────────
    def substep_action(self, x):
        """Re-close the loop at the integration rate. In hold mode this recomputes
        the stabilizing LQR torque from the current sub-step state (the whole
        point -- a torque held for a full 20 Hz step cannot balance this plant).
        During swing-up it returns the policy's command unchanged (zero-order
        hold), so behaviour is identical to a plain DiffusionController.

        Note: in the noise-robustness sweep the sub-step state is the CLEAN plant
        state, so the catch sees un-noised feedback at dt_sim -- the hold's
        noise numbers are therefore optimistic relative to the 20 Hz policy."""
        if self.holding:
            self._u = self.hold.action(np.asarray(x, dtype=np.float64).ravel())
        return self._u
