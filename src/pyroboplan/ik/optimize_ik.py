import numpy as np
import pinocchio
from scipy.optimize import minimize
import time

from ..core.utils import (
    check_collisions_at_state,
    check_within_limits,
    get_random_state,
    get_random_collision_free_state,
)
from ..visualization.meshcat_utils import visualize_frame
from .differential_ik import DifferentialIkOptions
import vamp
from data_generation.helpers.metrics import (
    path_efficiency,
    max_deviation_along_line,
    compute_manipulability,
)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

Q_BUFFER = 0.1

Q_LIMS = np.array(
    [
        [
            -2.8973 + Q_BUFFER,
            -1.7628 + Q_BUFFER,
            -2.8973 + Q_BUFFER,
            -3.0718 + Q_BUFFER,
            -2.8973 + Q_BUFFER,
            -0.0175 + 10 * Q_BUFFER,
            -2.8973 + Q_BUFFER,
        ],
        [
            2.8973 - Q_BUFFER,
            1.7628 - Q_BUFFER,
            2.8973 - Q_BUFFER,
            -0.0698 - Q_BUFFER,
            2.8973 - Q_BUFFER,
            3.7525 - 10 * Q_BUFFER,
            2.8973 - Q_BUFFER,
        ],
    ]
)


class OptimizationIk:
    def __init__(
        self,
        model,
        collision_model=None,
        data=None,
        collision_data=None,
        visualizer=None,
        options=DifferentialIkOptions(),
    ):
        self.model = model
        self.collision_model = collision_model
        self.data = data or model.createData()
        self.collision_data = collision_data or (
            collision_model.createData() if collision_model else None
        )
        self.visualizer = visualizer
        self.options = options

    def _joint_limit_penalty(self, q):
        q = np.array(q[:7])  # Only penalize the arm, not the full model
        lower_viol = np.maximum(Q_LIMS[0] - q, 0)
        upper_viol = np.maximum(q - Q_LIMS[1], 0)
        return np.sum(lower_viol**2 + upper_viol**2)

    def solve(
        self,
        target_frame,
        target_tform,
        init_state=None,
        nullspace_components=[],
        verbose=False,
    ):
        np.random.seed(self.options.rng_seed)
        target_frame_id = self.model.getFrameId(target_frame)

        active_joint_indices = [
            idx
            for idx in range(self.model.nq)
            if idx not in self.options.ignore_joint_indices
        ]

        if init_state is None:
            init_state = get_random_state(self.model)

        def objective(q_full):
            pinocchio.framesForwardKinematics(self.model, self.data, q_full)
            cur_tform = self.data.oMf[target_frame_id]
            error = -pinocchio.log(target_tform.actInv(cur_tform)).vector

            nullspace_term = np.zeros_like(q_full)
            for comp in nullspace_components:
                nullspace_term += comp(self.model, q_full)

            # Apply weights if provided
            limit_penalty = self._joint_limit_penalty(q_full)

            if self.options.joint_weights:
                weights = np.ones_like(q_full)
                weights[active_joint_indices] = np.array(self.options.joint_weights)
                loss = np.sum(weights * (error**2))
            else:
                loss = np.sum(error**2)

            loss += 0.001 * np.sum(nullspace_term**2)
            loss += 10.0 * limit_penalty  # <-- Weight for soft joint limit violation
            return loss

        bounds = [
            (
                (self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i])
                if i in active_joint_indices
                else (init_state[i], init_state[i])
            )
            for i in range(self.model.nq)
        ]

        # Inside your solve() method, wrap the minimize() call
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

        result = minimize(
            objective,
            init_state,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.options.max_iters, "disp": verbose},
        )

        if result.success:
            q_sol = result.x
            if not check_within_limits(self.model, q_sol) or not vamp.panda.validate(
                q_sol[:7]
            ):
                if verbose:
                    print("Solved but outside joint limits")
                return None
            if self.collision_model and check_collisions_at_state(
                self.model, self.collision_model, q_sol, self.data, self.collision_data
            ):
                if verbose:
                    print("Solved but in collision")
                return None
            if self.visualizer:
                self.visualizer.display(q_sol)
                visualize_frame(self.visualizer, "ik_target_pose", target_tform)
            return q_sol
        else:
            if verbose:
                print("Optimization failed")
            return None


from scipy.optimize import minimize
import numpy as np
import pinocchio
import vamp

class MinimumJerkWithEEConstraint:
    def __init__(self, model, data, ee_frame,
                 jerk_weight=10.0, ee_weight=1000.0,
                 jerk_threshold=1e-4, ee_threshold=1e-4,
                 velocity_limit=1.5, accel_weight=5.0, vel_weight=5.0,
                 orientation_weight=0.1, limit_weight=100.0,
                 ee_barrier_max_weight=500.0):
        self.model = model
        self.data = data
        self.frame_id = model.getFrameId(ee_frame)

        self.jerk_weight = jerk_weight
        self.ee_weight = ee_weight
        self.jerk_threshold = jerk_threshold
        self.ee_threshold = ee_threshold

        self.velocity_limit = velocity_limit
        self.accel_weight = accel_weight
        self.vel_weight = vel_weight
        self.orientation_weight = orientation_weight
        self.limit_weight = limit_weight

        self.ee_barrier_max_weight = ee_barrier_max_weight
        self._iter_count = 0
        self.barrier_start_iter = 100
        self.barrier_ramp_iters = 25

        self._last_q_flat = None

    def _normalize_q(self, q):
        min_vals = np.min(q, axis=0)
        ptp_vals = np.ptp(q, axis=0)
        ptp_vals[ptp_vals < 1e-3] = 1.0
        return (q - min_vals) / ptp_vals

    def _jerk_penalty(self, q_traj):
        q = q_traj.reshape(self.T, self.dof)
        q_norm = self._normalize_q(q)
        jerk = q_norm[3:] - 3*q_norm[2:-1] + 3*q_norm[1:-2] - q_norm[:-3]
        return np.sum(jerk**2) / (self.T - 3)

    def _accel_penalty(self, q_traj):
        q = q_traj.reshape(self.T, self.dof)
        accel = q[2:] - 2*q[1:-1] + q[:-2]
        return np.sum(accel**2) / (self.T - 2)

    def _velocity_spike_penalty(self, q_traj):
        q = q_traj.reshape(self.T, self.dof)
        vel = np.diff(q, axis=0)
        excess = np.maximum(np.abs(vel) - self.velocity_limit, 0)
        return np.sum(excess**2) / (self.T - 1)

    def _ee_pose_error(self, q_traj, T_targets, fk_cache):
        q = q_traj.reshape(self.T, self.dof)
        total_error = 0.0
        for t in range(self.T):
            T_current = fk_cache[t]
            twist = pinocchio.log(T_targets[t].actInv(T_current)).vector
            pos_error = np.sum(twist[:3] ** 2)
            rot_error = np.sum(twist[3:] ** 2)
            total_error += pos_error + self.orientation_weight * rot_error
        return total_error / self.T

    def _joint_limit_penalty(self, q_traj):
        q = q_traj.reshape(self.T, self.dof)
        q_arm = q[:, :7]
        lower_viol = np.maximum(Q_LIMS[0] - q_arm, 0)
        upper_viol = np.maximum(q_arm - Q_LIMS[1], 0)
        return np.sum(lower_viol**2 + upper_viol**2) / self.T

    def _joint_limit_barrier(self, q_traj, epsilon=1e-6):
        q = q_traj.reshape(self.T, self.dof)
        q_arm = q[:, :7]
        diff_lower = q_arm - Q_LIMS[0] + epsilon
        diff_upper = Q_LIMS[1] - q_arm + epsilon
        if np.any(diff_lower <= 0) or np.any(diff_upper <= 0):
            return np.inf
        return -np.sum(np.log(diff_lower) + np.log(diff_upper)) / self.T

    def _max_deviation_barrier(self, ee_positions, delta=0.01, margin=0.005):
        max_dev = max_deviation_along_line(ee_positions)
        if max_dev < delta:
            return 0.0
        return ((max_dev - delta) / margin) ** 2

    def _inefficiency_barrier(self, ee_positions, ineff_thresh=0.005):
        ineff = path_efficiency(ee_positions) - 1.0
        if ineff < ineff_thresh:
            return 0.0
        return ((ineff - ineff_thresh) / ineff_thresh) ** 2

    def _manipulability_barrier(self, q_traj, manip_thresh=1e-2, epsilon=1e-6):
        vals = np.array([compute_manipulability(self.model, self.data, q, self.frame_id) for q in q_traj])
        min_val = np.min(vals)
        if min_val > manip_thresh:
            return 0.0
        return -np.log(min_val - manip_thresh + epsilon)

    def _ee_constraint_barrier(self, q_traj,
                               delta=0.01, ineff_thresh=0.005,
                               manip_thresh=1e-2,
                               weights=dict(dev=1.0, ineff=25.0, manip=1.0)):
        ee_positions = np.stack([vamp.panda.eefk(q[:7])[0] for q in q_traj])
        barrier = 0.0
        barrier += weights["dev"]   * self._max_deviation_barrier(ee_positions, delta)
        barrier += weights["ineff"] * self._inefficiency_barrier(ee_positions, ineff_thresh)
        barrier += weights["manip"] * self._manipulability_barrier(q_traj, manip_thresh)
        return barrier

    def solve(self, q_init_traj, target_positions=None, fixed_rotation=None):
        self.T, self.dof = q_init_traj.shape

        use_ee_objective = target_positions is not None and fixed_rotation is not None
        if use_ee_objective:
            T_targets = [pinocchio.SE3(fixed_rotation, p) for p in target_positions]

        def objective(q_flat):
            self._iter_count += 1

            q_traj = q_flat.reshape(self.T, self.dof)

            # Cache FK once per iteration
            fk_cache = []
            for t in range(self.T):
                pinocchio.forwardKinematics(self.model, self.data, q_traj[t])
                fk_cache.append(pinocchio.updateFramePlacement(self.model, self.data, self.frame_id))

            jerk_norm = self._jerk_penalty(q_flat)
            accel_pen = self._accel_penalty(q_flat)
            vel_pen = self._velocity_spike_penalty(q_flat)

            delta = self._iter_count - self.barrier_start_iter
            alpha = np.clip((delta + 0.5 * self.barrier_ramp_iters) / self.barrier_ramp_iters, 0.0, 1.0)

            soft = self._joint_limit_penalty(q_flat)
            barrier = self._joint_limit_barrier(q_flat)
            limit_pen = (1 - alpha) * soft + alpha * barrier

            ee_err_norm = self._ee_pose_error(q_flat, T_targets, fk_cache) if use_ee_objective else 0.0
            ee_barrier_pen = self._ee_constraint_barrier(q_traj)

            self._last_q_flat = q_flat.copy()

            if jerk_norm < self.jerk_threshold and (not use_ee_objective or ee_err_norm < self.ee_threshold):
                raise StopIteration

            adaptive_j_weight = self.jerk_weight * (1 + ee_err_norm)
            adaptive_ee_weight = self.ee_weight * (1 + jerk_norm) if use_ee_objective else 0.0
            ee_barrier_weight = self.ee_barrier_max_weight * alpha

            return (
                adaptive_j_weight * jerk_norm
                + adaptive_ee_weight * ee_err_norm
                + self.accel_weight * accel_pen
                + self.vel_weight * vel_pen
                + self.limit_weight * limit_pen
                + ee_barrier_weight * ee_barrier_pen
            )

        try:
            result = minimize(
                objective,
                q_init_traj.flatten(),
                method="SLSQP",
                options={
                    "maxiter": 300,
                    "ftol": 1e-4,
                    "eps": 1e-3,
                    "disp": False
                },
            )
            return (
                result.x.reshape(self.T, self.dof)
                if result.success
                else (
                    self._last_q_flat.reshape(self.T, self.dof)
                    if self._last_q_flat is not None
                    else None
                )
            )
        except StopIteration:
            return (
                self._last_q_flat.reshape(self.T, self.dof)
                if self._last_q_flat is not None
                else None
            )

