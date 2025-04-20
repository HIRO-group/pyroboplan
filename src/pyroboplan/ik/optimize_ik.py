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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



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
            idx for idx in range(self.model.nq)
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
            if self.options.joint_weights:
                weights = np.ones_like(q_full)
                weights[active_joint_indices] = np.array(self.options.joint_weights)
                return np.sum(weights * (error ** 2)) + 0.001 * np.sum(nullspace_term ** 2)
            else:
                return np.sum(error ** 2) + 0.001 * np.sum(nullspace_term ** 2)

        bounds = [
            (self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i])
            if i in active_joint_indices else (init_state[i], init_state[i])
            for i in range(self.model.nq)
        ]

        # Inside your solve() method, wrap the minimize() call
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

            result = minimize(
                objective,
                init_state,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': self.options.max_iters, 'disp': verbose}
            )


        if result.success:
            q_sol = result.x
            if not check_within_limits(self.model, q_sol) or not vamp.panda.validate(q_sol[:7]):
                if verbose:
                    print("Solved but outside joint limits")
                return None
            if self.collision_model and check_collisions_at_state(self.model, self.collision_model, q_sol, self.data, self.collision_data):
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



class MinimumJerkWithEEConstraint:
    def __init__(self, model, data, ee_frame,
                 jerk_weight=10.0, ee_weight=100.0,
                 jerk_threshold=1e-4, ee_threshold=1e-4,
                 velocity_limit=1.5, accel_weight=5.0, vel_weight=5.0,
                 orientation_weight=1.0):
        """
        Args:
            model, data, ee_frame: Robot and kinematic info.
            jerk_weight, ee_weight: Base weights.
            jerk_threshold, ee_threshold: Early stopping thresholds.
            velocity_limit: Max velocity allowed per joint (rad/s).
            accel_weight, vel_weight: Additional weights for penalties.
            orientation_weight: Weight for orientation error vs translation error.
        """
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

        self._last_q_flat = None

    def _normalize_q(self, q):
        min_vals = np.min(q, axis=0)
        ptp_vals = np.ptp(q, axis=0)
        ptp_vals[ptp_vals < 1e-3] = 1.0  # prevent division by tiny ranges
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

    def _ee_pose_error(self, q_traj, T_targets):
        q = q_traj.reshape(self.T, self.dof)
        total_error = 0.0
        for t in range(self.T):
            pinocchio.forwardKinematics(self.model, self.data, q[t])
            T_current = pinocchio.updateFramePlacement(self.model, self.data, self.frame_id)

            twist = pinocchio.log(T_targets[t].actInv(T_current)).vector
            pos_error = np.sum(twist[:3] ** 2)
            rot_error = np.sum(twist[3:] ** 2)

            total_error += pos_error + self.orientation_weight * rot_error

        return total_error / self.T

    def solve(self, q_init_traj, target_positions=None, fixed_rotation=None):
        self.T, self.dof = q_init_traj.shape

        use_ee_objective = target_positions is not None and fixed_rotation is not None
        if use_ee_objective:
            T_targets = [pinocchio.SE3(fixed_rotation, p) for p in target_positions]

        def objective(q_flat):
            jerk_norm = self._jerk_penalty(q_flat)
            accel_pen = self._accel_penalty(q_flat)
            vel_pen = self._velocity_spike_penalty(q_flat)

            if use_ee_objective:
                ee_err_norm = self._ee_pose_error(q_flat, T_targets)
            else:
                ee_err_norm = 0.0

            self._last_q_flat = q_flat.copy()

            if jerk_norm < self.jerk_threshold and (not use_ee_objective or ee_err_norm < self.ee_threshold):
                raise StopIteration

            adaptive_j_weight = self.jerk_weight * (1 + ee_err_norm)
            adaptive_ee_weight = self.ee_weight * (1 + jerk_norm) if use_ee_objective else 0.0

            loss = (
                adaptive_j_weight * jerk_norm +
                adaptive_ee_weight * ee_err_norm +
                self.accel_weight * accel_pen +
                self.vel_weight * vel_pen
            )
            return loss

        try:
            result = minimize(
                objective,
                q_init_traj.flatten(),
                method='L-BFGS-B',
                jac='2-point',
                options={'maxiter': 500, 'disp': True}
            )
            return result.x.reshape(self.T, self.dof) if result.success else (
                self._last_q_flat.reshape(self.T, self.dof) if self._last_q_flat is not None else None
            )

        except StopIteration:
            return self._last_q_flat.reshape(self.T, self.dof) if self._last_q_flat is not None else None
