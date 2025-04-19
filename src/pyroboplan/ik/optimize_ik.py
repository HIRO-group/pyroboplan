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

        result = minimize(
            objective,
            init_state,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.options.max_iters, 'disp': verbose}
        )

        if result.success:
            q_sol = result.x
            q_sol = (q_sol + np.pi) % (2 * np.pi) - np.pi
            if not check_within_limits(self.model, q_sol):
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
