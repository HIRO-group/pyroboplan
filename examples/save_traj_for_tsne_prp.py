import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import time

from pyroboplan.models.panda import load_models
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.core.utils import get_random_state, check_collisions_at_state


# === Joint Limits ===
Q_BUFFER = 0.1
FP_BUFFER = 1e-3

Q_LIMS = np.array(
    [
        [
            -2.8973 + Q_BUFFER - FP_BUFFER,
            -1.7628 + Q_BUFFER - FP_BUFFER,
            -2.8973 + Q_BUFFER - FP_BUFFER,
            -3.0718 + Q_BUFFER - FP_BUFFER,
            -2.8973 + Q_BUFFER - FP_BUFFER,
            -0.0175 + 10 * Q_BUFFER - FP_BUFFER,
            -2.8973 + Q_BUFFER - FP_BUFFER,
        ],
        [
            2.8973 - Q_BUFFER + FP_BUFFER,
            1.7628 - Q_BUFFER + FP_BUFFER,
            2.8973 - Q_BUFFER + FP_BUFFER,
            -0.0698 - Q_BUFFER + FP_BUFFER,
            2.8973 - Q_BUFFER + FP_BUFFER,
            3.7525 - 10 * Q_BUFFER + FP_BUFFER,
            2.8973 - Q_BUFFER + FP_BUFFER,
        ],
    ]
)
DQ_LIMS = np.array(
    [
        [
            -2.1750 - FP_BUFFER,
            -2.1750 - FP_BUFFER,
            -2.1750 - FP_BUFFER,
            -2.1750 - FP_BUFFER,
            -2.6100 - FP_BUFFER,
            -2.6100 - FP_BUFFER,
            -2.6100 - FP_BUFFER,
        ],
        [
            2.1750 + FP_BUFFER,
            2.1750 + FP_BUFFER,
            2.1750 + FP_BUFFER,
            2.1750 + FP_BUFFER,
            2.6100 + FP_BUFFER,
            2.6100 + FP_BUFFER,
            2.6100 + FP_BUFFER,
        ],
    ]
)
DDQ_LIMS = np.array(
    [
        [
            -15 - FP_BUFFER,
            -7.5 - FP_BUFFER,
            -10 - FP_BUFFER,
            -12.5 - FP_BUFFER,
            -15 - FP_BUFFER,
            -20 - FP_BUFFER,
            -20 - FP_BUFFER,
        ],
        [
            15 + FP_BUFFER,
            7.5 + FP_BUFFER,
            10 + FP_BUFFER,
            12.5 + FP_BUFFER,
            15 + FP_BUFFER,
            20 + FP_BUFFER,
            20 + FP_BUFFER,
        ],
    ]
)


# === Plotting Functions ===
def plot_trajectory(
    data, start_state, goal_state, title, ylabel, filename, state_index, failure_cond
):
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(title, fontsize=14)

    for i in range(7):
        if failure_cond[i] == 0:
            axes[i].set_facecolor((1.0, 0.9, 0.9))  # Light red color for locked joint

        axes[i].plot(data[:, i], label=f"Joint {i+1}")
        axes[i].scatter(
            0,
            start_state[state_index + i],
            color="green",
            marker="o",
            label="Start",
            zorder=3,
        )
        axes[i].scatter(
            len(data) - 1,
            goal_state[state_index + i],
            color="red",
            marker="o",
            label="Goal",
            zorder=3,
        )
        axes[i].set_ylabel(ylabel)
        axes[i].set_ylim(Q_LIMS[1][i], Q_LIMS[0][i])

    axes[-1].set_xlabel("Time Steps")
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_joint_trajectories(
    q_traj, dq_traj, ddq_traj, start_state, goal_state, failure_cond, save_path
):
    plot_trajectory(
        q_traj,
        start_state,
        goal_state,
        "Joint Position Trajectory",
        "Position (rad)",
        f"{save_path}/q_traj.png",
        0,
        failure_cond,
    )
    plot_trajectory(
        dq_traj,
        start_state,
        goal_state,
        "Joint Velocity Trajectory",
        "Velocity (rad/s)",
        f"{save_path}/dq_traj.png",
        7,
        failure_cond,
    )
    plot_trajectory(
        ddq_traj,
        start_state,
        goal_state,
        "Joint Acceleration Trajectory",
        "Acceleration (rad/s²)",
        f"{save_path}/ddq_traj.png",
        14,
        failure_cond,
    )


# === Joint Limit Checking ===
def exceeds_joint_limits(q_traj, dq_traj, ddq_traj, failure_cond):
    # Position limits
    q_exceeded = not np.all(
        (q_traj[:, :7] >= failure_cond[::2][:7]) & (q_traj[:, :7]  <= failure_cond[1::2][:7])
    )
    # Velocity limits
    dq_exceeded = not np.all(
        (dq_traj[:, :7]  >= failure_cond[14::2][:7]) & (dq_traj[:, :7]  <= failure_cond[15::2][:7])
    )
    # Acceleration limits
    ddq_exceeded = not np.all(
        (ddq_traj[:, :7]  >= failure_cond[28::2][:7]) & (ddq_traj[:, :7]  <= failure_cond[29::2][:7])
    )

    if q_exceeded:
        return 20
    if dq_exceeded:
        return 21
    if ddq_exceeded:
        return 22
    return None


# === Failure Condition Sampling ===
def generate_random_failure_condition(model, failure_type=None, joint=None):
    """
    Generates a random failure condition based on joint, velocity, and acceleration limits.

    Parameters
    ----------
        model : pinocchio.Model
            The robot model.
        failure_type : str
            "range", "velocity", "acceleration", "locked"
        joint : int
            Joint index to apply the failure to.

    Returns
    -------
        failure_cond : np.array
            Interleaved position, velocity, and acceleration limits.
    """
    q = Q_LIMS.T.reshape(-1)  # Position limits interleaved
    dq = DQ_LIMS.T.reshape(-1)  # Velocity limits interleaved
    ddq = DDQ_LIMS.T.reshape(-1)  # Acceleration limits interleaved

    failure_cond = np.concatenate((q, dq, ddq), axis=0)

    if failure_type is not None and joint is not None:
        if failure_type == "range":
            # Reduce position range
            range_size = np.random.uniform(0.1, 0.5) * (
                Q_LIMS[1, joint] - Q_LIMS[0, joint]
            )
            center = np.random.uniform(
                Q_LIMS[0, joint] + range_size / 2, Q_LIMS[1, joint] - range_size / 2
            )
            failure_cond[joint * 2] = lower = center - range_size / 2
            failure_cond[joint * 2 + 1] = upper = center + range_size / 2

            model.lowerPositionLimit = np.append(lower, [0.14, 0.14])
            model.upperPositionLimit = np.append(upper, [0.14, 0.14])

        elif failure_type == "velocity":
            # Reduce velocity range
            range_size = np.random.uniform(0.1, 0.5) * (
                DQ_LIMS[1, joint] - DQ_LIMS[0, joint]
            )
            center = 0
            failure_cond[14 + joint * 2] = lower = center - range_size / 2
            failure_cond[14 + joint * 2 + 1] = upper = center + range_size / 2

            model.velocityLimit[joint] = upper

        elif failure_type == "acceleration":
            # # Reduce acceleration range
            # range_size = np.random.uniform(0.1, 0.5) * (
            #     DDQ_LIMS[1, joint] - DDQ_LIMS[0, joint]
            # )
            # center = 0
            # failure_cond[28 + joint * 2] = lower = center - range_size / 2
            # failure_cond[28 + joint * 2 + 1] = upper = center + range_size / 2

            # model.accelerationLimit[joint] = upper
            raise NotImplementedError("Acceleration limits are not currently supported.")

        elif failure_type == "locked":
            # Create a very small but non-zero range to simulate a locked joint
            locked_value = np.random.uniform(Q_LIMS[0, joint], Q_LIMS[1, joint])
            small_range = np.random.uniform(0.005, 0.01)

            lower = max(Q_LIMS[0, joint], locked_value - small_range)
            upper = min(Q_LIMS[1, joint], locked_value + small_range)

            failure_cond[joint * 2] = lower
            failure_cond[joint * 2 + 1] = upper

            # Lock velocity and acceleration limits to zero
            failure_cond[14 + joint * 2] = 0.0 - FP_BUFFER
            failure_cond[14 + joint * 2 + 1] = 0.0 + FP_BUFFER
            failure_cond[28 + joint * 2] = 0.0 - FP_BUFFER
            failure_cond[28 + joint * 2 + 1] = 0.0 + FP_BUFFER

            model.lowerPositionLimit[joint] = lower
            model.upperPositionLimit[joint] = upper
            model.velocityLimit[joint] = 0.0
            # model.accelerationLimit[joint] = 0.0

    return failure_cond


# === Random Start and Goal State Generation ===
def get_random_collision_free_state(
    model, collision_model, joint_padding=0.0, distance_padding=0.0, max_tries=100
):
    """
    Returns a random collision-free state within joint limits.
    """
    for _ in range(max_tries):
        state = get_random_state(model, padding=joint_padding)
        if not check_collisions_at_state(
            model, collision_model, state, distance_padding=distance_padding
        ):
            return state
    print(f"Could not generate collision-free state after {max_tries} tries.")
    return None


def generate_random_start_goal(
    model,
    collision_model,
    failure_type=None,
    joint=None,
    max_attempts=100,
):
    """
    Generates a random start and goal configuration with optional failure conditions.
    """
    # Sample failure condition and apply to the model
    failure_cond = generate_random_failure_condition(model, failure_type, joint)

    # Attempt to generate a valid start state
    start_q = None
    for _ in range(max_attempts):
        start_q = get_random_collision_free_state(model, collision_model)
        if start_q is not None:
            break
    else:
        print("Failed to generate valid start state after multiple attempts.")
        return None, None, None, False

    # Attempt to generate a valid goal state
    goal_q = None
    for _ in range(max_attempts):
        goal_q = get_random_collision_free_state(model, collision_model)
        if goal_q is not None:
            break
    else:
        print("Failed to generate valid goal state after multiple attempts.")
        return None, None, None, False

    return start_q, goal_q, failure_cond, True


# === Main Script ===
def main(args):
    model, collision_model, visual_model = load_models()

    options = RRTPlannerOptions(
        max_step_size=0.05,
        max_connection_dist=0.5,
        rrt_connect=False,
        bidirectional_rrt=True,
        rrt_star=True,
        max_rewire_dist=3.0,
        max_planning_time=5.0,
        goal_biasing_probability=0.1,
    )

    planner = RRTPlanner(model, collision_model, options)

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_save_path = os.path.join("./figs", f"eval_figs/{dt_str}")
    os.makedirs(base_save_path, exist_ok=True)

    traj_h5_file = os.path.join(base_save_path, "trajectories-rvl.hdf5")
    # failure_type = "locked"
    
    num_trajs = 100000
    pbar = tqdm(total=7*num_trajs)

    with h5py.File(traj_h5_file, "w") as traj_h5:
        for joint in range(7):
            group = traj_h5.create_group(str(joint))
            group.create_dataset("trajectories", (3, num_trajs, 1000, 9), dtype="float64")
            group.create_dataset("locked_values", (num_trajs,), dtype="float64")
            group.create_dataset("valid_mask", (num_trajs,), dtype="bool")
            group.create_dataset("failure_reason", (num_trajs,), dtype="int32")
            group.create_dataset("failure_cond", (num_trajs, 42), dtype="float64")
            
            # ✅ New datasets for start and goal states
            group.create_dataset("start_states", (num_trajs, 9), dtype="float64")
            group.create_dataset("goal_states", (num_trajs, 9), dtype="float64")

        for joint in range(7):
            locked_values = np.linspace(Q_LIMS[1][joint], Q_LIMS[0][joint], num_trajs)

            for i, locked_value in enumerate(locked_values):
                pbar.update(1)

                # Generate a random failure condition and apply to the model
                failure_type = np.random.choice(["range", "velocity", "locked"])


                start_state, goal_state, failure_cond, success = generate_random_start_goal(
                    model, collision_model, failure_type=failure_type, joint=joint
                )
                
                
                planner.update_model(model, collision_model)
                # print("model updated")
                if not success:
                    # Failed to generate valid start/goal state
                    traj_h5[str(joint)]["trajectories"][0, i, :, :] = np.nan
                    traj_h5[str(joint)]["trajectories"][1, i, :, :] = np.nan
                    traj_h5[str(joint)]["trajectories"][2, i, :, :] = np.nan
                    traj_h5[str(joint)]["locked_values"][i] = np.nan
                    traj_h5[str(joint)]["valid_mask"][i] = False
                    traj_h5[str(joint)]["failure_reason"][i] = 1
                    traj_h5[str(joint)]["failure_cond"][i] = np.zeros(42)
                    # ✅ Save empty start and goal states on failure
                    traj_h5[str(joint)]["start_states"][i] = start_state
                    traj_h5[str(joint)]["goal_states"][i] = goal_state
                    continue


                path = planner.plan(start_state, goal_state)
                # print("plan made")

                if path is None:
                    # Failed to generate trajectory
                    traj_h5[str(joint)]["trajectories"][0, i, :, :] = np.nan
                    traj_h5[str(joint)]["trajectories"][1, i, :, :] = np.nan
                    traj_h5[str(joint)]["trajectories"][2, i, :, :] = np.nan
                    traj_h5[str(joint)]["locked_values"][i] = np.nan
                    traj_h5[str(joint)]["valid_mask"][i] = False
                    traj_h5[str(joint)]["failure_reason"][i] = 3
                    traj_h5[str(joint)]["failure_cond"][i] = failure_cond
                    # ✅ Save empty start and goal states on failure
                    traj_h5[str(joint)]["start_states"][i] = start_state
                    traj_h5[str(joint)]["goal_states"][i] = goal_state
                    continue

                # Discretize and calculate trajectory data
                q_traj = np.array(discretize_joint_space_path(path, options.max_step_size))
                dq_traj = np.gradient(q_traj, axis=0)
                ddq_traj = np.gradient(dq_traj, axis=0)

                exceed_reason = exceeds_joint_limits(q_traj, dq_traj, ddq_traj, failure_cond)
                # ✅ Pad trajectory data to length of 1000
                q_traj_padded = np.zeros((1000, 9))
                dq_traj_padded = np.zeros((1000, 9))
                ddq_traj_padded = np.zeros((1000, 9))

                q_traj_padded[:len(q_traj)] = q_traj
                dq_traj_padded[:len(dq_traj)] = dq_traj
                ddq_traj_padded[:len(ddq_traj)] = ddq_traj

                if exceed_reason:
                    # Trajectory exceeded joint limits
                    traj_h5[str(joint)]["trajectories"][0, i, :, :] = q_traj_padded
                    traj_h5[str(joint)]["trajectories"][1, i, :, :] = dq_traj_padded
                    traj_h5[str(joint)]["trajectories"][2, i, :, :] = ddq_traj_padded
                    traj_h5[str(joint)]["locked_values"][i] = locked_value
                    traj_h5[str(joint)]["valid_mask"][i] = False
                    traj_h5[str(joint)]["failure_reason"][i] = exceed_reason
                    traj_h5[str(joint)]["failure_cond"][i] = failure_cond
                    # ✅ Save empty start and goal states on failure
                    traj_h5[str(joint)]["start_states"][i] = start_state
                    traj_h5[str(joint)]["goal_states"][i] = goal_state
                    continue

                # ✅ Save start and goal states
                traj_h5[str(joint)]["start_states"][i] = start_state
                traj_h5[str(joint)]["goal_states"][i] = goal_state

                

                # ✅ Save trajectory data
                traj_h5[str(joint)]["trajectories"][0, i, :, :] = q_traj_padded
                traj_h5[str(joint)]["trajectories"][1, i, :, :] = dq_traj_padded
                traj_h5[str(joint)]["trajectories"][2, i, :, :] = ddq_traj_padded

                # ✅ Save the locked joint value
                traj_h5[str(joint)]["locked_values"][i] = locked_value
                traj_h5[str(joint)]["failure_cond"][i] = failure_cond

                # ✅ Mark as valid
                traj_h5[str(joint)]["valid_mask"][i] = True
                traj_h5[str(joint)]["failure_reason"][i] = 0

        print(f"✅ Saved trajectories to: {traj_h5_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
