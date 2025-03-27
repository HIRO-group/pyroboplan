"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm on a 7-DOF Panda robot.
"""

from pinocchio.visualize import MeshcatVisualizer
import time
import numpy as np

from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames
import pinocchio
import coal


def is_valid_configuration(q, model):
    return np.all(q >= model.lowerPositionLimit) and np.all(q <= model.upperPositionLimit)


def check_path_limits(path, model):
    """
    Check if the path respects joint limits.
    Returns:
        bool: True if the entire path respects joint limits, False otherwise.
    """
    path = np.array(path)  # Convert to numpy array for fast comparison
    lower_violation = path < model.lowerPositionLimit
    upper_violation = path > model.upperPositionLimit
    success = False

    if np.any(lower_violation) or np.any(upper_violation):
        # Print violations for debugging
        for i in range(path.shape[0]):
            for j in range(model.nq):
                if lower_violation[i, j] or upper_violation[i, j]:
                    print(
                        f"❌ Step {i}: Joint {j} out of bounds (value={path[i, j]:.3f}, "
                        f"limit=[{model.lowerPositionLimit[j]:.3f}, {model.upperPositionLimit[j]:.3f}])"
                    )
        success = True
    else:
        success = False
    return success



def plan_and_animate(q_start, q_end, planner, options, viz, model, collision_model):
    print("\nPlanning from:")
    print("Start:", q_start)
    print("End:", q_end)

    # ✅ Check joint limits before planning
    # if not is_valid_configuration(q_start, model) or not is_valid_configuration(q_end, model):
    #     print("❌ Start or goal configuration is out of joint limits!")
    #     return

    path = planner.plan(q_start, q_end)
    if path is None:
        print("❌ No path found (likely due to joint limits or obstacles).")
        return
    
    planner.visualize(viz, "panda_hand", show_tree=True, show_path=True)
    violations = check_path_limits(path, model)
    if violations:
        print("\n⚠️ Joint limit violations detected:")
        for v in violations:
            print(v)
    else:
        print("\n✅ Path respects joint limits.")

    print("✅ Path found!")
    discretized_path = discretize_joint_space_path(path, options.max_step_size)

    # ✅ Check if discretized path obeys joint limits
    violations = check_path_limits(discretized_path, model)
    if violations:
        print("\n⚠️ Joint limit violations detected:")
        for v in violations:
            print(v)
    else:
        print("\n✅ Path respects joint limits.")

    input("Press 'Enter' to animate the path.")
    for q in discretized_path:
        viz.display(q)
        time.sleep(0.05)

    input("Press 'Enter' to continue...")



if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    # ✅ Adjust joint 0's limits to [-1, 1]
    # joint_id = 0
    # model.lowerPositionLimit[joint_id] = -1.0
    # model.upperPositionLimit[joint_id] = 1.0
    
    # breakpoint()
    
    data = model.createData()
    collision_data = collision_model.createData()
    
    add_self_collisions(model, collision_model)
    collision_objects = {}

    # Ground plane
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.151])),
        coal.Box(2.0, 2.0, 0.3),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    collision_objects["ground_plane"] = ground_plane

    # Spheres
    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
        coal.Sphere(0.2),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    collision_objects["obstacle_sphere_1"] = obstacle_sphere_1

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
        coal.Sphere(0.25),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    collision_objects["obstacle_sphere_2"] = obstacle_sphere_2

    # Boxes
    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
        coal.Box(0.25, 0.55, 0.55),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    collision_objects["obstacle_box_1"] = obstacle_box_1

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
        coal.Box(0.33, 0.33, 0.33),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    collision_objects["obstacle_box_2"] = obstacle_box_2

    # Now use the updated function
    add_object_collisions(model, collision_model, visual_model, collision_objects)

    # print(f"Updated joint {joint_id} limits:")
    # print("Lower limit:", model.lowerPositionLimit[joint_id])
    # print("Upper limit:", model.upperPositionLimit[joint_id])
    

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Configure the RRT planner
    options = RRTPlannerOptions(
        max_step_size=0.05,
        max_connection_dist=0.5,
        rrt_connect=True,
        bidirectional_rrt=True,
        rrt_star=True,
        max_rewire_dist=3.0,
        max_planning_time=10.0,
        rng_seed=None,
        fast_return=True,
        goal_biasing_probability=0.15,
        collision_distance_padding=0.0,
    )

    planner = RRTPlanner(model, collision_model, options=options)

    # ✅ TEST 1: Plan within joint limits
    print("\n[TEST 1] Planning within joint limits:")
    # q_start = get_random_collision_free_state(model, collision_model)
    # q_end = get_random_collision_free_state(model, collision_model)
    q_start = np.array([-0.258,  0.826, -0.432, -1.794, 0.54 ,  2.499 , 2.917 , 0.032 , 0.032])
    q_end = np.array([-0.381,  1.698,  1.438, -1.719, -1.717,  1.682, -1., 0.036,  0.005])

    # # Clamp joint 0 within limits
    # q_start[joint_id] = np.random.uniform(-1.0, 1.0)
    # q_end[joint_id] = np.random.uniform(-1.0, 1.0)

    plan_and_animate(q_start, q_end, planner, options, viz, model, collision_model)

    # ✅ TEST 2: Plan outside joint limits (expect failure)
    # print("\n[TEST 2] Attempting to plan outside joint limits:")
    # q_start = get_random_collision_free_state(model, collision_model)
    # q_end = get_random_collision_free_state(model, collision_model)

    # # Set joint 0 outside limits
    # q_start[joint_id] = 2.0  # Outside of [-1, 1]
    # q_end[joint_id] = -2.0   # Outside of [-1, 1]

    # plan_and_animate(q_start, q_end, planner, options, viz, model, collision_model)
