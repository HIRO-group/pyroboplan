"""
This example shows PyRoboPlan capabilities for inverse kinematics (IK).
IK defines the task of finding a set of joint positions for a robot model to
achieve a desired target pose for a specific coordinate frame.
"""

from pinocchio.visualize import MeshcatVisualizer
import numpy as np

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    get_random_collision_free_transform,
)
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import (
    joint_limit_nullspace_component,
    collision_avoidance_nullspace_component,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
import pinocchio
import coal


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    
    joint_id = 0
    model.lowerPositionLimit[joint_id] = -0.3
    model.upperPositionLimit[joint_id] = 0.3
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
    # obstacle_sphere_1 = pinocchio.GeometryObject(
    #     "obstacle_sphere_1",
    #     0,
    #     pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
    #     coal.Sphere(0.2),
    # )
    # obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    # collision_objects["obstacle_sphere_1"] = obstacle_sphere_1

    # obstacle_sphere_2 = pinocchio.GeometryObject(
    #     "obstacle_sphere_2",
    #     0,
    #     pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
    #     coal.Sphere(0.25),
    # )
    # obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    # collision_objects["obstacle_sphere_2"] = obstacle_sphere_2

    # # Boxes
    # obstacle_box_1 = pinocchio.GeometryObject(
    #     "obstacle_box_1",
    #     0,
    #     pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    #     coal.Box(0.25, 0.55, 0.55),
    # )
    # obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    # collision_objects["obstacle_box_1"] = obstacle_box_1

    # obstacle_box_2 = pinocchio.GeometryObject(
    #     "obstacle_box_2",
    #     0,
    #     pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
    #     coal.Box(0.33, 0.33, 0.33),
    # )
    # obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    # collision_objects["obstacle_box_2"] = obstacle_box_2

    # Now use the updated function
    add_object_collisions(model, collision_model, visual_model, collision_objects, inflation_radius=0.0)

    data = model.createData()
    collision_data = collision_model.createData()

    target_frame = "panda_hand"
    ignore_joint_indices = [
        model.getJointId("panda_finger_joint1") - 1,
        model.getJointId("panda_finger_joint2") - 1,
    ]

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    np.set_printoptions(precision=3)

    # Set up the IK solver
    options = DifferentialIkOptions(
        damping=0.0001,
        min_step_size=0.025,
        max_step_size=0.1,
        ignore_joint_indices=ignore_joint_indices,
        rng_seed=None,
    )
    ik = DifferentialIk(
        model,
        data=data,
        collision_model=collision_model,
        options=options,
        visualizer=viz,
    )
    nullspace_components = [
        lambda model, q: collision_avoidance_nullspace_component(
            model,
            data,
            collision_model,
            collision_data,
            q,
            gain=1.0,
            dist_padding=0.05,
        ),
        lambda model, q: joint_limit_nullspace_component(
            model, q, gain=0.1, padding=0.025
        ),
    ]

    # # Solve IK several times and print the results
    # for _ in range(10):
    # init_state = get_random_collision_free_state(model, collision_model)
    # breakpoint()
    init_state = np.array([0,  0.259,  2.4,   -2.098, -0.183 , 1.9  , -2.21  , 0.028,  0.014])
    # init_state = np.array([-2.144, -0.793,  2.81,  -1.808,  0.417,  2.531, -1.944,  0.028,  0.014])
    # target_tform = get_random_collision_free_transform(
    #     model,
    #     collision_model,
    #     target_frame,
    #     joint_padding=0.05,
    # )
    # Identity rotation matrix (no rotation)
    R = np.array([
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0]
    ])

    # Translation vector
    t = np.array([0.4, -0.4, 0.12])

    # SE3 transformation
    T = pinocchio.SE3(R, t)
    
    q_sol = ik.solve(
        target_frame,
        T,
        init_state=init_state,
        nullspace_components=nullspace_components,
        verbose=True,
    )
    print(f"Solution configuration:\n{q_sol}\n")
