"""Utilities to load example Franka Emika Panda model."""

import coal
import numpy as np
import os
import pinocchio

from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models(use_sphere_collisions=False):
    """
    Gets the example Panda models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "panda_description")
    urdf_filename = "panda_spheres.urdf" if use_sphere_collisions else "panda.urdf"
    urdf_filepath = os.path.join(package_dir, "urdf", urdf_filename)

    return pinocchio.buildModelsFromUrdf(urdf_filepath, package_dirs=models_folder)


def add_self_collisions(model, collision_model, srdf_filename=None):
    """
    Adds link self-collisions to the Panda collision model.

    This uses an SRDF file to remove any excluded collision pairs.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        srdf_filename : str, optional
            Path to the SRDF file describing the excluded collision pairs.
            If not specified, uses a default file included with the Panda model.
    """
    if srdf_filename is None:
        models_folder = get_example_models_folder()
        package_dir = os.path.join(models_folder, "panda_description")
        srdf_filename = os.path.join(package_dir, "srdf", "panda.srdf")

    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)


def add_object_collisions(model, collision_model, visual_model, collision_objects_dict, inflation_radius=0.0):
    """
    Adds user-defined collision objects to the collision and visual models.

    Parameters
    ----------
        model : `pinocchio.Model`
            The robot model.
        collision_model : `pinocchio.GeometryModel`
            The robot collision geometry model.
        visual_model : `pinocchio.GeometryModel`
            The robot visual geometry model.
        collision_objects_dict : dict
            Dictionary where keys are object names and values are `pinocchio.GeometryObject` instances.
        inflation_radius : float, optional
            An inflation radius, in meters, around the objects (applied if the shape has an `inflated` method).
    """
    for name, geom_obj in collision_objects_dict.items():
        shape = geom_obj.geometry
        # Inflate the shape if it has an 'inflated' method
        if hasattr(shape, 'inflated'):
            min_inflation = shape.minInflationValue()
            if inflation_radius >= min_inflation:
                geom_obj.geometry = shape.inflated(inflation_radius)
            else:
                raise ValueError(f"Inflation radius {inflation_radius} is less than the minimum required {min_inflation} for {name}.")
        elif isinstance(shape, coal.Sphere):
            # Directly inflate spheres by increasing the radius
            shape.radius += inflation_radius

        visual_model.addGeometryObject(geom_obj)
        collision_model.addGeometryObject(geom_obj)

    # Define active collision pairs between robot and obstacle links
    collision_names = [cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name]
    obstacle_names = list(collision_objects_dict.keys())

    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)

    # Exclude the collision between the ground and the base link
    if "ground_plane" in collision_objects_dict:
        set_collisions(model, collision_model, "panda_link0", "ground_plane", False)
