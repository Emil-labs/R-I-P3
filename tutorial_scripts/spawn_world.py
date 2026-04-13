import os
import numpy as np
import pydot
from IPython.display import SVG, display

# Import necessary parts of Drake
from pydrake.geometry import StartMeshcat, SceneGraph, Box as DrakeBox, HalfSpace
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

# --------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------
# Start Meshcat visualizer: this will give you a link in the notebook/terminal
meshcat = StartMeshcat()

visualize = True  # If True: use ModelVisualizer interactively, if False: run a simulation


model_path = os.path.join(
    "..", "models", "objects_scenes", "project_03_shape_formation_t_world.sdf"
)

# --------------------------------------------------------------------
# Function: create_sim_scene
# --------------------------------------------------------------------
def create_sim_scene(sim_time_step):   
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)

    parser = Parser(plant)
    world_model = parser.AddModelsFromUrl("file://" + os.path.abspath(model_path))[0]


    j1 = plant.GetJointByName("j2s7s300_joint_1")
    j2 = plant.GetJointByName("j2s7s300_joint_2")
    j3 = plant.GetJointByName("j2s7s300_joint_3")
    j4 = plant.GetJointByName("j2s7s300_joint_4")
    j5 = plant.GetJointByName("j2s7s300_joint_5")
    j6 = plant.GetJointByName("j2s7s300_joint_6")
    #j7 = plant.GetJointByName("j2s7s300_joint_7")

    f1 = plant.GetJointByName("j2s7s300_joint_finger_1")
    f2 = plant.GetJointByName("j2s7s300_joint_finger_2")
    #f3 = plant.GetJointByName("j2s7s300_joint_finger_3")

    # bras
    j1.set_default_angle(-10.0)
    j2.set_default_angle(1.0)  
    j3.set_default_angle(0.0)
    j4.set_default_angle(1.0)
    j5.set_default_angle(0.0)
    j6.set_default_angle(1.0)
    #j7.set_default_angle(0.0)

    # Doigts
    f1.set_default_angle(0.0)
    f2.set_default_angle(0.0)
    #f3.set_default_angle(0.0)


    plant.Finalize()

    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    return diagram

# --------------------------------------------------------------------
# Function: run_simulation
# --------------------------------------------------------------------
def run_simulation(sim_time_step):
    """
    Either run an interactive visualizer, or simulate the system.
    """
    if visualize:
        # If visualize=True, just load and display the robot interactively
        visualizer = ModelVisualizer(meshcat=meshcat)
        visualizer.parser().AddModelsFromUrl("file://" + os.path.abspath(model_path))
        visualizer.Run()
        
    else:
        # Otherwise, build the scene and simulate
        diagram = create_sim_scene(sim_time_step)

        # Create and configure the simulator
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)  # Try to match real time
        simulator.Initialize()
        simulator.set_publish_every_time_step(True)  # publish at each step

        sim_time = 5.0  # seconds of simulated time

        meshcat.StartRecording()         # Start recording the sim
        simulator.AdvanceTo(sim_time)    # Runs the simulation for sim_time seconds
        meshcat.PublishRecording()       # Publish recording to replay in Meshcat
            
        # Save system block diagram as PNG
        svg_data = diagram.GetGraphvizString(max_depth=2)
        graph = pydot.graph_from_dot_data(svg_data)[0]
        image_path = "figures/block_diagram_02.png"
        graph.write_png(image_path)
        print(f"\nBlock diagram saved as: {image_path}")


# --------------------------------------------------------------------
# Run the simulation
# --------------------------------------------------------------------
# Try playing with the time step (e.g. 0.001 vs 0.01 vs 0.1)
run_simulation(sim_time_step=0.01)
