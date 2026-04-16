import os
import numpy as np

# Drake imports
from pydrake.geometry import StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

# --------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------
meshcat = StartMeshcat()

visualize = False # False = simulation + accès aux objets

# Chemin robuste vers le SDF
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    current_dir,
    "..",
    "models",
    "objects_scenes",
    "project_03_shape_formation_t_world.sdf"
)

# --------------------------------------------------------------------
# Create simulation scene
# --------------------------------------------------------------------
def create_sim_scene(sim_time_step):
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)

    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + os.path.abspath(model_path))

    # Le fichier sdf contient déjà tous les liens entre les joints ducoup pas besoin 
    plant.Finalize()

    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()

    return diagram, plant

# --------------------------------------------------------------------
# Run simulation
# --------------------------------------------------------------------
def run_simulation(sim_time_step):

    if visualize: #j'ai littéralement repris ce qu'ils ont expliqué dans le tuto 2 pour le visualize
        visualizer = ModelVisualizer(meshcat=meshcat)
        visualizer.parser().AddModelsFromUrl("file://" + os.path.abspath(model_path))
        visualizer.Run()

    else:
        diagram, plant = create_sim_scene(sim_time_step)

        # Context Drake
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        # on va chercher les différents bodies
        print("\n Les bodies :")

        for i in range(plant.num_bodies()):
            body = plant.get_body(BodyIndex(i))
            model_instance = body.model_instance()
            model_name = plant.GetModelInstanceName(model_instance)

            print(f"{body.name()} model: {model_name}")
        
        # Les cubes et les targets doivent d'office se trouver dans les bodies
        cubes = []
        targets = []

        for i in range(plant.num_bodies()):
            body = plant.get_body(BodyIndex(i))
            model_name = plant.GetModelInstanceName(body.model_instance()).lower()
            

            if "cube" in model_name:
                cubes.append(body)

            if "target" in model_name:
                targets.append(body)

        # On recherche 
        print("\n Cubes positions : ")
        for cube in cubes:
            X = plant.EvalBodyPoseInWorld(plant_context, cube)
            print(cube.name(), X.translation())

        # On recherche les coordonnées des targets
        print("\n Targets positions : ")
        for target in targets:
            X = plant.EvalBodyPoseInWorld(plant_context, target)
            print(target.name(), X.translation())

        # -----------------------------
        # Simulation
        # -----------------------------
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()

        sim_time = 5.0

        meshcat.StartRecording()
        simulator.AdvanceTo(sim_time)
        meshcat.PublishRecording()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
run_simulation(sim_time_step=0.001)