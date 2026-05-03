import os
import shutil
import numpy as np
import pydot
from IPython.display import SVG, display
import matplotlib.pyplot as plt
from pydrake.multibody.tree import BodyIndex
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.math import RotationMatrix, RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput
from pydrake.all import Variable, MakeVectorVariable

from helper.dynamics import CalcRobotDynamics
from pydrake.all import (
    InverseKinematics,
    Solve,
)
# --------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------
meshcat = StartMeshcat()

visualize = False # False = simulation + accès aux objets

# path to the sdf file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    current_dir,
    "..",
    "models",
    "objects_scenes",
    "project_03_shape_formation_t_world.sdf"
)


# --------------------------------------------------------------------
# Create the IK Solver 
# --------------------------------------------------------------------
#pris du tuto 4
def solve_ik(plant, context, frame_E, X_WE_desired):
    """
    Solves inverse kinematics for a given end-effector pose.

    Args:
        plant: MultibodyPlant
        context: plant.CreateDefaultContext() or similar
        frame_E: End-effector Frame (e.g. plant.GetFrameByName("ee"))
        X_WE_desired: RigidTransform of desired world pose of end-effector

    Returns:
        q_solution: numpy array of joint positions if successful, else None
    """
    ik = InverseKinematics(plant, context)

    # Set nominal joint positions to current positions
    q_nominal = plant.GetPositions(context).reshape((-1, 1))

    
    # Constrain position and orientation
    # Position constraint
    p_AQ = X_WE_desired.translation().reshape((3, 1))
    ik.AddPositionConstraint(
        frameB=frame_E,
        p_BQ=np.zeros((3, 1)),  # Here, p_BQ = [0, 0, 0] means we’re constraining the origin of the E frame.
        frameA=plant.world_frame(),
        p_AQ_lower=p_AQ,
        p_AQ_upper=p_AQ
    )

    # Orientation constraint
    """
    theta_bound = 0.3 #c'était à 0.01  # radians
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),      # world frame
        R_AbarA=X_WE_desired.rotation(),    # desired orientation
        frameBbar=frame_E,                  # end-effector frame
        R_BbarB=RotationMatrix(),           # current orientation
        theta_bound=theta_bound             # allowable deviation
    )
    """
    # Access the underlying MathematicalProgram to add costs and constraints manually.
    prog = ik.prog()
    q_var = ik.q()  # decision variables (joint angles)
    # Add a quadratic cost to stay close to the nominal configuration:
    #   cost = (q - q_nominal)^T * W * (q - q_nominal)
    W = np.identity(q_nominal.shape[0])
    prog.AddQuadraticErrorCost(W, q_nominal, q_var)

    # Enforce joint position limits from the robot model.
    lower = plant.GetPositionLowerLimits()
    upper = plant.GetPositionUpperLimits()
    prog.AddBoundingBoxConstraint(lower, upper, q_var)


    # Solve the optimization problem using Drake’s default solver.
    # The initial guess is the nominal configuration (q_nominal).
    result = Solve(prog, q_nominal)

    # Check if the solver succeeded and return the solution.
    if result.is_success():
        q_sol = result.GetSolution(q_var)
        return q_sol
    else:
        print("IK did not converge!")
        return None


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

     #Application of IK to one Cube
     
          # on prend le premier cube trouvé                   
        cube_test = cubes[0]                               

        # on récupère sa vraie position dans le monde      
        X_WCube = plant.EvalBodyPoseInWorld(plant_context, cube_test)  # c'est mieux d'aller chercher la position dans 'le monde ' qu'on a crée que de la rentrer manuellement
        p_cube = X_WCube.translation()                      # on extrait seulement la position du cube pas l'orientation (genre c'est comme la matrice homogeneous la forme)

        # on vise une position au-dessus du cube et pas le centre du cube  
        p_des = p_cube + np.array([0.0, 0.0, 0.40])        # c'est le meilleur moyen pour éviter les collisons de toujours se déplacer au dessus et de après descendre 
        R_des = RotationMatrix(RollPitchYaw(np.pi, 0, np.pi/2)) #mettre autre chose que 0 en z pout être sûrs qu'en s'ouvrant la pince ne touche pas les cubes d'à coté
        X_WE_desired = RigidTransform(R_des, p_des)               # J'ai rajouté les rotations parce que la ça donnait la bonne position initial (genre au dessus du cube mais avec une rot de 180° ducoup la pince était vers le haut)
        # littérallemnt ce qui est fait dans le tuto 4
        frame_E = plant.GetFrameByName("panda_hand") # on récupère le frame de la main du robot (c'est ce frame qu'on va contraindre dans l'IK)

        #make the robot go to that position in the Robot context (not the global)
        q_target_pick = solve_ik(plant, plant_context, frame_E, X_WE_desired)   #joints configuration so that we reach the desired position 
        if q_target_pick is not None:
            plant.SetPositions(plant_context, q_target_pick)
            plant.SetVelocities(plant_context, np.zeros(plant.num_velocities()))

            X_WHand = plant.EvalBodyPoseInWorld(
                plant_context,
                plant.GetBodyByName("panda_hand")
             )

            print("Position désirée :", X_WE_desired.translation())
            print("Position réelle panda_hand :", X_WHand.translation())

            diagram.ForcedPublish(context)
            input("Appuie sur Entrée pour lancer la simulation...")

        # parce que c'est bien de faire des prints pour voir si on a ce qu'on veut
        print("\n Test IK sur un cube :")                   
        print("Cube choisi :", plant.GetModelInstanceName(cube_test.model_instance()))  
        print("Position cube :", p_cube)                    
        print("Position désirée main :", p_des)  
              

        if q_target_pick is not None:                            
            print("IK réussie")                             
            print("q_target_pick =", q_target_pick)                   
        else:                                               
            print("IK échouée")   


             #Application of IK to one Target

        # on prend la première target trouvée
        target_test = targets[1]

        # on récupère sa vraie position dans le monde
        X_WTarget = plant.EvalBodyPoseInWorld(plant_context, target_test)
        p_target = X_WTarget.translation()

        # on vise une position au-dessus de la target
        p_des_target = p_target + np.array([0.0, 0.0, 0.40])
        X_WE_target_desired = RigidTransform(R_des, p_des_target)
        context_target = diagram.CreateDefaultContext()
        plant_context_target = plant.GetMyContextFromRoot(context_target)
        # configuration articulaire pour aller au-dessus de la target
        q_target_place = solve_ik(plant, plant_context_target, frame_E, X_WE_target_desired)

        print("\n Test IK sur une target :")
        print("Target choisie :", plant.GetModelInstanceName(target_test.model_instance()))
        print("Position target :", p_target)
        print("Position désirée main au-dessus de la target :", p_des_target)

        if q_target_place is not None:
            print("IK target réussie")
            print("q_target_place =", q_target_place)
        else:
            print("IK target échouée")

        # -----------------------------
        # Simulation
        # -----------------------------
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()

        sim_time = 10.0

        meshcat.StartRecording()
        simulator.AdvanceTo(sim_time)
        meshcat.PublishRecording()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
run_simulation(sim_time_step=0.001)
