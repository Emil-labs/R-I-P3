import os
import shutil
import numpy as np
import pydot
from IPython.display import SVG, display
import matplotlib.pyplot as plt
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
from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.trajectories import PiecewisePolynomial


def maybe_save_block_diagram(diagram, image_path):
    """Save the diagram image when Graphviz is installed."""
    if shutil.which("dot") is None:
        print("Skipping block diagram export because Graphviz 'dot' was not found in PATH.")
        return

    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    graph.write_png(image_path)
    print(f"Block diagram saved as: {image_path}")


# Start the visualizer and clean up previous instances
meshcat = StartMeshcat()
meshcat.Delete()
meshcat.DeleteAddedControls()

# Set the path to your robot model:
robot_path = os.path.join(
    "..", "models", "objects_scenes", "project_03_shape_formation_t_world.sdf"
)

def plot_joint_tracking(logger_state, logger_traj, simulator_context, num_joints=9):
    """
    Plot actual vs reference joint positions and velocities from logs.
    """
    log_state = logger_state.FindLog(simulator_context)
    log_traj = logger_traj.FindLog(simulator_context)

    time = log_state.sample_times()
    q_actual = log_state.data()[:num_joints, :]
    qdot_actual = log_state.data()[num_joints:, :]

    q_ref = log_traj.data()[:num_joints, :]
    qdot_ref = log_traj.data()[num_joints:, :]

    # --- Joint positions ---
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i in range(7):
        axes[i].plot(time, q_actual[i, :], label='q_actual')
        axes[i].plot(time, q_ref[i, :], '--', label='q_ref')
        axes[i].set_ylabel(f'Joint {i+1} [rad]')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_ylim(-2.5, 2.5)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Joint Positions: Actual vs Reference')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # --- Joint velocities ---
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i in range(7):
        axes[i].plot(time, qdot_actual[i, :], label='qdot_actual')
        axes[i].plot(time, qdot_ref[i, :], '--', label='qdot_ref')
        axes[i].set_ylabel(f'Joint {i+1} [rad/s]')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_ylim(-1.0, 1.0)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Joint Velocities: Actual vs Reference')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######   
######################################################################################################

class Controller(LeafSystem):
    def __init__(self, plant,joint_names):
        super().__init__()

        self.joint_names = joint_names

        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()
        self.nu = plant.num_actuators()

        # Jaco joints and their indices in q and v
        self.joints = [plant.GetJointByName(name) for name in joint_names]
        self.pos_indices = [j.position_start() for j in self.joints]
        self.vel_indices = [j.velocity_start() for j in self.joints]

        self.n_panda = len(self.joints)

        # Full state from plant: x = [q; v]
        self._current_state_port = self.DeclareVectorInputPort(
            name="Current_state",
            size=self.nq + self.nv,
        )

        # Desired positions only for the selected Jaco joints
        self._desired_state_port = self.DeclareVectorInputPort(
            name="Desired_state",
            size=88,
        )

        # PD+G gains (Kp and Kd)
        self.Kp_ = np.array([120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0, 120, 120])
        self.Kd_ = 3*np.array([8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 2.0, 5, 5])

        # Store plant and context for dynamics calculations
        self.plant, self.plant_context_ad = plant, plant.CreateDefaultContext()

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(self.nv)  # 9 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1/1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u) # Call the Update method defined below.

    def compute_tau_u(self, context, discrete_state):
        # Evaluate the input ports
        x = self._current_state_port.Eval(context)  # [q; v]
        q_des = self._desired_state_port.Eval(context)  # desired positions for Jaco joints

        # Safety: avoid NaN propagation
        if not np.all(np.isfinite(x)):
            tau_zero = np.zeros(self.nv)
            discrete_state.get_mutable_vector().SetFromVector(tau_zero)
            return

        q = x[:self.nq]
        v = x[self.nq:]

        # Set current state in plant context
        self.plant.SetPositions(self.plant_context_ad, q)
        self.plant.SetVelocities(self.plant_context_ad, v)

        # Gravity forces for the current state (generalized forces, size = nv)
        gravity = -self.plant.CalcGravityGeneralizedForces(self.plant_context_ad)

        # Initialize tau for all DOFs
        tau = np.zeros(self.nv)

        # PD + gravity only on selected Jaco joints
        for i in range(self.n_panda):
            qi_idx = self.pos_indices[i]
            vi_idx = self.vel_indices[i]

            position_error = q_des[i] - q[qi_idx]
            velocity = v[vi_idx]

            tau[qi_idx] = (
                    self.Kp_[i] * position_error
                    - self.Kd_[i] * velocity
                    + gravity[vi_idx]
            )

        # Optional: torque saturation (for safety)
        tau_max = 60.0
        tau = np.clip(tau, -tau_max, tau_max)

        # Update the output discrete state
        discrete_state.get_mutable_vector().SetFromVector(tau)


######################################################################################################
#                     ########Define Trajectory Generator as a LeafSystem #######   
######################################################################################################

class JointSpaceTrajectorySystem(LeafSystem):
    def __init__(self, q_start, q_goal, v_max, a_max):
        super().__init__()
        self.q_start = np.array(q_start)
        self.q_goal  = np.array(q_goal)
        self.v_max   = np.broadcast_to(v_max, self.q_start.shape)
        self.a_max   = np.broadcast_to(a_max, self.q_start.shape)
        self.n = len(q_start)
        # Precompute motion profiles for each joint.
        # This sets up the timing and kinematic parameters (acceleration time,
        # flat time, total duration, etc.) used later to evaluate q_ref(t) and qd_ref(t).
        self._compute_profiles()
        self.DeclareVectorOutputPort("joint_ref", BasicVector(2*self.n), self._output_reference)

    def _compute_profiles(self):
        """
        Precompute motion parameters for each joint.

        The trapezoidal velocity profile consists of three phases:
        1. Acceleration with constant a_max
        2. Constant velocity at v_max (flat section)
        3. Deceleration with constant -a_max

        However, depending on the motion distance (Δq) and limits (v_max, a_max),
        the profile can take two shapes:

            • **Trapezoidal** → when the joint reaches v_max before decelerating.
            • **Triangular**  → when the joint does not reach v_max; 
                                acceleration and deceleration phases overlap.

        This method computes the timing parameters (t_acc, t_flat, T_total)
        for each joint and ensures all joints are synchronized by using
        the maximum duration among them.
        """
        self.profiles = []
        self.duration = 0.0

        for i in range(self.n):
            q0, qf = self.q_start[i], self.q_goal[i]
            dq = qf - q0
            s = np.sign(dq) if dq != 0 else 1.0
            dq_abs = abs(dq)
            v = self.v_max[i]
            a = self.a_max[i]

            # Acceleration time and peak velocity
            t_acc = v / a if a > 0 else 0.0

            # Check for triangular vs. trapezoidal
            if dq_abs < a * t_acc**2:
                # Triangular (no constant-velocity phase)
                t_acc = np.sqrt(dq_abs / max(a, 1e-9))
                t_flat = 0.0
                v_peak = a * t_acc
            else:
                # Trapezoidal (with constant velocity)
                v_peak = v
                t_flat = (dq_abs - a * t_acc**2) / max(v, 1e-9)

            T_trap = 2 * t_acc + t_flat
            self.duration = max(self.duration, T_trap)

            self.profiles.append(
                {
                    "q0": q0,
                    "dq": dq,
                    "s": s,
                    "t_acc": t_acc,
                    "t_flat": t_flat,
                    "T": T_trap,
                    "a": a,
                    "v_peak": v_peak,
                }
            )

    def _eval_trapezoid(self, t, p):
        """
        Compute joint position and velocity for a trapezoidal velocity profile.

        Each joint moves following a constant-acceleration, constant-velocity,
        constant-deceleration pattern. The parameter 's' ∈ {+1, -1} represents
        the direction of motion (sign of Δq = q_goal - q_start).

        Motion phases and equations:
            1. Acceleration (0 ≤ t < t_acc)
            q = q0 + s·½·a·t²
            q̇ = s·a·t

            2. Constant velocity (t_acc ≤ t < t_acc + t_flat)
            q = q0 + s·[½·a·t_acc² + v_peak·(t - t_acc)]
            q̇ = s·v_peak

            3. Deceleration (t_acc + t_flat ≤ t < T)
            q = q0 + s·[½·a·t_acc² + v_peak·t_flat
                            + v_peak·t_d - ½·a·t_d²]
            q̇ = s·(v_peak - a·t_d)
            where t_d = t - (t_acc + t_flat)

        Total duration:  T = 2·t_acc + t_flat
        """
        # Extract parameters for this joint
        q0, s, a, t_acc, t_flat, T, dq = (
            p["q0"], p["s"], p["a"], p["t_acc"], p["t_flat"], p["T"], p["dq"]
        )

        if t <= 0:
            q, qd = q0, 0.0
        elif t < t_acc:  # accelerating
            q = q0 + s * 0.5 * a * t**2
            qd = s * a * t
        elif t < t_acc + t_flat:  # constant velocity
            q = q0 + s * (0.5 * a * t_acc**2 + p["v_peak"] * (t - t_acc))
            qd = s * p["v_peak"]
        elif t < T:  # decelerating
            td = t - (t_acc + t_flat)
            q = (q0 + s * (0.5 * a * t_acc**2 + p["v_peak"] * t_flat +
                        p["v_peak"] * td - 0.5 * a * td**2))
            qd = s * (p["v_peak"] - a * td)
        else:  # end of motion
            q, qd = q0 + p["dq"], 0.0
        return q, qd
    
    def _output_reference(self, context, output):
        """Compute [q_ref, qd_ref] at current simulation time."""
        t = context.get_time()
        q_ref, qd_ref = np.zeros(self.n), np.zeros(self.n)
        for i, p in enumerate(self.profiles):
            q_ref[i], qd_ref[i] = self._eval_trapezoid(t, p)
        output.SetFromVector(np.hstack([q_ref, qd_ref]))


######################################################################################################
#                             ########IK Solver #######
######################################################################################################

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
    theta_bound = 1e-2  # radians
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),  # world frame
        R_AbarA=X_WE_desired.rotation(),  # desired orientation
        frameBbar=frame_E,  # end-effector frame
        R_BbarB=RotationMatrix(),  # current orientation
        theta_bound=theta_bound  # allowable deviation
    )

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


######################################################################################################

# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):   
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)
    model_instances = parser.AddModelsFromUrl("file://" + os.path.abspath(robot_path))

    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2"
    ]
    q_start = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    # Find your robot (by name is safest)
    idx = 0
    for name, value in zip(joint_names, q_start):
        joint = plant.GetJointByName(name)
        if idx <7:
            joint.set_default_angle(value)
        else:
            joint.set_default_translation(value)
        idx += 1

    plant.Finalize()

    joints = [plant.GetJointByName(name) for name in joint_names]
    pos_indices = [j.position_start() for j in joints]
    vel_indices = [j.velocity_start() for j in joints]

    plant_context = plant.CreateDefaultContext()
    frame_E = plant.GetFrameByName("panda_hand")

    q_start_full = plant.GetPositions(plant_context)
    q_start_jaco = np.array([q_start_full[i] for i in pos_indices])
    print(q_start_jaco)
    print(q_start_full)

    plant.SetPositions(plant_context, q_start_full)

    X_WE_desired = RigidTransform(
        RollPitchYaw(np.pi, 0, 0),
        [0.64, 0.18, 0.455]
    )

    q_target = [-0.5, -1.5, 0.5, -1.356, 0.5, 0.5, 0.0, 0.0, 0.0]

    q_target = solve_ik(plant, plant_context, frame_E, X_WE_desired)
    print("ttttt:",q_target)
    print(len(q_target))

    print(plant.GetDefaultPositions())
    print(len(plant.GetDefaultPositions()))
    #q_target_full = plant.GetPositions(plant_context)
    #q_target_full[:9] = q_target
    q_target_full = q_target

    print(q_target_full)


    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD+G controller to regulate the robot
    controller = builder.AddNamedSystem("PD+G controller", Controller(plant,joint_names))

    # Create a constant source for desired positions
    traj_system = builder.AddNamedSystem(
        "Trajectory Generator",
        JointSpaceTrajectorySystem(
            q_start=q_start_full,
            q_goal=q_target_full,
            v_max=0.2,   # rad/s
            a_max=2.0,    # rad/s²
        )
    )
    # des_pos = builder.AddNamedSystem("Desired position", ConstantVectorSource(q_target))

    # Connect systems: plant outputs to controller inputs, and vice versa
    builder.Connect(
        plant.get_state_output_port(),
        controller.get_input_port(0),  # "Current_state"
    )
    builder.Connect(controller.GetOutputPort("tau_u"), plant.GetInputPort("applied_generalized_force"))
    # builder.Connect(des_pos.get_output_port(), controller.GetInputPort("Desired_state"))
    builder.Connect(traj_system.get_output_port(0), controller.GetInputPort("Desired_state"))


    logger_state = LogVectorOutput(plant.get_state_output_port(), builder)
    logger_state.set_name("State logger")

    logger_traj = LogVectorOutput(traj_system.get_output_port(0), builder)
    logger_traj.set_name("Trajectory logger")

    # Build and return the diagram
    diagram = builder.Build()
    return diagram, logger_state, logger_traj, q_target_full

# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram, logger_state, logger_traj, q_target = create_sim_scene(sim_time_step)
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    simulator.Initialize()
    simulator.set_target_realtime_rate(2.)

    maybe_save_block_diagram(diagram, "figures/block_diagram_04_traj.png")
    
    # Run simulation and record for replays in MeshCat
    meshcat.StartRecording()
    simulator.AdvanceTo(6.0)  # Adjust this time as needed
    meshcat.PublishRecording()

    # At the end of the simulation
    plot_joint_tracking(logger_state, logger_traj, simulator.get_context())

# Run the simulation with a specific time step. Try gradually increasing it!
run_simulation(sim_time_step=0.002)
