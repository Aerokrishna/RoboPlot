import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# essential classes Pose, Velocity, Robot
class Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def to_array(self):
        return np.array([self.x, self.y, self.theta])

class Velocity:
    def __init__(self, vx=0.0, vy=0.0, omega=0.0):
        self.vx = vx
        self.vy = vy
        self.omega = omega

    def to_array(self):
        return np.array([self.vx, self.vy, self.omega])

class Robot:
    def __init__(self, pose: Pose):
        self.pose = pose
        self.velocity = Velocity(0.0, 0.0, 0.0)

    def update(self, velocity: Velocity, dt: float):
        raise NotImplementedError

# robot models which take in a velocity and update pose accordingly
class HolonomicRobot(Robot):
    def update(self, velocity: Velocity, dt: float):
        self.pose.x += velocity.vx * dt
        self.pose.y += velocity.vy * dt
        self.pose.theta += velocity.omega * dt

class DifferentialDriveRobot(Robot):
    def update(self, velocity: Velocity, dt: float):
        self.pose.x += velocity.vx * np.cos(self.pose.theta) * dt
        self.pose.y += velocity.vx * np.sin(self.pose.theta) * dt
        self.pose.theta += velocity.omega * dt

class RoboPlot:
    def __init__(self, num_bots, obstacles : list[Pose], arena_size=(4, 4), model="holonomic", initial_poses=None):

        if not num_bots == len(initial_poses):
            raise ValueError("Number of bots must match the length of initial_poses or be zero for random initialization.")
        
        self.num_bots = num_bots
        self.arena_size = arena_size
        self.model = model
        self.robots = []

        if initial_poses is None:
            initial_poses = np.random.uniform(0, arena_size[0], (num_bots, 2))

        # convert initial poses to Robot Models
        for pose in initial_poses:
            p = Pose(pose[0], pose[1], pose[2])  # Initial theta = 0
            if model == "holonomic":
                self.robots.append(HolonomicRobot(p))
            elif model == "diff":
                self.robots.append(DifferentialDriveRobot(p))
            else:
                raise ValueError("Unknown model type")

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scat = self.ax.scatter([r.pose.x for r in self.robots],
                                    [r.pose.y for r in self.robots],
                                    s=100, c='blue')

        # Plot obstacles as black points if provided
        if obstacles:
            obs_xs = [obs.x for obs in obstacles]
            obs_ys = [obs.y for obs in obstacles]
            self.ax.scatter(obs_xs, obs_ys, s=100, c='black', marker='o', label='Obstacle')

        # Add quivers for orientation arrows
        self.quivers = self.ax.quiver(
            [r.pose.x for r in self.robots],
            [r.pose.y for r in self.robots],
            [np.cos(r.pose.theta) for r in self.robots],
            [np.sin(r.pose.theta) for r in self.robots],
            angles='xy', scale_units='xy', scale=5, color='red'
        )

        self.ax.set_xlim(0, arena_size[0])
        self.ax.set_ylim(0, arena_size[1])
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True)
        self.ax.set_title(f"{num_bots} Robots in Arena")

        # Store obstacles
        self.obstacles = obstacles

    # function which updates the robots' positions and orientations and animates the simulation
    def simulate(self, velocities, dt=0.1, steps=None, mode="live"):
        """
        Simulate the robots. Two modes:
        - 'live': non-blocking, call this once, and then repeatedly call `sim.update_once(dt)` and `plt.pause(dt)`
        - 'auto': internally uses FuncAnimation (blocking unless threaded)
        """
        if len(velocities) != self.num_bots:
            raise ValueError("Velocities list length must match number of robots")

        self._dt = dt
        self._velocities = velocities

        def update(frame=None):
            for i, robot in enumerate(self.robots):
                vel = self._velocities[i]
                robot.update(vel, self._dt)
                robot.velocity = vel
            xs = [r.pose.x for r in self.robots]
            ys = [r.pose.y for r in self.robots]
            us = [np.cos(r.pose.theta) for r in self.robots]
            vs = [np.sin(r.pose.theta) for r in self.robots]
            self.scat.set_offsets(np.column_stack((xs, ys)))
            self.quivers.set_offsets(np.column_stack((xs, ys)))
            self.quivers.set_UVC(us, vs)
            return self.scat, self.quivers

        self._update_callback = update

        if mode == "live":
            print("[RoboPlot] Running in live mode. Use `sim.update_once()` and `plt.pause()` in your loop.")
            plt.ion()
            self.fig.show()

        # elif mode == "auto":
        #     print("[RoboPlot] Running in auto animation mode.")
        #     self.ani = FuncAnimation(self.fig, update, frames=steps or 100, interval=dt * 1000, blit=True)
        #     plt.show()

        else:
            raise ValueError("Unknown mode. Use 'live'.")

    def update_once(self):
        """Call this in live mode to update robot positions and stop robots if needed"""
        if self._update_callback:
            # Check for collisions between robots
            stopped = set()
            for i, robot in enumerate(self.robots):
                # Check collision with other robots
                for j, other in enumerate(self.robots):
                    if i != j:
                        distance = np.sqrt((robot.pose.x - other.pose.x) ** 2 + (robot.pose.y - other.pose.y) ** 2)
                        if distance < 0.1:
                            stopped.add(i)
                            stopped.add(j)
                # Check collision with obstacles
                if hasattr(self, 'obstacles') and self.is_too_close_to_obstacle(robot, self.obstacles):
                    stopped.add(i)
                # Check collision with edges
                if self.is_too_close_to_edge(robot):
                    stopped.add(i)
            # Stop robots that are too close
            for i in stopped:
                self._velocities[i].vx = 0.0
                self._velocities[i].vy = 0.0
                self._velocities[i].omega = 0.0
            self._update_callback()
            plt.pause(0.1)

    def feedback(self, bot_id):
        """
        Returns a list of tuples (bot_id, Pose, Velocity) for all robots.
        Each tuple contains:
            - bot_id (int): The index of the robot
            - Pose: The current pose of the robot
            - Velocity: The last velocity applied to the robot (if available, else zeros)
        """
        feedback_list = [bot_id, self.robots[bot_id].pose, self.robots[bot_id].velocity]
        # for i, robot in enumerate(self.robots):
        #     feedback_list.append((i, robot.pose, robot.velocity))
        return feedback_list

    def is_collision(self, bot_id):
        """
        Check if the robot with the given bot_id is colliding with any other robot.
        Returns True if there is a collision, False otherwise.
        """
        target_robot = self.robots[bot_id]
        for i, robot in enumerate(self.robots):
            if i != bot_id:
                distance = np.sqrt((target_robot.pose.x - robot.pose.x) ** 2 +
                                   (target_robot.pose.y - robot.pose.y) ** 2)
                if distance < 0.1:
                    return True
        return False

    def is_too_close_to_obstacle(self, robot, obstacles, threshold=0.15):
        for obs in obstacles:
            distance = np.sqrt((robot.pose.x - obs.x) ** 2 + (robot.pose.y - obs.y) ** 2)
            if distance < threshold:
                return True
        return False

    def is_too_close_to_edge(self, robot, margin=0.1):
        if (robot.pose.x < margin or robot.pose.x > self.arena_size[0] - margin or
            robot.pose.y < margin or robot.pose.y > self.arena_size[1] - margin):
            return True
        return False