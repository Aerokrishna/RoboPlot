import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

    def update(self, velocity: Velocity, dt: float):
        raise NotImplementedError

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
    def __init__(self, num_bots, arena_size=(4, 4), model="holonomic", initial_poses=None):
        self.num_bots = num_bots
        self.arena_size = arena_size
        self.model = model
        self.robots = []

        if initial_poses is None:
            initial_poses = np.random.uniform(0, arena_size[0], (num_bots, 2))

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

    def simulate(self, get_velocity_fn, dt=0.1):
        def update(frame):
            for i, robot in enumerate(self.robots):
                vel = get_velocity_fn(i, frame)
                robot.update(vel, dt)

            xs = [r.pose.x for r in self.robots]
            ys = [r.pose.y for r in self.robots]
            us = [np.cos(r.pose.theta) for r in self.robots]
            vs = [np.sin(r.pose.theta) for r in self.robots]

            self.scat.set_offsets(np.column_stack((xs, ys)))
            self.quivers.set_offsets(np.column_stack((xs, ys)))
            self.quivers.set_UVC(us, vs)

            return self.scat, self.quivers

        self.ani = FuncAnimation(self.fig, update, frames=100, interval=0.1 * 1000, blit=True)
        plt.show()
