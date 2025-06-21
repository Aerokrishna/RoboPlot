from roboplot import RoboPlot, Velocity, Pose
import numpy as np

# Setup
num_bots = 3
initial_poses = np.array([[2.5, 2.5, 1.57], [1.5, 1.5, 0], [3.0, 3.0, 3.14]])
velocities = [Velocity(0.5, 0, 1.0), Velocity(0.2, 0, 0.5), Velocity(0.7, 0, -0.5)]
obstacles = [Pose(0.5, 0.5, 0.0), Pose(2.0, 2.0, 0.0), Pose(0.7, 0.7, 0.0)]


sim = RoboPlot(num_bots=num_bots, obstacles=obstacles, model="diff", initial_poses=initial_poses)
sim.simulate(velocities, dt=0.1, mode="live")

t = 0
try:
    while t < 1000:
        for i in range(num_bots):
            velocities[i].vx = 0.5 + 0.01 * t
            velocities[i].omega = 1.0 + 0.01 * t

            print("BOT", sim.feedback(i)[0], "POSE", sim.feedback(i)[1].to_array())

        sim.update_once()
        t += 1
except KeyboardInterrupt:
    print("Simulation interrupted by user.")
