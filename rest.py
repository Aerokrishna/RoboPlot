from roboplot import RoboPlot, Velocity
import numpy as np
def controller_fn(bot_id, frame):
    if bot_id == 0:
        return Velocity(0.5, 0, -0.7)  # Fast forward, turning left
    elif bot_id == 1:
        return Velocity(0.2, 0, 0.3)   # Slow forward, turning right

sim = RoboPlot(num_bots=2, model="diff", initial_poses=np.array([[2.5, 2.5, 0], [1.5, 1.5, 0]]))
sim.simulate(controller_fn)
