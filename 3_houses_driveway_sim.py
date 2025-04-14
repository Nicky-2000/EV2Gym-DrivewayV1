import os
from ev2gym_driveway.models.ev2gym_driveway_env import EV2GymDriveway
from ev2gym_driveway.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym_driveway.baselines.heuristics import ChargeAsFastAsPossible

# Change current working directory to EV2Gym submodule directory
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/ev2gym_driveway")

config_file = "example_config_files/3_houses.yaml"

# Initialize the environment
env = EV2GymDriveway(config_file=config_file)
state, _ = env.reset()

# Create Agents
agent = ChargeAsFastAsPossible()  # heuristic

for t in range(env.simulation_length):
    actions = agent.get_action(env)  # get action from the agent/ algorithm
    new_state, reward, done, truncated, stats = env.step(actions)
