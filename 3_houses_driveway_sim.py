import os
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

# Change current working directory to EV2Gym submodule directory
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/EV2Gym")

config_file = "example_config_files/3_houses.yaml"

# Initialize the environment
env = EV2Gym(config_file=config_file, save_replay=True, save_plots=True)
state, _ = env.reset()

# Create Agents
agent = V2GProfitMaxOracle(env, verbose=True)  # optimal solution
agent = ChargeAsFastAsPossible()  # heuristic

for t in range(env.simulation_length):
    actions = agent.get_action(env)  # get action from the agent/ algorithm
    new_state, reward, done, truncated, stats = env.step(actions)
