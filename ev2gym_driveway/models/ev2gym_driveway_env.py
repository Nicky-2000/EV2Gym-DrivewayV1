"""
This file contains the EVCity class, which is used to represent the environment of the city.
The environment is a gym environment and can be also used with the OpenAI gym standards and baselines.
The environment an also be used for standalone simulations without the gym environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import datetime
import pickle
import os
import random
import yaml

from ev2gym_driveway.utilities.utils import (
    EV_spawner_for_driveways,
    calculate_charge_power_potential,
)
from ev2gym_driveway.utilities.statistics import get_statistics, print_statistics
from ev2gym_driveway.utilities.loaders import (
    load_ev_spawn_scenarios,
    load_transformers,
    load_ev_charger_profiles,
    load_electricity_prices,
    load_weekly_EV_profiles,
)
from ev2gym_driveway.models.Household import Household
from ev2gym_driveway.models.ev_charger import EV_Charger
from ev2gym_driveway.models.ev import EV


from ev2gym_driveway.rl_agent.reward import ProfitMax_TrPenalty_UserIncentives
from ev2gym_driveway.rl_agent.state import V2G_profit_max


class EV2GymDriveway(gym.Env):

    def __init__(
        self,
        config_file=None,
        seed=None,
        state_function=V2G_profit_max,
        reward_function=ProfitMax_TrPenalty_UserIncentives,
        cost_function=None,  # cost function to use in the simulation
        # whether to empty the ports at the end of the simulation or not
        verbose=True,
    ):

        super(EV2GymDriveway, self).__init__()

        if verbose:
            print(f"Initializing EV2Gym environment...")

        # read yaml config file
        assert config_file is not None, "Please provide a config file!!!"
        self.config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

        self.verbose = verbose  # Whether to print the simulation progress or not


        self.reward_function = reward_function
        self.state_function = state_function
        self.cost_function = cost_function

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.tr_seed = self.config["tr_seed"]
        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        self.cs = self.config["number_of_charging_stations"]
        assert self.cs is not None, "Please provide the number of charging stations"

        self.number_of_ports_per_cs = self.config["number_of_ports_per_cs"]
        self.number_of_transformers = self.config["number_of_transformers"]
        self.timescale = self.config["timescale"]
        self.scenario = self.config["scenario"]
        self.simulation_length = int(self.config["simulation_length"])

        # Simulation time
        self.sim_date = datetime.datetime(
            self.config["year"],
            self.config["month"],
            self.config["day"],
            self.config["hour"],
            self.config["minute"],
        )
        self.sim_starting_date = self.sim_date

        self.sim_name = f"sim_" + f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

        self.heterogeneous_specs = self.config["heterogeneous_ev_specs"]
        self.stats = None

        # Set up the transformers
        self.cs_transformers = [*np.arange(self.number_of_transformers)] * (
            self.cs // self.number_of_transformers
        )

        self.cs_transformers += random.sample(
            [*np.arange(self.number_of_transformers)],
            self.cs % self.number_of_transformers,
        )
        random.shuffle(self.cs_transformers)

        # Instatiate Transformers
        self.transformers = load_transformers(self)
        self.n_transformers = len(self.transformers)
        for tr in self.transformers:
            tr.reset(step=0)

        # Instatiate Charging Stations
        self.charging_stations = load_ev_charger_profiles(self)
        for cs in self.charging_stations:
            cs.reset()

        # Calculate the total number of ports in the simulation
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]
        ).sum()

        # Load EV spawn scenarios
        load_ev_spawn_scenarios(self)

        # Spawn EVs
        self.EVs_for_driveways: list[EV] = EV_spawner_for_driveways(self)
        self.weekly_EV_profiles: list[dict] = load_weekly_EV_profiles(self.cs, self.timescale)

        # Initialize Households (Driveways)
        self.households: list[Household] = []
        for cs, ev, ev_profile in zip(
            self.charging_stations, self.EVs_for_driveways, self.weekly_EV_profiles
        ):
            household = Household(
                charging_station=cs,
                ev=ev,
                ev_weekly_profile=ev_profile,
                timescale=self.timescale,
                sim_starting_date=self.sim_starting_date
            )
            self.households.append(household)

        # Load Electricity prices for every charging station
        self.price_data = None
        self.charge_prices, self.discharge_prices = load_electricity_prices(self)

        self.current_power_usage = np.zeros(self.simulation_length)
        self.charge_power_potential = np.zeros(self.simulation_length)

        self.init_statistic_variables()

        # Variable showing whether the simulation is done or not
        self.done = False

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        high = np.ones([self.number_of_ports])
        if self.config["v2g_enabled"]:
            lows = -1 * np.ones([self.number_of_ports])
        else:
            lows = np.zeros([self.number_of_ports])
        self.action_space = spaces.Box(low=lows, high=high, dtype=np.float64)

        # Observation space: is a matrix of size ("Sum of all ports of all charging stations",n_features)
        obs_dim = len(self._get_observation())

        high = np.inf * np.ones([obs_dim])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

    def reset(self, seed=None, options=None, **kwargs):
        """Resets the environment to its initial state"""

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.tr_seed == -1:
            self.tr_seed = self.seed
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        self.current_step = 0
        self.stats = None
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()

        for tr in self.transformers:
            tr.reset(step=self.current_step)

        # TODO: Might need to reset the households

        # Reset the simulation date
        self.sim_date = self.sim_starting_date

        self.init_statistic_variables()

        return self._get_observation(), {}

    def init_statistic_variables(self):
        """
        Initializes the variables used for keeping simulation statistics
        """
        self.current_step = 0
        self.total_reward = 0

        self.previous_power_usage = self.current_power_usage
        self.current_power_usage = np.zeros(self.simulation_length)

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])

        self.tr_overload = np.zeros(
            [self.number_of_transformers, self.simulation_length]
        )

        self.tr_inflexible_loads = np.zeros(
            [self.number_of_transformers, self.simulation_length]
        )

        self.tr_solar_power = np.zeros(
            [self.number_of_transformers, self.simulation_length]
        )

        self.port_current = np.zeros(
            [self.number_of_ports, self.cs, self.simulation_length],
            dtype=np.float16,
        )
        self.port_current_signal = np.zeros(
            [self.number_of_ports, self.cs, self.simulation_length],
            dtype=np.float16,
        )

        self.port_energy_level = np.zeros(
            [self.number_of_ports, self.cs, self.simulation_length], dtype=np.float16
        )
        self.port_arrival = dict(
            {
                f"{j}.{i}": []
                for i in range(self.number_of_ports)
                for j in range(self.cs)
            }
        )

        self.done = False

    def step(self, actions):
        """'
        Takes an action as input and returns the next state, reward, and whether the episode is done
        Inputs:
            - actions: is a vector of size "Sum of all ports of all charging stations taking values in [-1,1]"
        Returns:
            - observation: is a matrix with the complete observation space
            - reward: is a scalar value representing the reward of the current step
            - done: is a boolean value indicating whether the episode is done or not
        """
        assert not self.done, "Episode is done, please reset the environment"

        if self.verbose:
            print("-" * 80)

        total_costs = 0
        total_invalid_action_punishment = 0
        self.departing_evs = []

        port_counter = 0

        # Reset current power of all transformers
        for tr in self.transformers:
            tr.reset(step=self.current_step)

        # Call step for each household
        for i, household in enumerate(self.households):
            cs: EV_Charger = household.charging_station
            n_ports = household.charging_station.n_ports

            assert n_ports == 1, "Only one port is supported for now"

            invalid_action_punishment = household.step(
                actions[port_counter : port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step],
                self.sim_date,
            )

            self.current_power_usage[self.current_step] += cs.current_power_output

            # Update transformer variables for this timestep
            self.transformers[cs.connected_transformer].step(
                cs.current_total_amps, cs.current_power_output
            )

            total_invalid_action_punishment += invalid_action_punishment

            port_counter += n_ports

        self._update_power_statistics()

        self.current_step += 1
        self._step_date()

        if self.current_step < self.simulation_length:
            self.charge_power_potential[self.current_step] = (
                calculate_charge_power_potential(self)
            )

        user_satisfaction_list = []
        reward = self._calculate_reward(
            total_costs, user_satisfaction_list, total_invalid_action_punishment
        )

        if self.cost_function is not None:
            cost = self.cost_function(
                self,
                total_costs,
                user_satisfaction_list,
                total_invalid_action_punishment,
            )
        else:
            cost = None

        return self._check_termination(reward, cost)

    def _check_termination(self, reward, cost):
        """Checks if the episode is done or any constraint is violated"""
        truncated = False
        action_mask = np.zeros(self.number_of_ports)
        
        # action mask is 1 if an EV is connected to the port
        for i, cs in enumerate(self.charging_stations):
            for j in range(cs.n_ports):
                if cs.evs_connected[j] is not None:
                    action_mask[i * cs.n_ports + j] = 1

        # Check if the episode is done or any constraint is violated
        if self.current_step >= self.simulation_length or (
            any(tr.is_overloaded() > 0 for tr in self.transformers)
        ):
            """Terminate if:
            - The simulation length is reached
            - Any user satisfaction score is below the threshold
            - Any charging station is overloaded
            Dont terminate when overloading if :
            - generate_rnd_game is True
            Carefull: if generate_rnd_game is True,
            the simulation might end up in infeasible problem
            """

            self.done = True
            self.stats = get_statistics(self)

            self.stats["action_mask"] = action_mask
            self.cost = cost

            if self.verbose:
                print_statistics(self)

                if any(tr.is_overloaded() for tr in self.transformers):
                    print(f"Transformer overloaded, {self.current_step} timesteps\n")
                else:
                    print(f"Episode finished after {self.current_step} timesteps\n")


            if self.cost_function is not None:
                return self._get_observation(), reward, True, truncated, self.stats
            else:
                return self._get_observation(), reward, True, truncated, self.stats
        else:
            stats = {
                "cost": cost,
                "action_mask": action_mask,
            }

            if self.cost_function is not None:
                return self._get_observation(), reward, False, truncated, stats
            else:
                return self._get_observation(), reward, False, truncated, stats


    def _update_power_statistics(self):
        """Updates the power statistics of the simulation"""

        # if not self.lightweight_plots:
        for tr in self.transformers:
            # self.transformer_amps[tr.id, self.current_step] = tr.current_amps
            self.tr_overload[tr.id, self.current_step] = tr.get_how_overloaded()
            self.tr_inflexible_loads[tr.id, self.current_step] = tr.inflexible_load[
                self.current_step
            ]
            self.tr_solar_power[tr.id, self.current_step] = tr.solar_power[
                self.current_step
            ]

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id, self.current_step] = cs.current_total_amps

            for port in range(cs.n_ports):
                self.port_current_signal[port, cs.id, self.current_step] = (
                    cs.current_signal[port]
                )
                ev = cs.evs_connected[port]
                if ev is not None:
                    # self.port_power[port, cs.id,
                    #                 self.current_step] = ev.current_energy
                    self.port_current[port, cs.id, self.current_step] = (
                        ev.actual_current
                    )

                    self.port_energy_level[port, cs.id, self.current_step] = (
                        ev.current_capacity / ev.battery_capacity
                    )


    def _step_date(self):
        """Steps the simulation date by one timestep"""
        self.sim_date = self.sim_date + datetime.timedelta(minutes=self.timescale)

    def _get_observation(self):

        return self.state_function(self)

    def set_cost_function(self, cost_function):
        """
        This function sets the cost function of the environment
        """
        self.cost_function = cost_function

    def set_reward_function(self, reward_function):
        """
        This function sets the reward function of the environment
        """
        self.reward_function = reward_function

    def _calculate_reward(
        self, total_costs, user_satisfaction_list, invalid_action_punishment
    ):
        """Calculates the reward for the current step"""

        reward = self.reward_function(
            self, total_costs, user_satisfaction_list, invalid_action_punishment
        )
        self.total_reward += reward

        return reward
