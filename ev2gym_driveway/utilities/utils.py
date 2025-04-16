# This file contains support functions for the EV City environment.

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from typing import List

from ev2gym_driveway.models.ev import EV


def spawn_single_EV(
    env,
    scenario,
    cs_id,
    port,
) -> EV:
    """
    This function spawns a single EV and returns it
    """

    # required energy independent of time of arrival
    # required_energy = env.df_energy_demand[scenario].iloc[np.random.randint(
    #     0, 100, size=1)].values[0]  # kWh

    # required_energy_mean = env.df_req_energy[
    #     (env.df_req_energy['Arrival Time'] == arrival_time)
    # ][scenario].values[0]

    # required_energy = np.random.normal(
    #     required_energy_mean, 0.5*required_energy_mean)  # kWh

    # if required_energy < 5:
    required_energy = np.random.randint(5, 10)

    if env.heterogeneous_specs:
        sampled_ev = np.random.choice(
            list(env.ev_specs.keys()), p=env.normalized_ev_registrations
        )
        battery_capacity = env.ev_specs[sampled_ev]["battery_capacity"]
    else:
        battery_capacity = env.config["ev"]["battery_capacity"]

    if battery_capacity < required_energy:
        initial_battery_capacity = np.random.randint(1, battery_capacity)
    else:
        initial_battery_capacity = battery_capacity - required_energy

    if initial_battery_capacity > env.config["ev"]["desired_capacity"]:
        initial_battery_capacity = np.random.randint(1, battery_capacity)

    if (
        initial_battery_capacity < env.config["ev"]["min_battery_capacity"]
        and battery_capacity > 2 * env.config["ev"]["min_battery_capacity"]
    ):
        initial_battery_capacity = env.config["ev"]["min_battery_capacity"]

    if "transition_soc_multiplier" in env.config["ev"]:
        transition_soc_multiplier = env.config["ev"]["transition_soc_multiplier"]
    else:
        transition_soc_multiplier = 1

    min_emergency_battery_capacity = env.config["ev"]["min_emergency_battery_capacity"]

    if min_emergency_battery_capacity > battery_capacity:
        min_emergency_battery_capacity = 0.7 * battery_capacity

    if env.heterogeneous_specs:
        # get charge efficiency from env.ev_specs dict
        # if there is key charge_efficiency_v
        if "3ph_ch_efficiency" in env.ev_specs[sampled_ev]:
            charge_efficiency_v = env.ev_specs[sampled_ev]["3ph_ch_efficiency"]
            current_levels = env.ev_specs[sampled_ev]["ch_current"]
            assert len(charge_efficiency_v) == len(current_levels)
            assert all([0 <= x <= 100 for x in charge_efficiency_v])

            # make a dict with charge leves kay and charge efficiency value
            charge_efficiency = dict(zip(current_levels, charge_efficiency_v))

            for i in range(0, 101):
                if i not in charge_efficiency or charge_efficiency[i] == 0:
                    nonzero_keys = [k for k, v in charge_efficiency.items() if v != 0]
                    if nonzero_keys:
                        closest = min(nonzero_keys, key=lambda x: abs(x - i))
                        charge_efficiency[i] = charge_efficiency[closest]

            discharge_efficiency = charge_efficiency.copy()

        else:
            charge_efficiency = np.round(
                1 - (np.random.rand() + 0.00001) / 20, 3
            )  # [0.95-1]
            discharge_efficiency = np.round(
                1 - (np.random.rand() + 0.00001) / 20, 3
            )  # [0.95-1]

        return EV(
            id=port,
            battery_capacity_at_arrival=initial_battery_capacity,
            max_ac_charge_power=env.ev_specs[sampled_ev]["max_ac_charge_power"],
            max_dc_charge_power=env.ev_specs[sampled_ev]["max_dc_charge_power"],
            max_discharge_power=-env.ev_specs[sampled_ev]["max_dc_discharge_power"],
            min_emergency_battery_capacity=min_emergency_battery_capacity,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            transition_soc=np.round(
                0.9 - (np.random.rand() + 0.00001) / 5, 3
            ),  # [0.7-0.9]
            transition_soc_multiplier=transition_soc_multiplier,
            battery_capacity=battery_capacity,
            desired_capacity=env.config["ev"]["desired_capacity"] * battery_capacity,
            ev_phases=3,
            timescale=env.timescale,
        )
    else:
        raise NotImplementedError("Must Use Heterogeneous EV specs for EV spawner")


def EV_spawner_for_driveways(env) -> list[EV]:
    """
    Spawns 1 EV for each charging station
    Returns:
        EVs: list of EVs (1 for each charging station)
    """

    ev_list = []

    scenario = env.scenario

    ## WE SHOULD CHANGE THIS Probablyyyy
    for cs in env.charging_stations:
        assert (
            cs.n_ports == 1
        ), "Only one port per charging station is allowed for driveways"
        for port in range(cs.n_ports):
            ev = spawn_single_EV(env=env, scenario=scenario, cs_id=cs.id, port=port)

            if ev is None:
                raise ValueError("EV is None! Check the EV spawning function.")
            ev_list.append(ev)

    return ev_list


def median_smoothing(v, window_size) -> np.ndarray:
    smoothed_v = np.zeros_like(v)
    half_window = window_size // 2

    for i in range(len(v)):
        start = max(0, i - half_window)
        end = min(len(v), i + half_window + 1)
        smoothed_v[i] = np.median(v[start:end])

    return smoothed_v


def generate_power_setpoints(env) -> np.ndarray:
    """
    This function generates the power setpoints for the entire simulation using
    the list of EVs and the charging stations from the environment.

    It considers the ev SoC and teh steps required to fully charge the EVs.

    Returns:
        power_setpoints: np.ndarray

    """

    power_setpoints = np.zeros(env.simulation_length)
    # get normalized prices
    prices = abs(env.charge_prices[0])
    prices = prices / np.max(prices)

    required_energy_multiplier = 100 + env.config["power_setpoint_flexiblity"]

    min_cs_power = env.charging_stations[0].get_min_charge_power()
    max_cs_power = env.charging_stations[0].get_max_power()

    total_evs_spawned = 0
    for t in range(env.simulation_length):
        counter = total_evs_spawned
        for _, ev in enumerate(env.EVs_profiles[counter:]):
            if ev.time_of_arrival == t + 1:
                total_evs_spawned += 1

                required_energy = ev.battery_capacity - ev.battery_capacity_at_arrival
                required_energy = required_energy * required_energy_multiplier / 100
                min_power_limit = max(ev.min_ac_charge_power, min_cs_power)
                max_power_limit = min(ev.max_ac_charge_power, max_cs_power)

                # Spread randomly the required energy over the time of stay using the prices as weights
                shifted_load = np.random.normal(
                    loc=1 - prices[t + 2 : ev.time_of_departure],
                    scale=min(prices[t + 2 : ev.time_of_departure]),
                    size=ev.time_of_departure - t - 2,
                )
                # make shifted load positive
                shifted_load = np.abs(shifted_load)
                shifted_load = shifted_load / np.sum(shifted_load)
                shifted_load = shifted_load * required_energy * 60 / env.timescale

                # find power lower than min_power_limit and higher than max_power_limit
                step = 0
                while (
                    np.min(shifted_load[shifted_load != 0]) < min_power_limit
                    or np.max(shifted_load) > max_power_limit
                ):

                    if step > 10:
                        break

                    # print(f"Shifted load: {shifted_load}")
                    for i in range(len(shifted_load)):
                        if shifted_load[i] < min_power_limit and shifted_load[i] > 0:
                            load_to_shift = shifted_load[i]
                            shifted_load[i] = 0

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i + 1] += load_to_shift

                        elif shifted_load[i] > max_power_limit:
                            load_to_shift = shifted_load[i] - max_power_limit
                            shifted_load[i] = max_power_limit

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i + 1] += load_to_shift
                    step += 1

                power_setpoints[t + 2 : ev.time_of_departure] += shifted_load

            elif ev.time_of_arrival > t + 1:
                break

    multiplier = int(15 / env.timescale)
    if multiplier < 1:
        multiplier = 1
    return median_smoothing(power_setpoints, 5 * multiplier)


def calculate_charge_power_potential(env) -> float:
    """
    This function calculates the total charge power potential of all currently parked EVs for the current time step
    """

    power_potential = 0
    for cs in env.charging_stations:
        cs_power_potential = 0
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                if ev.get_soc() < 1:  # and ev.time_of_departure > env.current_step:
                    phases = min(cs.phases, ev.ev_phases)
                    ev_current = (
                        ev.max_ac_charge_power * 1000 / (math.sqrt(phases) * cs.voltage)
                    )
                    current = min(cs.max_charge_current, ev_current)
                    cs_power_potential += (
                        math.sqrt(phases) * cs.voltage * current / 1000
                    )

        max_cs_power = math.sqrt(cs.phases) * cs.voltage * cs.max_charge_current / 1000
        min_cs_power = math.sqrt(cs.phases) * cs.voltage * cs.min_charge_current / 1000

        if cs_power_potential > max_cs_power:
            power_potential += max_cs_power
        elif cs_power_potential < min_cs_power:
            power_potential += 0
        else:
            power_potential += cs_power_potential

    return power_potential
