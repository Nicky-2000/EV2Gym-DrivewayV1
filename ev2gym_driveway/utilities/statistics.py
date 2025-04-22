import numpy as np


def get_statistics(env) -> dict:
    total_inflexible_load = np.sum( env.tr_inflexible_loads, axis=0)
    
    # Get total Money Spent and Earned charging and discharging
    total_money_spent_charging = np.array(
        [household.total_money_spent_charging for household in env.households]
    ).sum()
    total_money_earned_discharging = np.array(
        [household.total_money_earned_discharging for household in env.households]
    ).sum()
    
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]
    ).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]
    ).sum()

    # get transformer overload from env.tr_overload
    total_transformer_overload = np.array(env.tr_overload).sum()
    
    # TODO: FIX BATTERY DEGRADATION... 
    # We need to calculate the total time the car was charging and use this info

    # calculate total batery degradation
    evs = [household.ev for household in env.households]
    
    # battery_degradation = np.array(
    #     [np.array(ev.get_battery_degradation()).reshape(-1) for ev in evs]
    # )
    # if len(battery_degradation) == 0:
    #     battery_degradation = np.zeros((1, 2))
    # battery_degradation_calendar = battery_degradation[:, 0].sum()
    # battery_degradation_cycling = battery_degradation[:, 1].sum()
    # battery_degradation = battery_degradation.sum()

    # total_steps_min_emergency_battery_capacity_violation = 0
    # energy_user_satisfaction = np.zeros((len(env.EVs)))
    # for i, ev in enumerate(env.EVs):
    #     e_actual = ev.current_capacity
    #     e_max = ev.max_energy_AFAP
    #     energy_user_satisfaction[i] = (e_actual / e_max) * 100
    #     total_steps_min_emergency_battery_capacity_violation += (
    #         ev.min_emergency_battery_capacity_metric
    #     )
    
    stats = {
        "total_energy_charged": total_energy_charged,
        "total_energy_discharged": total_energy_discharged,
        "total_money_spent_charging": total_money_spent_charging,
        "total_money_earned_discharging": total_money_earned_discharging,
        #  'average_user_satisfaction': average_user_satisfaction,
        # "energy_user_satisfaction": np.mean(energy_user_satisfaction),
        # "std_energy_user_satisfaction": np.std(energy_user_satisfaction),
        # "min_energy_user_satisfaction": np.min(energy_user_satisfaction),
        # "total_steps_min_emergency_battery_capacity_violation": total_steps_min_emergency_battery_capacity_violation,
        "total_transformer_overload": total_transformer_overload,
        # "battery_degradation": battery_degradation,
        # "battery_degradation_calendar": battery_degradation_calendar,
        # "battery_degradation_cycling": battery_degradation_cycling,
        "total_reward": env.total_reward,
    }
    
    
    return stats


def print_statistics(env) -> None:
    # We want a plot of the inflexible loads and the flexible loads. 
    # and the total load profile
    for key, value in get_statistics(env).items():
        print(f"{key}: {value}")