import numpy as np
import matplotlib.pyplot as plt

def get_statistics(env) -> dict:
    
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
    plot_load_statistics(env)
    for key, value in get_statistics(env).items():
        print(f"{key}: {value}")
        

def plot_load_statistics(env, save_path="load_vs_usage.png"):
    # total_inflexible_load: shape (T,) or (T, D), we reduce to (T,)
    total_inflexible_load = np.sum(env.tr_inflexible_loads, axis=0)
    
    # current_power_usage: assumed shape (T,)
    current_power_usage = env.current_power_usage

    # Ensure both arrays are 1D and aligned
    total_inflexible_load = np.sum(total_inflexible_load, axis=-1) if total_inflexible_load.ndim > 1 else total_inflexible_load
    assert total_inflexible_load.shape == current_power_usage.shape, "Shape mismatch"

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(total_inflexible_load, label="Total Inflexible Load", linewidth=2)
    plt.plot(current_power_usage, label="Current Power Usage", linewidth=2)

    # Emphasize when usage occurs during low inflexible load
    low_load_threshold = np.percentile(total_inflexible_load, 25)
    plt.fill_between(
        x=np.arange(len(total_inflexible_load)),
        y1=current_power_usage,
        where=total_inflexible_load < low_load_threshold,
        color='green',
        alpha=0.2,
        label="Usage During Low Load"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Power (kW or normalized)")
    plt.title("Current Power Usage vs. Total Inflexible Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
