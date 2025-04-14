import numpy as np


def get_statistics(env) -> dict:
    # total_ev_served = np.array(
    #     [cs.total_evs_served for cs in env.charging_stations]).sum()
    # total_profits = np.array(
    #     [cs.total_profits for cs in env.charging_stations]).sum()
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]
    ).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]
    ).sum()
    average_user_satisfaction = np.array(
        [
            cs.get_avg_user_satisfaction()
            for cs in env.charging_stations
            if cs.total_evs_served > 0
        ]
    ).mean()

    # get transformer overload from env.tr_overload
    total_transformer_overload = np.array(env.tr_overload).sum()

    tracking_error = 0
    energy_tracking_error = 0
    power_tracker_violation = 0
    for t in range(env.simulation_length):
        # tracking_error += (min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                    env.current_power_usage[t])**2
        # energy_tracking_error += abs(min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                              env.current_power_usage[t])

        tracking_error += (env.power_setpoints[t] - env.current_power_usage[t]) ** 2
        energy_tracking_error += abs(
            env.power_setpoints[t] - env.current_power_usage[t]
        )

        if env.current_power_usage[t] > env.power_setpoints[t]:
            power_tracker_violation += (
                env.current_power_usage[t] - env.power_setpoints[t]
            )

    energy_tracking_error *= env.timescale / 60

    # calculate total batery degradation
    battery_degradation = np.array(
        [np.array(ev.get_battery_degradation()).reshape(-1) for ev in env.EVs]
    )
    if len(battery_degradation) == 0:
        battery_degradation = np.zeros((1, 2))
    battery_degradation_calendar = battery_degradation[:, 0].sum()
    battery_degradation_cycling = battery_degradation[:, 1].sum()
    battery_degradation = battery_degradation.sum()

    total_steps_min_emergency_battery_capacity_violation = 0
    energy_user_satisfaction = np.zeros((len(env.EVs)))
    for i, ev in enumerate(env.EVs):
        e_actual = ev.current_capacity
        e_max = ev.max_energy_AFAP
        energy_user_satisfaction[i] = (e_actual / e_max) * 100
        total_steps_min_emergency_battery_capacity_violation += (
            ev.min_emergency_battery_capacity_metric
        )

    stats = {
        # 'total_ev_served': total_ev_served,
        #  'total_profits': total_profits,
        "total_energy_charged": total_energy_charged,
        "total_energy_discharged": total_energy_discharged,
        #  'average_user_satisfaction': average_user_satisfaction,
        "power_tracker_violation": power_tracker_violation,
        "tracking_error": tracking_error,
        "energy_tracking_error": energy_tracking_error,
        "energy_user_satisfaction": np.mean(energy_user_satisfaction),
        "std_energy_user_satisfaction": np.std(energy_user_satisfaction),
        "min_energy_user_satisfaction": np.min(energy_user_satisfaction),
        "total_steps_min_emergency_battery_capacity_violation": total_steps_min_emergency_battery_capacity_violation,
        "total_transformer_overload": total_transformer_overload,
        "battery_degradation": battery_degradation,
        "battery_degradation_calendar": battery_degradation_calendar,
        "battery_degradation_cycling": battery_degradation_cycling,
        "total_reward": env.total_reward,
    }

    if env.eval_mode != "optimal" and env.replay is not None:
        if env.replay.optimal_stats is not None:
            stats["opt_profits"] = env.replay.optimal_stats["total_profits"]
            stats["opt_tracking_error"] = env.replay.optimal_stats["tracking_error"]
            stats["opt_actual_tracking_error"] = env.replay.optimal_stats[
                "energy_tracking_error"
            ]
            stats["opt_power_tracker_violation"] = env.replay.optimal_stats[
                "power_tracker_violation"
            ]
            stats["opt_energy_user_satisfaction"] = env.replay.optimal_stats[
                "energy_user_satisfaction"
            ]
            stats["opt_total_energy_charged"] = env.replay.optimal_stats[
                "total_energy_charged"
            ]

    return stats


def print_statistics(env) -> None:
    assert env.stats is not None, "No statistics available. Run the simulation first!"

    stats = env.stats

    total_ev_served = stats["total_ev_served"]
    total_profits = stats["total_profits"]
    total_energy_charged = stats["total_energy_charged"]
    total_energy_discharged = stats["total_energy_discharged"]
    average_user_satisfaction = stats["average_user_satisfaction"]
    total_transformer_overload = stats["total_transformer_overload"]
    tracking_error = stats["tracking_error"]
    energy_tracking_error = stats["energy_tracking_error"]
    power_tracker_violation = stats["power_tracker_violation"]
    energy_user_satisfaction = stats["energy_user_satisfaction"]
    std_energy_user_satisfaction = stats["std_energy_user_satisfaction"]
    min_energy_user_satisfaction = stats["min_energy_user_satisfaction"]

    total_transformer_overload = stats["total_transformer_overload"]
    battery_degradation = stats["battery_degradation"]
    battery_degradation_calendar = stats["battery_degradation_calendar"]
    battery_degradation_cycling = stats["battery_degradation_cycling"]

    print("\n\n==============================================================")
    print("Simulation statistics:")
    for cs in env.charging_stations:
        print(cs)
    print(
        f"  - Total EVs spawned: {env.total_evs_spawned} |  served: {total_ev_served}"
    )
    print(f"  - Total profits: {total_profits:.2f} €")
    print(f"  - Average user satisfaction: {average_user_satisfaction*100:.2f} %")

    print(
        f"  - Total energy charged: {total_energy_charged:.1f} | discharged: {total_energy_discharged:.1f} kWh"
    )
    print(
        f"  - Power Tracking squared error: {tracking_error:.2f}, Power Violation: {power_tracker_violation:.2f} kW"
    )
    print(f" - Actual Energy Tracking error: {energy_tracking_error:.2f} kW")
    print(
        f"  - Mean energy user satisfaction: {energy_user_satisfaction:.2f} % | Min: {min_energy_user_satisfaction:.2f} %"
    )
    print(f"  - Std Energy user satisfaction: {std_energy_user_satisfaction:.2f} %")
    print(
        f"  - Total Battery degradation: {battery_degradation:.5f}% | Calendar: {battery_degradation_calendar:.5f}%, Cycling: {battery_degradation_cycling:.5f}%"
    )
    print(f"  - Total transformer overload: {total_transformer_overload:.2f} kWh \n")

    print("==============================================================\n\n")
