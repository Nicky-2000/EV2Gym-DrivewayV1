'''  This file contains various example state functions for the RL agent '''
import math
import numpy as np


import numpy as np

def state_function_basic_profit_view(env):
    """
    Returns the observation vector for the agent.
    Each household (1 port per charger) contributes a row to the observation matrix.
    
    Features per household:
    - Current battery level (0 to 1)
    - Current charge price
    - Current discharge price
    - Time of day (sin, cos encoding)
    """

    obs = []
    time_step = min(env.current_step, env.charge_prices.shape[1] - 1)

    for household in env.households:
        ev = household.ev
        cs = household.charging_station

        # Battery level normalized
        battery_level = ev.current_capacity / ev.battery_capacity

        # Electricity prices
        charge_price = env.charge_prices[cs.id, time_step]
        discharge_price = env.discharge_prices[cs.id, time_step]

        # Time encoding
        minutes_in_day = 24 * 60
        t_min = env.sim_date.hour * 60 + env.sim_date.minute
        time_sin = np.sin(2 * np.pi * t_min / minutes_in_day)
        time_cos = np.cos(2 * np.pi * t_min / minutes_in_day)

        features = [
            battery_level,
            charge_price,
            discharge_price,
            time_sin,
            time_cos
        ]

        obs.append(features)

    return np.array(obs).flatten()


def state_function_with_future_trip(env):
    """
    Extended state with future trip info.
    
    Features per household:
    - Current battery level
    - Current charge price
    - Current discharge price
    - Time of day (sin, cos)
    - Energy required for next trip (kWh / capacity)
    - Time until departure in minutes
    """

    obs = []

    for household in env.households:
        ev = household.ev
        cs = household.charging_station
        time_step = min(env.current_step, env.charge_prices.shape[1] - 1)
        
        future_window = 24  # next 2 hours if 15min timescale
        end_idx = min(time_step + future_window, env.charge_prices.shape[1])
        avg_future_charge = np.mean(env.charge_prices[cs.id, time_step:end_idx])
        avg_future_discharge = np.mean(env.discharge_prices[cs.id, time_step:end_idx])
        
        battery_level = ev.current_capacity / ev.battery_capacity
        charge_price = env.charge_prices[cs.id, time_step]
        discharge_price = env.discharge_prices[cs.id, time_step]

        # Time encoding
        minutes_in_day = 24 * 60
        t_min = env.sim_date.hour * 60 + env.sim_date.minute
        time_sin = np.sin(2 * np.pi * t_min / minutes_in_day)
        time_cos = np.cos(2 * np.pi * t_min / minutes_in_day)

        # Next trip info (fallback to 0 if not available)
        energy_needed = ev.energy_required_for_next_trip if ev.energy_required_for_next_trip else 0.0
        
        # Time until departure (in steps), fallback to 0
        if hasattr(ev, 'time_of_departure') and ev.time_of_departure is not None:
            steps_until_departure = max(ev.time_of_departure - env.current_step, 0)
            time_until_departure = steps_until_departure * env.timescale  # convert to minutes
        else:
            time_until_departure = 0.0
            
        # EV is home flag
        ev_is_home_flag = 1.0 if household.ev_is_home else 0.0

            
        features = [
            battery_level,
            charge_price,
            discharge_price,
            time_sin,
            time_cos,
            energy_needed / ev.battery_capacity,
            time_until_departure / 1440.0,  # normalized over 1 day
            ev_is_home_flag,
            avg_future_charge / 100,  # normalize based on expected max
            avg_future_discharge / 100
        ]


        obs.append(features)

    return np.array(obs).flatten()


def PublicPST(env, *args):
    '''This state function is the public power setpoints
    The state is the public power setpoints
    The state is a vector '''

    state = [
        (env.current_step/env.simulation_length),
        # env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        # math.sin(env.sim_date.hour/24*2*math.pi),
        # math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # the final state of each simulation
    # if env.current_step < env.simulation_length:        
    #     setpoint = min(env.power_setpoints[env.current_step], env.charge_power_potential[env.current_step])        
    # else:
    #     setpoint = 0       
    if env.current_step < env.simulation_length:  
        # setpoint = env.power_setpoints[env.current_step:env.current_step+10]
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = np.zeros((1))
        
    # if len(setpoint) < 10:
    #     setpoint = np.append(setpoint, np.zeros(10-len(setpoint)))
    
    state.append(setpoint)
    state.append(env.current_power_usage[env.current_step-1])

    # For every transformer
    for tr in env.transformers:
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0.5,  # we know if the EV is full
                            EV.total_energy_exchanged,
                            # EV.max_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            # EV.min_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            (env.current_step-EV.time_of_arrival)
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state

def V2G_profit_max(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario.
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

    charge_prices = abs(env.charge_prices[0, env.current_step:
        env.current_step+20])
    
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state

def V2G_profit_max_loads(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario with loads
    '''
    
    state = [
        (env.current_step),        
    ]

    state.append(env.current_power_usage[env.current_step-1])

    charge_prices = abs(env.charge_prices[0, env.current_step:
        env.current_step+20])
    
    if len(charge_prices) < 20:
        charge_prices = np.append(charge_prices, np.zeros(20-len(charge_prices)))
    
    state.append(charge_prices)
    
    # For every transformer
    for tr in env.transformers:
        loads, pv = tr.get_load_pv_forecast(step = env.current_step,
                                            horizon = 20)
        power_limits = tr.get_power_limits(step = env.current_step,
                                           horizon = 20)
        state.append(loads-pv)
        state.append(power_limits)
        
        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                            ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state
    


def BusinessPSTwithMoreKnowledge(env, *args):
    '''
    This state function is used for the business case scenario that requires more knowledge such as SoC and time of departure for each EV present.
    '''

    state = [
        (env.current_step) / env.simulation_length,
        #env.sim_date.weekday() / 5,
        # turn hour and minutes in sin and cos
        #math.sin(env.sim_date.hour/12*2*math.pi),
        #math.cos(env.sim_date.hour/12*2*math.pi),
    ]

    # the final state of each simulation
    if env.current_step < env.simulation_length:
        state.append(env.power_setpoints[env.current_step]) #/100
        state.append(env.charge_power_potential[env.current_step]) #/100
    else:
        state.append(env.power_setpoints[env.current_step-1]) #/100
        state.append(env.charge_power_potential[env.current_step-1]) #/100   

    for tr in env.transformers:
        state.append(tr.max_current/100)
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([#EV.total_energy_exchanged / EV.battery_capacity, #how much soc we charge
                                      #EV.max_ac_charge_power*1000 /            same EVs, no need right now
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      #EV.min_ac_charge_power*1000 /
                                      #(cs.voltage*math.sqrt(cs.phases)),
                                      EV.time_of_arrival / env.simulation_length,  # time of arrival
                                      EV.etime_of_departure / env.simulation_length,  # time of departure
                                      EV.get_soc(),  # soc
                                      #(EV.etime_of_departure - env.current_step) \
                                      #  / env.simulation_length, #remaining time
                                      #(env.current_step-EV.time_of_arrival) \
                                      #  / env.simulation_length,  # time stayed
                                      #(EV.etime_of_departure - \
                                      # EV.time_of_arrival) / env.simulation_length, # total staying time
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) /
                                      #  (EV.etime_of_departure - EV.time_of_arrival)) / EV.max_ac_charge_power),  # average charging speed
                                      #(((EV.battery_capacity - EV.battery_capacity_at_arrival) / EV.battery_capacity)) \
                                      #  / ((EV.etime_of_departure - env.current_step + 1) / env.simulation_length),   #charging priority
                                      #EV.required_power / EV.battery_capacity,  # required energy
                                      ])
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state