"""
This file contains the Household class which represents a household with an electric vehicle (EV) and a charging station.
"""

from ev2gym_driveway.models.ev_charger import EV_Charger
from ev2gym_driveway.models.ev import EV

from ev2gym_driveway.models.ev_charger import EV_Charger
from ev2gym_driveway.models.ev import EV
from datetime import datetime, time, timedelta
import math



class Household:
    def __init__(
        self,
        charging_station: EV_Charger,
        ev: EV,
        ev_weekly_profile: list[dict],
        timescale: int = 15,
        sim_starting_date: datetime = None,
    ):
        self.ev = ev
        self.charging_station = charging_station
        self.ev_weekly_profile = ev_weekly_profile
        self.timescale = timescale
        self.sim_starting_date = sim_starting_date
        self.total_money_spent_charging = 0
        self.total_money_earned_discharging = 0
        self.total_invalid_action_punishment = 0
        
        self.current_trip = None
        
    def step(self, actions, charge_price, discharge_price, sim_timestamp):
        self.update_household(sim_timestamp)

        
        if self.current_trip is None and self.ev_is_home: # EV should be connected to charger
            self.charging_station.evs_connected[0] = (
                self.ev
            )  # Assuming only one EV per household currently
        else:
            self.charging_station.evs_connected[0] = None

        money_spent_charging, money_earned_discharging, invalid_action_punishment = (
            self.charging_station.step(
                actions, charge_price, discharge_price
            )
        )
        self.total_money_spent_charging += money_spent_charging
        self.total_money_earned_discharging += money_earned_discharging
        self.total_invalid_action_punishment += invalid_action_punishment
        
        return invalid_action_punishment

    def update_household(self, sim_timestamp: datetime):
        """
        Determine if the EV is connected to the charging station.
        If the EV is arriving from a trip update the SoC of the EV.
        """

        weekday = sim_timestamp.weekday() + 1  # 1=Monday
        day_profile = self.ev_weekly_profile[weekday]
        
        # Check if the EV is arriving from the current trip
        if self.current_trip is not None: 
            if self._is_ev_arriving(weekday, sim_timestamp):
                self.ev_is_home = True
                self.current_trip_weekday = None
                miles_driven = self.current_trip["miles"]
                self.ev.update_battery_capacity_after_trip(miles_driven)
        
        # Check if the EV is leaving on a new trip
        self.current_trip = self._get_current_trip(day_profile, sim_timestamp)
        
        if self.current_trip is None:
            self.ev_is_home = True
            self.current_trip_weekday = None
            # Find the timestamp that the EV will be departing home
            next_trip, days_ahead = self._get_next_trip(sim_timestamp)
            next_trip_departure_time = _hhmm_to_datetime(next_trip["departure"], sim_timestamp, days_ahead)
            next_trip_departure_time_step = timestamp_to_simulation_step(
                self.sim_starting_date, self.timescale, next_trip_departure_time
            )
            self.ev.set_time_of_departure(next_trip_departure_time_step)
            
        else: # EV is on a trip
            self.current_trip_weekday = weekday
            self.ev_is_home = False

    def _get_current_trip(self, day_profile: dict, sim_timestamp: datetime):
        """
        Returns the current trip if the EV is on a trip.
        A trip is considered ongoing if the current time is >= departure and < arrival.
        The EV is considered home at the arrival time.
        """
        for trip in day_profile.get("trips", []):
            departure = _hhmm_to_datetime(trip["departure"], sim_timestamp)
            arrival = _hhmm_to_datetime(trip["arrival"], sim_timestamp)
            if departure <= sim_timestamp < arrival:
                return trip
        return None

    def _is_ev_arriving(self, current_trip_weekday: int, sim_timestamp: datetime) -> bool:
        simulation_weekday = sim_timestamp.weekday() + 1  # 1=Monday
        if (current_trip_weekday % 7) + 1 == simulation_weekday: 
            # The current day is one day ahead of the trip day.
            arrival_time = _hhmm_to_datetime(self.current_trip["arrival"], sim_timestamp, days_ahead=-1)
        else:
            # Otherwise assume that the trip is happening on the current weekday as the timestamp
            arrival_time = _hhmm_to_datetime(self.current_trip["arrival"], sim_timestamp)
        
        return arrival_time <= sim_timestamp
    
    def _get_next_trip(self, sim_timestamp: datetime):
        """
        Returns the next trip if the EV is scheduled to leave.
        A trip is considered scheduled if the current time is < departure.
        """
        all_weekdays = [1, 2, 3, 4, 5, 6, 7]
        current_weekday = sim_timestamp.weekday() + 1  # 1=Monday

        rotated_weekdays = all_weekdays[current_weekday - 1:] + all_weekdays[:current_weekday - 1]
        
        # Add the current weekday to the end, maybe the next trip is on this day but next week.
        rotated_weekdays.append(current_weekday)  
        
        # Right now we are assuming that every vehicle has at least one trip per day
        for days_ahead, weekday in enumerate(rotated_weekdays):
            # Check all the trips to find the next one (hopefully there is one)
            day_profile = self.ev_weekly_profile[weekday]
            for trip in day_profile.get("trips", []):
                # If the next trip is not on the current weekday then we need to add the days ahead
                departure = _hhmm_to_datetime(trip["departure"], sim_timestamp, days_ahead)
                if sim_timestamp < departure:
                    return trip, days_ahead
        return None


def _hhmm_to_datetime(hhmm: int, simulation_timestamp: datetime, days_ahead: int = 0) -> datetime:
    hour, minute = divmod(hhmm, 100)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Invalid HHMM time: {hhmm}")
    
    new_date = simulation_timestamp + timedelta(days=days_ahead)
    return new_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def timestamp_to_simulation_step(sim_start_date: datetime, timescale: int, timestamp: datetime):
    """
    Convert a timestamp to the simulation step **after or at** the timestamp.
    That is, it rounds UP to the nearest timestep.
    
    Example:
    - timescale = 15 min
    - sim_start_date = 12:00
    - timestamp = 12:35 -> returns step 3 (which starts at 12:45)
    """
    if sim_start_date is None:
        raise ValueError("sim_start_date must be provided")
    
    delta_seconds = (timestamp - sim_start_date).total_seconds()
    step = math.ceil(delta_seconds / (timescale * 60))  # round UP
    return step
