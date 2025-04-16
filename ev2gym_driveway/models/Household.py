"""
This file contains the Household class which represents a household with an electric vehicle (EV) and a charging station.
"""

from ev2gym_driveway.models.ev_charger import EV_Charger
from ev2gym_driveway.models.ev import EV

from ev2gym_driveway.models.ev_charger import EV_Charger
from ev2gym_driveway.models.ev import EV
from datetime import datetime, time, timedelta


class Household:
    def __init__(
        self,
        charging_station: EV_Charger,
        ev: EV,
        ev_weekly_profile: list[dict],
        timescale: int = 15,
    ):
        self.ev = ev
        self.charging_station = charging_station
        self.ev_weekly_profile = ev_weekly_profile
        self.timescale = timescale
        self.total_money_spent_charging = 0
        self.total_money_earned_discharging = 0
        self.total_invalid_action_punishment = 0
        
        self.current_trip = None
        
    def step(self, actions, charge_price, discharge_price, sim_timestamp):
        self.update_household(sim_timestamp)

        if self.current_trip is not None:
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
            if self._is_ev_arriving(day_profile, sim_timestamp):
                self.ev_is_home = True
                self._update_ev_soc_from_trip(day_profile, sim_timestamp)
        
        # Check if the EV is leaving on a new trip
        self.current_trip = self._get_current_trip(day_profile, sim_timestamp)

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

    def _is_ev_arriving(self, day_profile: dict, sim_timestamp: datetime) -> bool:
        arrival_time = _hhmm_to_datetime(self.current_trip["arrival"], sim_timestamp)
        return arrival_time <= sim_timestamp

    def _update_ev_soc_from_trip(self, day_profile: dict, sim_timestamp: datetime):
        # TODO: Implement the logic to update the EV's state of charge (SoC) based on the trip data.
        pass
    
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

