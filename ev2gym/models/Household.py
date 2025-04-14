"""
This file contains the Household class which represents a household with an electric vehicle (EV) and a charging station.
"""

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV
from datetime import datetime, time


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
        self.ev_weekly_profile = self._normalize_ev_profile(
            ev_weekly_profile, timescale
        )
        self.timescale = timescale
        self.ev_is_home = True
        self.total_money_spent_charging = 0
        self.total_money_earned_discharging = 0
        self.total_invalid_action_punishment = 0

    def _normalize_ev_profile(
        self, weekly_profile: list[dict], step_minutes: int
    ) -> list[dict]:
        """
        Normalize the EV weekly profile to ensure that all trips are rounded to the nearest simulation step size.
        This makes checking for departure and arrival times easier.
        """

        def round_hhmm_to_step(hhmm: int) -> int:
            hours, minutes = divmod(hhmm, 100)
            total_minutes = hours * 60 + minutes
            rounded = (total_minutes // step_minutes) * step_minutes
            return (rounded // 60) * 100 + (rounded % 60)

        normalized_profile = {}
        for day, data in weekly_profile.items():
            trips = data.get("trips", [])
            normalized_trips = []
            for trip in trips:
                trip = trip.copy()
                trip["departure"] = round_hhmm_to_step(trip["departure"])
                trip["arrival"] = round_hhmm_to_step(trip["arrival"])
                # Some data checks
                assert (
                    trip["departure"] < trip["arrival"]
                ), f"Departure time {trip['departure']} must be less than arrival time {trip['arrival']}"
                assert (
                    trip["arrival"] <= 2359
                ), f"Arrival time {trip['arrival']} must be less than or equal to 2359"
                assert (
                    trip["departure"] >= 0
                ), f"Departure time {trip['departure']} must be greater than or equal to 0"

                normalized_trips.append(trip)

            normalized_profile[day] = {"trips": normalized_trips}
        return normalized_profile

    def step(self, actions, charge_price, discharge_price, sim_timestamp):
        self.update_household(sim_timestamp)

        if self.ev_is_home:
            self.charging_station.evs_connected[1] = (
                self.ev
            )  # Assuming only one EV per household currently
        else:
            self.charging_station.evs_connected[1] = None

        money_spent_charging, money_earned_discharging, invalid_action_punishment = (
            self.charging_station.step(
                actions, charge_price, discharge_price, sim_timestamp
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

        weekday = sim_timestamp.weekday() + 1
        day_profile = self.ev_weekly_profile[weekday]

        current_trip = self._get_current_trip(day_profile, sim_timestamp)
        self.ev_is_home = current_trip is None

        if self._is_ev_arriving(day_profile, sim_timestamp):
            self._update_ev_soc_from_trip(day_profile, sim_timestamp)

    def _get_current_trip(self, day_profile: dict, sim_timestamp: datetime):
        """
        Returns the current trip if the EV is on a trip.
        A trip is considered ongoing if the current time is >= departure and < arrival.
        The EV is considered home at the arrival time.
        """
        current_time = self._get_hhmm(sim_timestamp)
        for trip in day_profile.get("trips", []):
            if trip["departure"] <= current_time < trip["arrival"]:
                return trip
        return None

    def _is_ev_arriving(self, day_profile: dict, sim_timestamp: datetime) -> bool:
        current_time = self._get_hhmm(sim_timestamp)
        for trip in day_profile.get("trips", []):
            if trip["arrival"] == current_time:
                return True
        return False

    def _update_ev_soc_from_trip(self, day_profile: dict, sim_timestamp: datetime):
        # TODO: Implement the logic to update the EV's state of charge (SoC) based on the trip data.
        pass

    def _get_hhmm(self, dt: datetime) -> int:
        return dt.hour * 100 + dt.minute
