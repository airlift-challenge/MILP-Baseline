from airlift.solutions import Solution
from airlift.envs import PlaneState
from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
from pyomo.opt import SolverFactory

from solution.milptools import solve_multiple_aircraft




REPLANINTERVAL = 100




# **** Helpers for sequencing airplane actions based on event list generated by MILP


# Makes a "cargo event list", which indicates a sequence of airports to travel to and cargo to load/unload at each.
# Takes an the "MILP event list" generated by the MILP. Notes:
#  * We assume the MILP event list may consist of chained list from different solutions. In this caase, the start airport
#    of a given list should be the same as the end airport of the previous list.
#  * We check that the start and end events generated by the MILP are consistent, e.g., the airport of the end event is the
#    same as the airport for the last cargo event. Otherwise, we ignore these events since we do not track aircraft active time
#    in our scenarios,
#  * The MILP may have a series of pickup/dropoff events at the same airport. We consolidate these into one cargo event.
def make_cargo_event_list(MILP_event_list):
    new_event_list = []

    i = 0
    while i < len(MILP_event_list):
        event = MILP_event_list[i]
        airportid =  event["airportid"]
        eventType = event["eventType"]
        cargoId = event["cargoId"]
        if eventType == "start":
            if i > 0:
                assert MILP_event_list[i-1]["eventType"] == "end"
                assert airportid == MILP_event_list[i-1]["airportid"] # This could be relaxed
        elif eventType == "end":
            assert airportid == MILP_event_list[i-1]["airportid"] # This could be relaxed
        else: # Must be cargo event
            if eventType == "pickup":
                cargo_to_load = [cargoId]
                cargo_to_unload = []
            elif eventType == "drop":
                cargo_to_load = []
                cargo_to_unload = [cargoId]
            else:
                assert False, "Unrecognized event"

            # Update last cargo event
            if new_event_list and airportid == new_event_list[-1]["airportid"]:
                new_event_list[-1]["load"].extend(cargo_to_load)
                new_event_list[-1]["unload"].extend(cargo_to_unload)
            # Make a new cargo event
            else:
                new_event_list.append({
                    "airportid": airportid,
                    "load": cargo_to_load,
                    "unload": cargo_to_unload
                })

        i += 1

    return new_event_list



class AirplaneSequencer:
    # TODO: Update path if route goes offline
    def __init__(self):
        self._event_list = []
        # Indicates the next event which needs to occur. Will be None if the current event list has been completed (or empty)
        self._next_event_index = None
        self._path = None

    # Get the next event
    @property
    def _next_event(self):
        if self._next_event_index is None:
            return None
        else:
            return self._event_list[self._next_event_index]

    # Move to the next event
    def _advance_event_index(self):
        self._next_event_index += 1
        if self._next_event_index >= len(self._event_list):
            self._next_event_index = None

    # Generates an action for the given ariplane based on its event list
    def airplane_policy(self, obs):
        state = obs["globalstate"]

        # We will only assign a new action if airplane is ready for next orders and we have more events to process
        if oh.needs_orders(obs) and self._next_event is not None:
            # The next airport is the one we are at or the one we are in flight to
            next_airport = obs["current_airport"] if obs["destination"] == NOAIRPORT_ID else obs["destination"]

            # Let's start building the next action
            action = {"priority": 1,  # All airplanes are priority 1
                      "cargo_to_load": [],
                      "cargo_to_unload": [],
                      "destination": NOAIRPORT_ID}

            # Will we be arriving at the airport for the next event?
            if self._next_event["airportid"] == next_airport:
                # Indicate the cargo to load/unload associated with next event (once we arrive at the event airport)
                action["cargo_to_load"] = self._next_event["load"]
                action["cargo_to_unload"] = self._next_event["unload"]

                # We can start processing the next event (if there is one)
                self._advance_event_index()

            # If there are more events to process
            if self._next_event is not None:
                # If we don't have a path yet, set a path to the next event (note if we are already at the next event's airport, this will be an empty list)
                if not self._path:
                    path = oh.get_lowest_cost_path(state,
                                                   next_airport,
                                                   self._next_event["airportid"],
                                                   obs["plane_type"])
                    # The 1st airport in the path should be the airport we are going to (or already at)
                    assert path[0] == next_airport
                    self._path = path[1:]

                # Set the next airport according to the path (if there is a path)
                if self._path:
                    action["destination"] = self._path.pop(0)

            return action
        else:
            # We already have an action pending. Leave it as-is.
            return None

    # Which is the last airport in the current sequence. For planning additional events.
    @property
    def end_airport(self):
        if not self.event_list:
            return None
        else:
            return self.event_list[-1]["airport_id"]

    # This is the only way the event list should be updated
    def add_event_list(self, event_list):
        # If current event list is complete start at 1st element in new event list
        if self._next_event_index is None and event_list:
            self._next_event_index = len(self._event_list)

        self._event_list.extend(event_list)







# TODO: Handle disconnected routes
class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self):
        super().__init__()
        self._sequencer = None
        self._end_events = None

        self._new_cargo = None
        self._steps_since_last_replan = None

        # See https://stackoverflow.com/questions/51371067/pyomo-list-available-solvers
        if SolverFactory('cplex').available():
            print("Using cplex solver")
            self._solver = 'cplex'
        elif SolverFactory('glpk').available():
            print("Using glpk solver")
            self._solver = 'glpk'
        else:
            raise Exception("Either the glpk or cplex solver must be installed")

    # Takes in an event list from the MILP, converts into a consolidated "cargo event list", and adds this to each airplane.
    def _update_events(self, aircraft_id_to_key, eventLists):
        for i, event_list in eventLists.items():
            cargo_event_list = make_cargo_event_list(event_list)
            self._sequencer[aircraft_id_to_key[i]].add_event_list(cargo_event_list)

            if event_list:
                end_event = event_list[-1]
                assert end_event["eventType"] == "end"
                self._end_events[aircraft_id_to_key[i]] = end_event

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

        self._sequencer = {a: AirplaneSequencer() for a in obs}
        self._end_events = {a: None for a in obs}
        aircraft_id_to_key, S, Z, f_a, eventLists = solve_multiple_aircraft(obs, solver=self._solver)
        self._update_events(aircraft_id_to_key, eventLists)

        self._new_cargo = []
        self._steps_since_last_replan = 0

    def policies(self, obs, dones, infos):
        state = list(obs.values())[0]["globalstate"]

        # Collect new cargo and periodically plan delivery for the new cargo that as accumulated since the last replan
        self._new_cargo.extend(state["event_new_cargo"])
        self._steps_since_last_replan += 1
        if self._steps_since_last_replan > REPLANINTERVAL:
            end_airports = {}
            end_times = {}
            for a, e in self._end_events.items():
                # The airplane has not been used yet
                if e is None:
                    end_airports[a] = obs[a]["current_airport"]
                    end_times[a] = 0
                # The airplane is/has been used - capture its end state info
                else:
                    end_airports[a] = e["airportid"]
                    end_times[a] = e["eventTime"]

            # Do planning for new dynamic cargo - essentially we will let the airplanes finish what they are doing, then will have them handle the new dynamic cargo.
            # We pass in their end info as the start info for the next round of deliveries.
            aircraft_id_to_key, S, Z, f_a, eventLists = solve_multiple_aircraft(obs, start_airports=end_airports, earliest_start_times=end_times, cargo=self._new_cargo, solver=self._solver)
            self._update_events(aircraft_id_to_key, eventLists)

            self._new_cargo = []
            self._steps_since_last_replan = 0

        # Build actions using the sequencers
        actions = {}
        for a in self.agents:
            actions[a] = self._sequencer[a].airplane_policy(obs[a])

        return actions

