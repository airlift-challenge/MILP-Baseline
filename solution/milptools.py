# Import pyomo
import itertools
import sys
import random

import networkx as nx
import pyomo.environ as pyo
import numpy as np
import airlift as flatsky
from itertools import chain, combinations
from collections import namedtuple
from airlift.envs.airlift_env import ObservationHelper as oh
from pyomo.opt import SolverStatus, TerminationCondition


# Helper functions - not currently using
class CargoPickup:
    def __init__(self,id):
        self.cargoId = id

class CargoDropoff:
    def __init__(self,id):
        self.cargoId = id

class CargoRequirement:
    def __init__(self,id):
        self.cargoId = id
        self.pickup = CargoPickup(id)
        self.dropoff = CargoDropoff(id)


def powerset(iterable,n):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,n+1))


# Looks at x variable to determine which event happens in given slot.
def which_event_in_slot(x, p, pickup_r, dropoff_r,start,end):
    if x[start,p].value == 1:
        return start, None, 'start'

    if x[end,p].value == 1:
        return start, None, 'end'

    for r, e in pickup_r.items():
        if x[e,p].value == 1:
            return e, r, 'pickup'

    for r, e in dropoff_r.items():
        if x[e,p].value == 1:
            return e, r, 'drop'

    assert False, "event lookup failed"



def create_event_sequence(airplane_start, model,cargo_r,pickup_r,dropoff_r,start,end):
    eventList = list()
    for p in model.P:
        e, r, eventType = which_event_in_slot(model.x, p, pickup_r, dropoff_r, start, end)

        # Add data relevant to action
        if eventType == 'start':
            cid = None
            airportid = airplane_start
        elif eventType == 'end':
            cid = None
            airportid = eventList[-1]["airportid"]
        else:
            c = cargo_r[r]
            cid = c.id
            airportid = c.location if eventType == "pickup" else c.destination

        eventData = {'eventType': eventType,
                     'eventTime': model.t[p].value,
                     'cargoId': cid,
                     'airportid': airportid}
        eventList.append(eventData)
    return eventList

# Utility to convert between cargo assignments S and action space
def convert_s_into_flatsky_actions(aircrafts,S):
    actions_all = dict()
    for c_aircraft in S:
        actions = []
        for c_cargo in S[c_aircraft]:
            # Create pickup event
            actions.append(flatsky.load_action(c_cargo.id))
            # Create transport event (process doesn't generate takeoff event?)
            actions.append(flatsky.takeoff_action(c_cargo.airport_id))
            # Create dropoff event
            actions.append(flatsky.unload_action(c_cargo.id))
        actions_all[c_aircraft] = actions
    return actions_all


def path_time(digraph, source, dest, processing_time):
    hops = nx.shortest_path_length(digraph, source, dest, weight=None) - 1
    flighttime = nx.shortest_path_length(digraph, source, dest, weight="time")
    return processing_time * hops + flighttime

# Look at global state to determine travel time between events.
# TODO: Incorporate processing time
def tau_matrix(start_airport, globalstate, cargo_r, pickup_r, dropoff_r, start, end, processing_time):
    digraph = oh.get_multidigraph(globalstate)

    tau = {}
    for r, c in cargo_r.items():
        tau[start, pickup_r[r]] = path_time(digraph, start_airport, c.location, processing_time)
        tau[start, dropoff_r[r]] = path_time(digraph, start_airport, c.destination, processing_time)
        tau[pickup_r[r], end] = 0 # The airplane can stop wherever it is, requiring no travel time
        tau[dropoff_r[r], end] = 0


        for rp, cp in cargo_r.items():
            tau[pickup_r[r], pickup_r[rp]] = path_time(digraph, c.location, cp.location, processing_time)
            tau[pickup_r[r], dropoff_r[rp]] = path_time(digraph, c.location, cp.destination, processing_time)
            tau[dropoff_r[r], pickup_r[rp]] = path_time(digraph, c.destination, cp.location, processing_time)
            tau[dropoff_r[r], dropoff_r[rp]] = path_time(digraph, c.destination, cp.destination, processing_time)

    return tau

# Objective function for problem 2.
# We can call this wih a solved model and use pyo.value to find the optimal value
def objective2a(model):
    priority = 1
    C = 0.1

    P = max(model.P)

    summation = 0
    for e in model.E:
        for p in model.P:
            summation += priority * model.s[(e, p)]
    return summation + C * (model.t[P] - model.t[1])

# Solve Single Aircraft - Problem (2)
def solve_single_aircraft(aircraft_a, start_airport, requirements_S, solver = 'cplex', earliest_start_time=0):
    globalstate = aircraft_a['globalstate']

    # Make sure if an aircraft isn't needed, it doesn't waste time building a model
    if len(requirements_S) == 0:
        model = None
        eventList = []
        optimal_value = 0 # Assuming that objective is 0 if a plane does nothing
    else:
        # Create blank model
        model = pyo.ConcreteModel()

        # Define parameters, variables in relation to data from aircraft's global state and requirement set S
        ## S, set of cargo indices assigned to this aircraft
        S = len(requirements_S)
        P = 2*S+2
        model.P = pyo.RangeSet(P)
        model.P_m1 = pyo.RangeSet(1,P-1)
        model.P_2P = pyo.RangeSet(2,P)

        ## Defining parameters for cargo
        MaxWeight = aircraft_a['max_weight']
        delta = globalstate['scenario_info'][0].processing_time

        ## Define events
        model.S = pyo.RangeSet(S)
        model.pickup = pyo.RangeSet(1,S)
        model.dropoff = pyo.RangeSet(S+1,2*S)
        model.start = pyo.Set(initialize=[2*S+1])
        model.end = pyo.Set(initialize=[2*S+2])

        # Helper dictionaries associating each requirement number with associated cargo data, pickup events, and dropoff events.
        # We will use these to simplify indexing later.
        cargo_r = {r: c for r, c in zip(model.S, requirements_S)}
        pickup_r = {r: e for r, e in zip(model.S, model.pickup)}
        dropoff_r = {r: e for r, e in zip(model.S, model.dropoff)}
        start = model.start.at(1)
        end = model.end.at(1)

        model.E = model.pickup | model.dropoff | model.start | model.end
        model.EE = model.E * model.E
        model.EP = model.E * model.P
        model.x = pyo.Var(model.EP,initialize=0,within=pyo.NonNegativeIntegers) # covers 2u
        model.xby = pyo.Var(model.EP,initialize=0,within=pyo.Binary) # covers 2t
        model.EEP = model.E * model.E *model.P
        model.y = pyo.Var(model.EEP,initialize=0,within=pyo.NonNegativeReals) # covers 2v
        # model.z = pyo.Var(model.P,initialize=0,within=pyo.Binary) # covers 2w # We assume time is always needed to rest (aka process)
        model.t = pyo.Var(model.P,initialize=0,within=pyo.Reals)
        model.s = pyo.Var(model.EP,initialize=0,within=pyo.NonNegativeReals) # covers 2x
        # model.omega = pyo.Var(model.P,initialize=0,within=pyo.NonNegativeReals) # covers 2y # Not modeling active time

        ## Pull cargo weights and delivery times
        T = max(r.hard_deadline for r in requirements_S) + delta

        l = {}
        u = {}
        B = {}
        w = {}
        for r, c in cargo_r.items():
            epickup = pickup_r[r]
            l[epickup] = c.earliest_pickup_time
            u[epickup] = c.soft_deadline
            B[epickup] = c.hard_deadline-c.soft_deadline
            w[epickup] = 1.0*c.weight

            edropoff = dropoff_r[r]
            l[edropoff] = l[epickup]
            u[edropoff] = u[epickup]
            B[edropoff] = B[epickup]
            w[edropoff] = -w[epickup]

        l[start] = earliest_start_time
        u[start] = 0
        B[start] = 0
        w[start] = 0

        l[end] = 0
        u[end] = T
        B[end] = 0
        w[end] = 0

        model.l = pyo.Param(model.E,within=pyo.Reals,initialize=l)
        model.u = pyo.Param(model.E,within=pyo.Reals,initialize=u)
        model.B = pyo.Param(model.E,within=pyo.Reals,initialize=B)
        model.w = pyo.Param(model.E,within=pyo.Reals,initialize=w)

        # Define travel times from pickup to deliver
        tau = tau_matrix(start_airport, globalstate, cargo_r, pickup_r, dropoff_r, start, end, processing_time=delta)
        model.tau = pyo.Param(model.EE, initialize=tau, within=pyo.Reals, default=0) # If tau is not specified use a large value (essentially infinite to represent infeasibale)

        # ** Set rules / objective
        model.objective2a = pyo.Objective(rule=objective2a)

        def rule2b(model,e):
            return (sum(model.x[(e,p)] for p in model.P) == 1)
        model.rule2b = pyo.Constraint(model.E,rule=rule2b)

        def rule2c(model,p):
            return (sum(model.x[(e,p)] for e in model.E) == 1)
        model.rule2c = pyo.Constraint(model.P,rule=rule2c)

        def rule2d(model):
            return (model.x[start,1] == 1)
        model.rule2d = pyo.Constraint(rule=rule2d)

        def rule2e(model):
            return (model.x[end,P] == 1)
        model.rule2e = pyo.Constraint(rule=rule2e)

        def rule2f(model,e):
            return model.x[(e,1)] == model.xby[(e,1)]
        model.rule2f = pyo.Constraint(model.E,rule=rule2f)

        def rule2g(model,e,p):
            return model.xby[(e,p)] - model.xby[(e,p-1)] == model.x[(e,p)]
        model.rule2g = pyo.Constraint(model.E,model.P_2P,rule=rule2g)

        def rule2h(model,e,p):
            return model.xby[(e,p)] >= model.xby[(e,p-1)]
        model.rule2h = pyo.Constraint(model.E,model.P_2P,rule=rule2h)

        def rule2i(model,r,p):
            return model.xby[(model.dropoff.at(r),p)] <= model.xby[(model.pickup.at(r),p)]
        model.rule2i = pyo.Constraint(model.S,model.P,rule=rule2i)

        def rule2j(model,e,p):
            return sum(model.y[e,eprime,p] for eprime in model.E) == model.x[(e,p)]
        model.rule2j = pyo.Constraint(model.E,model.P_m1,rule=rule2j)

        def rule2k(model,e,p):
            return sum(model.y[eprime,e,p-1] for eprime in model.E) == model.x[(e,p)]
        model.rule2k = pyo.Constraint(model.E,model.P_2P,rule=rule2k)

        def rule2l(model,p):
            r = pyo.Reference(model.xby[:, p])
            return pyo.sum_product(model.w,r) <= MaxWeight
        model.rule2l = pyo.Constraint(model.P,rule=rule2l)

        def rule2m(model,p):
            # Produce summation
            summation = 0
            for e in model.E:
                for eprime in model.E:
                    #summation += model.tau[(e,eprime)]*model.y[(e,eprime,p-1)] + delta*model.z[p-1]
                    summation += model.tau[(e,eprime)] * model.y[(e,eprime,p-1)]
            return model.t[p] >= model.t[p-1] + summation + delta # Assume we alway sneed to rest (aka process before taking off again)
        model.rule2m = pyo.Constraint(model.P_2P,rule=rule2m)

        def rule2n(model,p):
            # Produce summation
            summation = 0
            for e in model.E:
                summation += model.l[e]*model.x[(e,p)]
            return model.t[p] >= summation
        model.rule2n = pyo.Constraint(model.P,rule=rule2n)

        def rule2o(model,p):
            summation_left = 0
            for e in model.E:
                summation_left += model.u[e]*model.x[(e,p)]
            summation_right = 0
            for e in model.E:
                summation_right += model.s[(e,p)]
            return model.t[p] <= (summation_left + summation_right)
        model.rule2o = pyo.Constraint(model.P,rule=rule2o)

        def rule2t(model,e,p):
            return model.xby[(e,p)] <= 1
        model.rule2t = pyo.Constraint(model.EP,rule=rule2t)




        # ** Solve model

        # tee indicates whether or not to print solver output to standard output
        # tee = True    # Print solver output
        tee = False     # Don't print solver output
        result = pyo.SolverFactory(solver).solve(model, tee=tee)

        # ** Interpret results
        stat = result.solver.status
        termination = result.solver.termination_condition
        if (stat == SolverStatus.ok) and (termination == TerminationCondition.optimal):
            optimal_value = model.objective2a()
            print("  Solution is is feasible and optimal. Optimal value: {}".format(optimal_value))
            eventList = create_event_sequence(start_airport, model, cargo_r, pickup_r, dropoff_r, start, end)
        elif termination == TerminationCondition.infeasible:
            optimal_value = np.inf
            eventList = []
        else:
            # If the solver fails for some reason, we will treat it as infeasible.
            # There's probably better ways to handle these cases...
            optimal_value = np.inf
            eventList = []

    return model, eventList, optimal_value

# Utility function to identify key from aircraft's index in A
def build_aircraft_key_with_id(aircrafts):
    key_id_dict = dict()
    id = 0
    for cKey in aircrafts.keys():
            key_id_dict[id] = cKey
            id += 1
    return key_id_dict

# Implement 3, produce set of actions
def algorithm3(aircrafts, start_airports, earliest_start_times, cargo, solver = 'cplex'):
    globalstate = list(aircrafts.values())[0]["globalstate"]

    actions = list()
    aircraft_id_to_key = build_aircraft_key_with_id(aircrafts)

    ## Algorithm 3 (4.5 solution steps)
    # Initial sets S_1-S_A to empty
    A = len(aircrafts)
    S = dict()
    for a in range(0,A):
        S[a] = []
    # Initial requirements
    R = cargo.copy()
    f_a = np.zeros([A])

    # Initial objective values with no assignments
    eventLists = dict()
    for a in range(0,A):
        aid = aircraft_id_to_key[a]
        _, eventLists[a], f_a[a] = solve_single_aircraft(aircrafts[aid], start_airports[aid], S[a], solver, earliest_start_time=earliest_start_times[aid])
    Z = np.sum(f_a)

    # Try each plane with each cargo
    random.shuffle(R)
    for i, r in enumerate(R): # Equivalent to the while loop in Algorithm 3.
        Za_pot = np.zeros([A,1])
        # For each aircraft, find Z_a 
        eventList_pot = dict()
        f_a_pot = np.zeros([A, 1])
        for a in range(0,A):
            aid = aircraft_id_to_key[a]
            S_pot = S[a].copy()
            S_pot.append(r)
            print("Solving cargo {}/{} for airplane {}/{}:".format(i+1, len(R), a+1, A))
            model_solution, eventList_pot[a], f_a_pot[a] = solve_single_aircraft(aircrafts[aid], start_airports[aid], S_pot, solver, earliest_start_time=earliest_start_times[aid])
            Za_pot[a] = np.sum(f_a[0:a]) + np.sum(f_a[(a+1):A]) + f_a_pot[a]
        # Pick lowest Za, argmin_a => k*
        a_star = np.argmin(Za_pot)
        # Update S for chosen a
        S[a_star].append(r)

        eventLists[a_star] = eventList_pot[a_star]
        f_a[a_star] = f_a_pot[a_star]

        Z = Za_pot[a_star]

    return aircraft_id_to_key, S, Z, f_a, eventLists


# Solve Overall Algorithm Approach in section 4.5 p786
def solve_multiple_aircraft(aircrafts, start_airports=None, earliest_start_times=None, cargo=None, solver='cplex'):
    if start_airports is None:
        start_airports = {a: obs["current_airport"] for a, obs in aircrafts.items()}
    if earliest_start_times is None:
        earliest_start_times = {a: 0 for a, obs in aircrafts.items()}
    if cargo is None:
        state = list(aircrafts.values())[0]["globalstate"]
        cargo = state['active_cargo']

    # Step 1) Initial partition (Algorithm 3)
    aircraft_id_to_key, S, Z, f_a, eventLists = algorithm3(aircrafts, start_airports, earliest_start_times, cargo, solver)

    # Step 2) Refine initial partition (Algorithm 4)
    # Pending....

    # Step 3) Algorithm 1/2
    # Pending....

    # Step 4) Solve master based on problem 3
    # Pending....

    return aircraft_id_to_key, S, Z, f_a, eventLists

