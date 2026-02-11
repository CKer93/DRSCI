from datetime import datetime
import gurobipy as gp
from gurobipy import GRB
import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters

logger = logging.getLogger(__name__)

def create_penalized_dummy_pendulum_tours(
    route_dict: dict, instance: object, infeas_penalty: float, dec_prec
) -> dict:
    """
    Function to add artificial pendulum tours to the route_dict of a dummy vehicle with very high fixed costs and veh_type_avail = self.dimension
    Returns an infeasible solution but gives a feeling of why no feasible solution to the SPP could be found.
    """
    len_of_route_dict = len(route_dict)
    demand_array = np.array(instance.stop_df.demand)
    if "FSM" in instance.type or "TW" in instance.type:
        if "FSM" in instance.type:
            pendulum_veh_type_row = instance.veh_type_df.loc[instance.veh_type_df['capacity'] == instance.veh_type_df['capacity'].max()]
        elif instance.type == "HFVRPTW":
            pendulum_veh_type_row = instance.veh_type_df.loc[instance.veh_type_df['capacity'] == instance.veh_type_df['capacity'].min()]
        pendulum_veh_type_vt_type = pendulum_veh_type_row.veh_type[0]
        pendulum_veh_type_capacity = pendulum_veh_type_row.capacity[0]
        max_distance_costs = instance.distance_matrix.max()*2
        penalty_costs = round(
            pendulum_veh_type_row.costs[0] + max_distance_costs * pendulum_veh_type_row.variable_distance_unit_costs[0],
            dec_prec
        )

        route_dict.update(
            {
                f"initial_feas_pend_tour_{len_of_route_dict + i}": {
                    "veh_id" : f"vt{pendulum_veh_type_vt_type}-{i}",
                    "route": [
                        instance.depot_identifier,
                        instance.map_idx_stop_id[i],
                        instance.depot_identifier,
                    ],
                    "route_id": f"initial_feas_pend_tour_{i}",
                    "veh_type": pendulum_veh_type_vt_type,
                    "total_costs": penalty_costs,
                }
                for i in range(1, instance.dimension)
                if i != instance.map_stop_id_idx[instance.depot_identifier]
                if demand_array[i] <= pendulum_veh_type_capacity
            }
        )
    else:
        # get maximum costs of all routes in route_dict
        penalty_costs = round((instance.veh_type_df.costs.max() + instance.distance_matrix.max() * 2) * infeas_penalty, dec_prec)

        # add a pendulum tour for each stop to the route_dict_long of a dummy vehicle with very high fixed costs and veh_type_avail = self.dimension
        route_dict.update(
            {
                f"pendulum_{len_of_route_dict + i}": {
                    "veh_id" : f"pendulum_{i}",
                    "route": [
                        instance.depot_identifier,
                        instance.map_idx_stop_id[i],
                        instance.depot_identifier,
                    ],
                    "route_id": f"pendulum_{i}",
                    "veh_type": "pendulum",
                    "total_costs": penalty_costs,
                }
                for i in range(1, instance.dimension)
                if i != instance.map_stop_id_idx[instance.depot_identifier]
            }
        )
        dummy_veh_type = {
            "veh_type": "pendulum",
            "capacity": instance.stop_df.demand.max(),
            "costs": penalty_costs,
            "variable_distance_unit_costs": instance.veh_type_df.variable_distance_unit_costs.max() * infeas_penalty,
            "no_of_available": instance.dimension,
            "fleet_share": 0,
            "fix_capacity_unit_costs": penalty_costs / instance.stop_df.demand.max() 
        }
        instance.veh_type_df.loc[instance.veh_type_df.shape[0]] = dummy_veh_type

    return route_dict

def process_gurobi_result_data_scp(
        x,
        model,
        route_dict,
        runtime_SCP
        ):
    # Check if an optimal solution was found
    if model.Status == 3:
        raise ValueError(
            "The SCP Gurobi model is infeasible. Check the input route_dict if it is sufficient."
        )
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.status == GRB.TIME_LIMIT:
            logger.info(f"Gap to lower bound: {model.MIPGap * 100:.2f}%, Runtime SCP: {runtime_SCP}")
        multiple_visited_stops = dict()
        optimal_route_dict = dict()
        for r in route_dict:
            if x[r].X > 0.5:  # Check if the route is selected
                optimal_route_dict[route_dict[r]['route_id']] = route_dict[r]
        # check if "pendulum" routes are in the optimal solution
        if any("pendulum" in route["veh_type"] for route in optimal_route_dict.values()):
            is_feasible = False
        else:
            is_feasible = True
        # add information about stops that are visited multiple times in form of a dictionary that maps the stop_ids to the route_ids that visit it
        for constr in model.getConstrs():
            if 'Visit_' in constr.ConstrName:
                if constr.Slack < -0.99:
                    multi_visit_stop = constr.ConstrName.split("_")[1]
                    multiple_visited_stops[multi_visit_stop] = []
                    for route_key, route in optimal_route_dict.items():
                        if route["veh_type"] != "pendulum":
                            if multi_visit_stop in route["route"]:
                                multiple_visited_stops[multi_visit_stop].append(route_key)
    else:
        raise ValueError("No feasible solution found in the SPP Gurobi model.")
    return optimal_route_dict, multiple_visited_stops, runtime_SCP, is_feasible,

def solve_set_covering_model_with_gurobi(
    vrp_object: object,
    route_dict: dict,
    time_limit: int,
    warm_start: dict = None,
    num_threads: int = 1,
    infeas_penalty: float = 100,
    verbose: bool = False,
) -> tuple:
    """ """
    # Create a set of customers (excluding the depot)
    # filter based on depot_identifier (matches stop_id in vrp_object.stop_df)
    customer_set = set(
        vrp_object.stop_df["stop_id"][
            vrp_object.stop_df["stop_id"] != vrp_object.depot_identifier
        ]
    )

    # Create the optimization model
    model = gp.Model("set_covering_problem")

    # Decision variables: x_r = 1 if route r is sThenelected, 0 otherwise
    x = model.addVars(route_dict.keys(), vtype=GRB.BINARY, name="x")

    if warm_start is not None:
        # Set the initial solution for variables matching the keys in "warm_start"
        for key in warm_start:
            if key in x:
                x[key].start = 1  # Set the starting value to 1

    # Objective function: Minimize total cost
    model.setObjective(
        gp.quicksum(route_dict[r]["total_costs"] * x[r] for r in route_dict), GRB.MINIMIZE
    )

    # Constraint 1: Each customer is visited exactly once (excluding depot that is always at the beginning and end of the route)
    for customer in customer_set:
        model.addConstr(
            gp.quicksum(
                x[r] for r in route_dict if customer in route_dict[r]["route"][1:-1]
            )
            >= 1,
            # == 1,
            name=f"Visit_{customer}",
        )

    # Constraint 2: Vehicle availability constraints for each vehicle type
    for row_id, row_cols in vrp_object.veh_type_df.iterrows():
        model.addConstr(
            gp.quicksum(
                x[r]
                for r in route_dict
                if route_dict[r]["veh_type"] == row_cols["veh_type"]
            )
            <= row_cols["no_of_available"],
            name=f"VehicleLimit_{row_cols['veh_type']}",
        )
    model.update()
    # model parameters
    model.params.OutputFlag = 0
    if verbose:
        model.params.OutputFlag = 1
    model.params.Threads = num_threads
    # set time limit
    model.params.TimeLimit = time_limit
    start_time = datetime.now()
    model.optimize()
    runtime_SCP = round((datetime.now() - start_time).seconds,2)
    return process_gurobi_result_data_scp(x, model, route_dict, runtime_SCP)

# solve set partitioning problem with Gurobi based on generated root pool
def get_optimal_route_dict_from_root_pool_using_scp(
        vrp_object: object,
        route_pool_df: dict,
        warm_start_sol: dict,
        hp_SCP: "Hyperparameters"
):
    route_dict_SCP, multiple_visited_stops, runtime_SCP, is_feasible = solve_set_covering_model_with_gurobi(
        vrp_object=vrp_object,
        route_dict=route_pool_df.T.to_dict(),
        time_limit=hp_SCP.rt_limit_SCP,
        warm_start=warm_start_sol,
        verbose=hp_SCP.verbose,
    )
    return route_dict_SCP, multiple_visited_stops, runtime_SCP, is_feasible
