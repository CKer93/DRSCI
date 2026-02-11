import logging
from copy import deepcopy
import numpy as np
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.vrp import VRP_OBJECT

from core.evaluation import calculate_variable_costs

logger = logging.getLogger(__name__)

def spp_from_scp_least_savings(vrp_object: "VRP_OBJECT", route_dict_scp: dict, multiple_visited_stops: list):
    

    def remove_stop(route_dict, stop_id, stop_id_idx_map):
        """
        Remove stop_id from route_dict['route'] and also from route_dict['route_idx'].
        """
        route_dict["route"].remove(stop_id)
        stop_idx = stop_id_idx_map[stop_id]
        route_dict["route_idx"].remove(stop_idx)

    def recalc_costs(route_dict, distance_matrix, unit_costs):
        """
        Re-calculate the route's variable cost using the cached route_idx.
        (Potentially replaced by partial re-computation if you only remove a single stop.)
        """
        return calculate_variable_costs(
            distance_matrix=distance_matrix,
            route_idx=route_dict["route_idx"],
            veh_type_unit_costs=unit_costs,
        )

    # Build the vehicle type -> cost map
    veh_type_cost_map = {
        row["veh_type"]: row["variable_distance_unit_costs"]
        for _, row in vrp_object.veh_type_df.iterrows()
    }
    # Step B: Build a map from stop_id -> map_stop_id_idx
    stop_id_idx_map = vrp_object.map_stop_id_idx

    # Precompute route_idx for each route in route_dict_scp
    for rid, rdict in route_dict_scp.items():
        rdict["route_idx"] = [stop_id_idx_map[sid] for sid in rdict["route"]]


    # Make a copy for SPP
    route_dict_spp = deepcopy(route_dict_scp)

    # For each stop that is in multiple routes
    for stop_id, list_of_route_ids in multiple_visited_stops.items():
        current_best_var_cost_savings = np.inf
        best_route_id = None

        # 1) Temporarily remove stop from route_dict_scp's copy to test cost
        for route_id in list_of_route_ids:
            # route_dict_scp has a separate object from route_dict_spp, so we can test freely
            route_dict = route_dict_scp[route_id]
            old_route = list(route_dict["route"])           # might store a copy if needed
            old_route_idx = list(route_dict["route_idx"])

            # Remove the stop (faster approach)
            remove_stop(route_dict, stop_id, stop_id_idx_map)

            # Recompute cost
            veh_type = route_dict["veh_type"]
            unit_costs = veh_type_cost_map[veh_type]
            new_route_costs = recalc_costs(route_dict,
                                           vrp_object.distance_matrix,
                                           unit_costs)

            var_cost_savings = route_dict["var_costs"] - new_route_costs

            # Revert route_dict_scp changes
            route_dict["route"] = old_route
            route_dict["route_idx"] = old_route_idx

            if var_cost_savings < current_best_var_cost_savings:
                current_best_var_cost_savings = var_cost_savings
                best_route_id = route_id

        # In route_dict_spp, remove that stop from all routes except best_route_id
        for i, route_id in enumerate(list_of_route_ids):
            if route_id != best_route_id:

                # Try original route_id, fallback to new_route_id if not found
                if "+SPP" not in route_id:
                    new_route_id = re.sub(r"(?=\+vt)", "+SPP", route_id)
                else:
                    new_route_id = deepcopy(route_id)
                try:
                    # Try to access the original key
                    route_id_dict = route_dict_spp[route_id]  
                    route_dict_spp.pop(route_id, None)
                except KeyError:
                    # Use the new route_id if the original does not exist
                    route_id_dict = route_dict_spp[new_route_id]
                # Remove redundant stop from route
                remove_stop(route_id_dict, stop_id, stop_id_idx_map)
                # Mark this route as changed by savings heuristic
                route_dict_spp[new_route_id] = route_id_dict
                route_dict_spp[new_route_id]["route_id"] = new_route_id
                if len(route_dict_spp[new_route_id]['route']) < 3:
                    route_dict_spp.pop(new_route_id, None)                    
    return route_dict_spp

def try_to_include_excluded_stops(vrp_object: "VRP_OBJECT", route_dict_spp: dict, **kwargs):

    
    # 1. Collect unserved customers
    unserved_customers = {}
    for route_key, route_data in route_dict_spp.items():
        if "pendulum_" in route_key:
            # route looks like [depot, unserved_customer, ... depot]
            # so we pick route_data["route"][1]
            unserved_customers[route_data["route"][1]] = route_key

    if not unserved_customers:
        return route_dict_spp

    # 2. Precompute demands in a dictionary: {stop_id -> demand}
    stop_demands = dict(zip(vrp_object.stop_df["stop_id"], vrp_object.stop_df["demand"]))

    # 3. Try adding each unserved customer to an existing route
    for stop_id, pend_tour_key in unserved_customers.items():
        demand_for_this_customer = stop_demands[stop_id]
        customer_served = False
        
        # Iterate over possible routes
        for route_key, route_data in route_dict_spp.items():
            # skip pendulum routes themselves
            if "pendulum_" in route_key:
                continue
            
            # check capacity without re-summing every time
            if route_data["demand"] + demand_for_this_customer <= route_data["capacity"]:
                # feasible to add this stop right after the depot
                route_data["route"].insert(1, stop_id)
                route_data["demand"] += demand_for_this_customer
                customer_served = True
                break
        # if added successfully, remove the old pendulum route and rename route
        if customer_served:
            if "+IES" not in route_key:
                new_route_id = re.sub(r"(?=\+vt)", "+IES", route_key)
                route_data["route_id"] = new_route_id
                route_dict_spp[new_route_id] = route_data
                route_dict_spp.pop(route_key)
            del route_dict_spp[pend_tour_key]
    # Delete pendulum vehicle type if all pendulum tours have been resolved
    if not any("pendulum" in route_key for route_key in route_dict_spp):
        vrp_object.veh_type_df = vrp_object.veh_type_df[vrp_object.veh_type_df["veh_type"] != "pendulum"]
    return route_dict_spp