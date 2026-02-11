import copy
from typing import TYPE_CHECKING

from core.data_manipulation.data_processing import (
    hetero_veh_type_assignment_to_subproblems,
)
from core.evaluation import VRP_SOLUTION
from core.utils import util_create_subproblem_df

if TYPE_CHECKING:
    from core.vrp import VRP_OBJECT

def find_inital_feasible_solution(instance: "VRP_OBJECT", sol_method, seed, dec_prec, multiplier):
    # Step 1: Try to find a feasible solution using the pyvrp implementation for HVRP(TW) instances
    hp_R = {
        "solver": "pyvrp",
        "stop_criteria_dict": {
            "no_improvement": 5000,
            "runtime": 60,
        },
        "pruning": True,
    }
    initial_feas_rout_dict, runtime_R_t, _ = instance.solve(hp_R, seed, multiplier)
    # Step 2: If no feasible solution is found, use the our custom heuristic to find a feasible solution
    if len(initial_feas_rout_dict) == 0:
        sol = create_initial_feasible_route_dict(instance)
        init_route_counter = 0
        for k, v in sol.items():
            instance_base_copy = copy.deepcopy(instance)
            list_of_stop_ids_to_keep = v["route"][1:-1]
            list_of_idx_to_keep = [
                instance.map_stop_id_idx[i] for i in list_of_stop_ids_to_keep
            ]
            # get all stops that are in a pendulum tour (except the first and last stop)
            instance_base_copy.name = f"{instance.name}_{k}"
            instance_base_copy.stop_df["c_idx"] = "-2"
            # set c_idx for row where stop_id == instance_base_copy.depot_identifier to "-1"
            instance_base_copy.stop_df.loc[
                instance_base_copy.stop_df.stop_id == instance_base_copy.depot_identifier,
                "c_idx",
            ] = "-1"
            # Create a subproblem_df with the cluster information
            subproblem_df = util_create_subproblem_df(instance_base_copy.stop_df, dec_prec)
            if instance_base_copy.veh_type_df.shape[0] > 1:
                instance_base_copy.veh_type_to_subprob_dict = (
                    hetero_veh_type_assignment_to_subproblems(
                        instance, subproblem_df, "pend", seed
                    )
                )
            instance_sub_infeas = instance_base_copy.create_sub_vrp(
                list_of_idx_to_keep=list_of_idx_to_keep, sup_prob_name_add_on="-2"
            )
            # only keep the row in veh_type_df where veh_type == v['veh_type']
            instance_sub_infeas.veh_type_df = instance_sub_infeas.veh_type_df[
                instance_sub_infeas.veh_type_df.veh_type == v["veh_type"]
            ]
            # reset the number of available vehicles to 1
            instance_sub_infeas.veh_type_df.no_of_available = 1
            hp_R = {
                "solver": "pyvrp",
                "stop_criteria_dict": {
                    "no_improvement": 100,
                    "runtime": 1,
                },
                "pruning": True,
            }
            route_dict_R, runtime_R, _ = instance_sub_infeas.solve(hp_R, seed, multiplier)
            runtime_R_t += runtime_R
            init_route_counter += 1
            # update name of route_dict_R
            route_dict_R[f"init-{init_route_counter}"] = route_dict_R.pop("0+c-2")

            initial_feas_rout_dict.update(route_dict_R)

    solution_dicts = {"init": {"route_dict": initial_feas_rout_dict, "runtime": 0}}
    instance_sol = VRP_SOLUTION(sol_method, seed, dec_prec, multiplier)
    instance_sol.create_sol_obj_after_run(instance, {}, solution_dicts)
    return instance_sol.route_dict_init, runtime_R_t


def create_initial_feasible_route_dict(instance):
    customer_data = instance.stop_df
    vehicle_data = instance.veh_type_df
    distance_matrix = instance.distance_matrix
    # Convert customer data to list of dictionaries
    customers = customer_data.to_dict("records")

    # Convert vehicle data to list of dictionaries
    vehicles = vehicle_data.to_dict("records")

    # Sort customers by demand in descending order (excluding the depot)
    customers_sorted = sorted(customers[1:], key=lambda x: x["demand"], reverse=True)

    # Sort vehicles by capacity in descending order
    vehicles_sorted = sorted(vehicles, key=lambda x: x["capacity"], reverse=True)

    # Initialize vehicles with empty routes
    vehicle_routes = []
    for vehicle in vehicles_sorted:
        for _ in range(vehicle["no_of_available"]):
            vehicle_routes.append(
                {
                    "veh_type": vehicle["veh_type"],
                    "capacity": vehicle["capacity"],
                    "route": [],
                    "remaining_capacity": vehicle["capacity"],
                }
            )

    # Assign customers to vehicles iteratively
    for stop_idx, customer in enumerate(customers_sorted):
        customer_assigned = False
        for vehicle in vehicle_routes:
            if vehicle["remaining_capacity"] >= customer["demand"]:
                vehicle["route"].append(customer["stop_id"])
                vehicle["remaining_capacity"] -= customer["demand"]
                customer_assigned = True
                break
        if not customer_assigned:
            raise ValueError(
                f"Customer {customer['stop_id']} with demand {customer['demand']} could not be assigned to any vehicle"
            )

    # delete vehicles with no route
    vehicle_routes = [vehicle for vehicle in vehicle_routes if vehicle["route"]]

    route_dict = {}
    for idx, vehicle in enumerate(vehicle_routes):
        # add instance.depot_identifier to the route at the start and end
        vehicle["route"] = (
            [instance.depot_identifier] + vehicle["route"] + [instance.depot_identifier]
        )
        route_dict[f"init-{idx}"] = {
            "veh_type": vehicle["veh_type"],
            "route": vehicle["route"],
        }
    return route_dict