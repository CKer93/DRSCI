import copy
import logging
from math import ceil
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.vrp import VRP_OBJECT

from core.solving.solve_gspi_with_pyvrp_n_gurobi import giant_tours_n_partitioning_MILP

logger = logging.getLogger(__name__)


def duplicate_routes_for_larger_vehicle_types(route_dict_long, veh_type_id_list):
    """"""
    # Create a list to store the new items to be added
    new_entries = []

    # Iterate through each key in the dictionary
    for key_id, value in route_dict_long.items():
        original_veh_type = value["veh_type"]

        # Create new entries for each veh_type greater than the current one
        for new_veh_type in veh_type_id_list:
            if new_veh_type > original_veh_type:
                new_value = value.copy()  # Create a copy of the original value
                new_value["veh_type"] = new_veh_type  # Update the veh_type
                new_key = (
                    f"{key_id}+vt{new_veh_type}"  # Create a new key to avoid collisions
                )
                new_entries.append((new_key, new_value))

    # Add the new entries to the original dictionary
    route_dict_long.update(new_entries)
    return route_dict_long

def GSPI(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_R: "Hyperparameters",
        hp_gurobi: "Hyperparameters",
        hp_I: "Hyperparameters"
):
    sol_dict = giant_tours_n_partitioning_MILP(
        vrp_object,
        gs,
        hp_R,
        hp_gurobi,
        hp_I
    )
    if sol_dict.get("GSPI", None) is not None:
        return sol_dict["GSPI"]["route_dict"], sol_dict["GS"]["runtime"]+sol_dict["GSP"]["runtime"]+sol_dict["GSPI"]["runtime"]
    if sol_dict.get("Infeas_GSP", None) is not None:
        return sol_dict["Infeas_GSP"]["route_dict"], sol_dict["Infeas_GSP"]["runtime"]

def PCVRP_fill(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_I: "Hyperparameters",
        hp_R: "Hyperparameters",
        hp_RP: "Hyperparameters"
    ) -> tuple[dict, float]:

    vrp_object_pc_sol_strategy = copy.deepcopy(vrp_object)
    # update instance type to enable PC solving with pyvrp
    vrp_object_pc_sol_strategy.type += "+PC"
    depot_idx = vrp_object_pc_sol_strategy.map_stop_id_idx[vrp_object_pc_sol_strategy.depot_identifier]
    route_dict_long = {}
    # index of vrp_object.stop_df as list
    list_of_unassigned_stop_ids = vrp_object_pc_sol_strategy.stop_df.index.tolist()
    runtime_R_t = 0
    # solve the heterogeneous fleet problem multiple times, once for each vehicle type
    # (i.e., it becomes a homogeneous fleet problem)
    for veh_type_row in vrp_object_pc_sol_strategy.veh_type_df.iterrows():
        vrp_object_sub_veh_type = copy.deepcopy(vrp_object_pc_sol_strategy)
        # make vehicle type homogeneous (iterate row by row) and ensure
        vrp_object_sub_veh_type.veh_type_df = pd.DataFrame(veh_type_row[1]).T
        vrp_object_sub_unassigned_stops = vrp_object_sub_veh_type.create_sub_vrp(
            list_of_idx_to_keep=list_of_unassigned_stop_ids,
            sup_prob_name_add_on="", # NOTE: Done by pyvrp-solver f"+vt{veh_type_row[1]['veh_type']}",
            update_veh_availablity=False,
        )
        # Add prizes to the sub-problem based on the given strategy in the hps yaml file
        # Keep in for loop to adapt the prizes based on the available vehicles in the subproblems
        vrp_object_sub_unassigned_stops.transform_to_prize_collecting_problem(
            prize_setting_strategy=hp_RP.subproblem_type.split("_")[1],
        )
        hp_R_vt_pc = copy.deepcopy(hp_R)
        hp_R_vt_pc.max_runtime = vrp_object_sub_unassigned_stops.dimension * hp_R_vt_pc.max_runtime_dimension_multiplier
        route_dict_R, runtime_R, _ = vrp_object_sub_unassigned_stops.solve(gs=gs, hp_R=hp_R_vt_pc)
        runtime_R_t += runtime_R
        route_dict_long.update(route_dict_R)
        # update list of unassigned stops by removing stops that have been assigned to a route
        stop_idx_route = set(
            [
                vrp_object_pc_sol_strategy.map_stop_id_idx[stop_id]
                for value in route_dict_R.values()
                for stop_id in value["route"][1:-1]
            ]
        )
        list_of_unassigned_stop_ids = list(
            set(list_of_unassigned_stop_ids) - stop_idx_route
        )
        if list_of_unassigned_stop_ids == [
            depot_idx
        ]:
            break
    # Transform overall problem into prize collecting to enable local search on it based on the created routes
    vrp_object_pc_sol_strategy.transform_to_prize_collecting_problem(
            prize_setting_strategy=hp_RP.subproblem_type.split("_")[1],
        )
    route_dict_long, runtime_I_t = vrp_object_pc_sol_strategy.improve(
        route_dict=route_dict_long,
        hp_I=hp_I,
        pruned_list_stop_neigh=[],
        seed=gs.seed,
        dec_prec=gs.dec_prec
    )
    return route_dict_long, runtime_R_t+runtime_I_t

def VRP_indiv(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_R: "Hyperparameters",
        hp_RP: "Hyperparameters"
    ) -> tuple[dict, float]:

    route_dict_long = {}
    runtime_R_t = 0
    hp_R_vt = copy.deepcopy(hp_R)
    hp_R_vt.max_runtime = int(vrp_object.dimension * hp_R_vt.max_runtime_dimension_multiplier)
    # solve the heterogeneous fleet problem multiple times, once for each vehicle type
    # (i.e., it becomes a homogeneous fleet problem)
    for row_id, veh_type_row in vrp_object.veh_type_df.iterrows():
        vrp_object_sub = copy.deepcopy(vrp_object)
        # make vehicle type homogeneous (iterate row by row)
        vrp_object_sub.veh_type_df = pd.DataFrame(veh_type_row).T
        # ensure that the number of available vehicles is sufficient (i.e., pendulum tours)
        vrp_object_sub.veh_type_df["no_of_available"] = vrp_object_sub.dimension
        # temporarly remove customers with demand larger than the capacity of the homogeneous vehicle type
        if veh_type_row["capacity"] < vrp_object_sub.stop_df.demand.max():
            vrp_object_sub = vrp_object_sub.resolve_capacity_infeasibility()
            if vrp_object_sub.dimension == 0:
                continue
        # ------------------------------------------------
        route_dict_R, runtime_R, _ = vrp_object_sub.solve(gs=gs, hp_R=hp_R_vt)
        route_dict_long.update(route_dict_R)
        runtime_R_t += runtime_R
    return route_dict_long, runtime_R_t

def create_routes(
        vrp_object: "VRP_OBJECT",
        gs:"Hyperparameters",
        hp_gurobi:"Hyperparameters",
        hp_I:"Hyperparameters",
        hp_R: "Hyperparameters",
        hp_RP: "Hyperparameters"
    ) -> tuple[dict, float]:

    if "GSPI" == hp_RP.subproblem_type:
        return GSPI(vrp_object, gs=gs, hp_R=hp_R, hp_gurobi=hp_gurobi, hp_I=hp_I)
    elif "VRP_indiv" == hp_RP.subproblem_type:
        return VRP_indiv(vrp_object, gs=gs, hp_R=hp_R, hp_RP=hp_RP)
    elif "PCVRP" in hp_RP.subproblem_type:
        return PCVRP_fill(vrp_object, gs=gs, hp_I=hp_I, hp_R=hp_R, hp_RP=hp_RP)
    elif "HVRP" in hp_RP.subproblem_type:
        hp_R_HVRP = copy.deepcopy(hp_R)
        hp_R_HVRP.max_runtime = vrp_object.dimension * hp_R.max_runtime_dimension_multiplier * 2
        route_dict, runtime_R, _ = vrp_object.solve(gs=gs, hp_R=hp_R_HVRP)
        return route_dict, runtime_R
    else:
        raise ValueError(f"{hp_RP.subproblem_type} is not available yet.")
