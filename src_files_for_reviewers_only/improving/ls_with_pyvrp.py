from copy import deepcopy
import logging
import numpy as np
import time
from typing import TYPE_CHECKING

from pyvrp import CostEvaluator, RandomNumberGenerator, Solution, Route
from pyvrp.search import (
    compute_neighbours,
    LocalSearch,
    NeighbourhoodParams,
    NODE_OPERATORS,
    ROUTE_OPERATORS,
)

if TYPE_CHECKING:
    from core.config import Hyperparameters

logger = logging.getLogger(__name__)

def transform_route_dict_to_list_of_pyvrp_routes(
    map_stop_id_idx: dict,
    pyvrp_model_data: object,
    route_dict: dict
) -> list:
    """
    Transforms a route dictionary into a list of PyVRP route objects.

    Args:
        map_stop_id_idx (dict): A mapping of stop IDs to their respective indices in the stop_df dataframe.
        pyvrp_model_data (object): The pyVRP model data object containing vehicle types
            and other relevant information.
        route_dict (dict): A dictionary where keys are identifiers and values contain
            'route' (a list of stops) and 'veh_type' (the name of the vehicle type).

    Returns:
        list: A list of route objects built from the provided route dictionary.
    """

    veh_type_map = {
        veh_type.name: i for i, veh_type in enumerate(pyvrp_model_data.vehicle_types())
    }

    routes_data = [
        Route(
            pyvrp_model_data,
            [map_stop_id_idx[stop] for stop in value["route"][1:-1]],
            veh_type_map[value["veh_type"]]
        )
        for value in route_dict.values()
        if len(value["route"]) # Make sure that non empty routes are added
    ]
    return routes_data


def run_pyvrp_local_search(
    pyvrp_model_data: object,
    incumbent_solution: list,
    hp_I: "Hyperparameters",
    pruned_list_stop_neigh: list,
    seed: int,
    dec_prec: int = 4
) -> tuple[object, float]:
    """Explores a granular neigbourhood (specific_neighborhood) in a very efficient manner using user-provided node and route operators. Calls Local Search components from pyvrp module.

    Args:

    Raises:
        ValueError: If solution obtained not feasible. Indicates that penalities must be set higher in CostEvaluator.

    Returns:

        dri_pyvrp_routes_list(list[list[int]]): A list containing the (improved) solution (DRI solution).
        final_ls_time(float): Duration of Local Search algorithm (in seconds)
    """
    def check_for_improvement(sol, current_best_sol, current_best_sol_obj):
        sol_obj = sol.distance_cost() + sol.fixed_vehicle_cost()
        if sol_obj < current_best_sol_obj:
            return sol, sol_obj, True
        else:
            return current_best_sol, current_best_sol_obj, False
    # set ls parameters:
    # if pruned_list_stop_neigh is not provided, compute neighbours for the local search without pruning (i.e., nb_granular=pyvrp_model_data.num_clients-1)
    if pruned_list_stop_neigh == []:
        pruned_list_stop_neigh = compute_neighbours(
            pyvrp_model_data,
            params=NeighbourhoodParams(
                weight_wait_time=0.2,
                weight_time_warp=1.0,
                nb_granular=max(1,pyvrp_model_data.num_clients-1),
                symmetric_proximity=True,
                symmetric_neighbours=False,
            ),
        )
    # create rng: shuffles operators and nodes/routes rng (decides in which order nodes/routes and operators are tried in ls.search and ls.intensify)
    rng = RandomNumberGenerator(seed=seed)
    # create local search class
    ls = LocalSearch(pyvrp_model_data, rng, pruned_list_stop_neigh)
    # ls operators
    # add node operators to ls (only impacts ls.search)
    # providing no node operators to ls will skip ls.search
    for (
        node_op
    ) in (
        NODE_OPERATORS
    ):  # adds ExchangeNM(10,20,30,11,21,31,22,32,33),MoveTwoClientsReversed,TwoOpt
        ls.add_node_operator(node_op(pyvrp_model_data))
    # add route operators to ls (only impacts ls.intensify)
    # providing no route operators to ls will skip ls.intensify
    for route_op in ROUTE_OPERATORS:  # adds RelocateStar,SwapRoutes,SwapStar
        ls.add_route_operator(route_op(pyvrp_model_data))

    dr_sol = Solution(data=pyvrp_model_data, routes=incumbent_solution)
    if not dr_sol.is_feasible() or not dr_sol.is_complete():
        raise ValueError(f"The solution for the problem with {pyvrp_model_data.num_clients} customers is infeasible")

    # define cost evaluator (penalities must be set high enough to ensure local search leads to feasible solution)
    cost_evaluator = CostEvaluator(
        load_penalty=hp_I.infeas_penalty,
        tw_penalty=hp_I.infeas_penalty,
        dist_penalty=hp_I.infeas_penalty,
    )

    ## run pyvrp local search (ls defined above)
    dri_sol = deepcopy(dr_sol)
    curr_best_obj = dri_sol.distance_cost() + dri_sol.fixed_vehicle_cost()
    improvement_found_in_ls = False
    improvement_found_in_intensify = False
    run_ls = True
    start_time = time.time()
    while run_ls:
        for method in hp_I.ls_method_list:
            if method == "search":
                # search method runs all NODE_OPERATORS added to ls
                sol = ls.search(dri_sol, cost_evaluator)  # restricted by neighbours
                dri_sol, curr_best_obj, improvement_found_in_ls = (
                    check_for_improvement(sol, dri_sol, curr_best_obj)
                )
            elif method == "intensify":
                # intensify method runs all ROUTE_OPERATORS added to ls
                sol = ls.intensify(
                    dri_sol, cost_evaluator, overlap_tolerance=hp_I.route_overlap
                )  # controls the amount of overlap needed before two routes are evaluated
                dri_sol, curr_best_obj, improvement_found_in_intensify = (
                    check_for_improvement(sol, dri_sol, curr_best_obj)
                )
            # check if runtime limit has been reached
            curr_time = time.time()
            if curr_time - start_time > hp_I.runtime:
                print("PYVRP local search reached maxmimum runtime limit.")
                run_ls = False
                break
        # if neither search nor intensify improved the solution, stop local search run
        if not improvement_found_in_ls and not improvement_found_in_intensify:
            curr_time = time.time()
            run_ls = False

    # obtain final runtime and retrieve new (hopefully improved) solution
    final_ls_time = round(curr_time - start_time, dec_prec)

    # check if final solution is feasible
    if not dri_sol.is_feasible():
        return dr_sol, final_ls_time, False

    if dr_sol == dri_sol:
        sol_has_changed = False
    else:
        sol_has_changed = True
    return dri_sol, final_ls_time, sol_has_changed
