from copy import deepcopy
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.evaluation import VRP_SOLUTION
    from core.vrp import VRP_OBJECT

from core.clustering.assign_vehicles_to_clusters import assign_vehicles_to_subproblems
from core.clustering.similarity_metric import caculate_similarity_matrix
from core.evaluation import create_sol_obj
from core.solving.solve_scp_to_spp import spp_from_scp_least_savings, try_to_include_excluded_stops
from core.solving.solve_scp_with_gurobi import create_penalized_dummy_pendulum_tours, get_optimal_route_dict_from_root_pool_using_scp
from core.utils import util_remove_duplicate_route_vt_pairs_in_route_pool

logger = logging.getLogger(__name__)

def decompose_vrp_object(
        vrp_object: "VRP_OBJECT", 
        hp_D:"Hyperparameters", 
        seed: int, dec_prec: int = 4,
        **kwargs
    ) -> tuple["VRP_OBJECT", pd.DataFrame, float]:

    vrp_object.similarity_matrix, feature_array = caculate_similarity_matrix(
        hp_D,
        vrp_object.stop_df,
        vrp_object.distance_matrix,
        vrp_object.veh_base_capacity,
        dec_prec
    )

    subproblem_df, runtime_D = vrp_object.decompose(
        feature_array,
        hp_D,
        seed,
        dec_prec,
        route_df=kwargs["route_df"]
    )

    return vrp_object, subproblem_df, runtime_D
    
def set_routing_strategy_n_no_of_clusters(
    vrp_object: "VRP_OBJECT",
    cluster_range_in_current_bks_sol: list, 
    hp_R: "Hyperparameters",
    hp_RP: "Hyperparameters", 
    hp_M: "Hyperparameters",
    enumerator_solver_list:int=-1,
    enumerator_cluster_list:int=-1
    ):
    if "random" in hp_M.no_of_cluster_list_strategy:
        enumerator_solver_list = random.randint(0, len(hp_RP.subproblem_type_list)-1)
        enumerator_cluster_list = random.randint(0, len(hp_M.list_no_of_clusters)-1)

        number_of_clusters = hp_M.list_no_of_clusters[enumerator_cluster_list]

    elif "ordered" in hp_M.no_of_cluster_list_strategy:
        enumerator_solver_list += 1
        if enumerator_solver_list >= len(hp_RP.subproblem_type_list):
            enumerator_solver_list = 0
        if enumerator_solver_list == 0:
            enumerator_cluster_list += 1
        if enumerator_cluster_list >= len(hp_M.list_no_of_clusters):
            random_cluster_no_picker = random.randint(0, len(cluster_range_in_current_bks_sol)-1)
            number_of_clusters = cluster_range_in_current_bks_sol[random_cluster_no_picker]
            hp_R.max_runtime_dimension_multiplier = hp_R.max_runtime_dimension_multiplier * 2
        else:
            number_of_clusters = hp_M.list_no_of_clusters[enumerator_cluster_list]
        

    hp_RP.subproblem_type = hp_RP.subproblem_type_list[enumerator_solver_list]
    return hp_RP, hp_R, number_of_clusters, enumerator_solver_list, enumerator_cluster_list

def solve_with_matheuristic(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_D_initial: "Hyperparameters",
        hp_gurobi: "Hyperparameters",
        hp_I: "Hyperparameters",
        hp_M: "Hyperparameters",
        hp_P: "Hyperparameters",
        hp_R_initial: "Hyperparameters",
        hp_RP_initial: "Hyperparameters",
        hp_SCP: "Hyperparameters",
        cluster_method: str = "vbc"
    ) -> tuple["VRP_OBJECT", "VRP_SOLUTION"]:
    """
    Umbrella function for my matheuristic algorithm
    """
    # ---------------------------------------------------------------------------------------
    # Initialize variables
    best_solution_value = np.inf
    current_best_sol_object = None
    route_df_for_clustering = None
    runtime_iteration = 0
    runtime_pool_creation_total = 0
    total_runtime = 0
    nb_iter_no_imprv = 0
    nb_iter = 1
    last_cluster_method = 'vbc'
    unique_vrp_object_name = deepcopy(vrp_object.name)
    enumerator_cluster_list = -1
    enumerator_solver_list = -1
    number_of_solvers = len(hp_RP_initial.subproblem_type_list)
    cluster_range_in_current_bks_sol = [1]
    # ---------------------------------------------------------------------------------------
    number_of_cluster_no_options = len(hp_M.list_no_of_clusters)
    true_total_start_time = datetime.now()
    true_total_runtime = 0
    #Track runtimes
    runtime_dict = {
        "instance_name": vrp_object.name,
        "seed": gs.seed,
        "set_routing_n_no_of_cluster_params": 0,
        "decompose_n_veh_assignment": 0,
        "route_GSPI": 0,
        "route_HVRP": 0,
        "route_PC": 0,
        "route_VRP_indiv": 0,
        "route_pool_post_processing": 0,
        "solve_SCP": 0,
        "scp_to_spp": 0,
        "improve_complete_HVRP_sol": 0,
        "post_rpocessing_iteration": 0,
    }

    route_pool = create_penalized_dummy_pendulum_tours(
        route_dict={},
        instance=vrp_object,
        infeas_penalty=100,
        dec_prec=gs.dec_prec
    )

    # Start Matheuristic
    while nb_iter_no_imprv < hp_M.no_of_iterations_without_improvement and true_total_runtime < hp_M.maximum_runtime:
        # Track runtimes
        start_set_routing_n_no_of_cluster_params = datetime.now()
        # Initialize vrp_object and hyperparameters that will be modified during the matheuristic algorithm
        vrp_object_iter = deepcopy(vrp_object)
        hp_D=deepcopy(hp_D_initial)
        hp_R=deepcopy(hp_R_initial)
        hp_RP=deepcopy(hp_RP_initial)
        hp_RP, hp_R, number_of_clusters, enumerator_solver_list, enumerator_cluster_list = set_routing_strategy_n_no_of_clusters(
            vrp_object=vrp_object_iter,
            cluster_range_in_current_bks_sol=cluster_range_in_current_bks_sol,
            hp_R=hp_R,
            hp_RP=hp_RP,
            hp_M=hp_M,
            enumerator_solver_list=enumerator_solver_list,
            enumerator_cluster_list=enumerator_cluster_list
        )

        hp_D.number_of_clusters =  number_of_clusters
        # Initalize bool variable if route pool must be trunctated because it became too large for efficient solving
        clean_route_pool = False
        if current_best_sol_object:
            if cluster_method =="vbc":
                hp_D.cluster_strategy = "vbc"
            elif cluster_method == "rbc":
                hp_D.cluster_strategy = "rbc"
                route_df_for_clustering = deepcopy(current_best_sol_object.solutions.I.route_df)
            elif random.random() < 0.5: # last_cluster_method == "vbc":
                hp_D.cluster_strategy = "rbc"
                route_df_for_clustering = deepcopy(current_best_sol_object.solutions.I.route_df)
            else:
                hp_D.cluster_strategy = "vbc"
        else:
            hp_D.cluster_strategy = "vbc"

        if random.random() < 0.5:
            # Change angle in 50% of the time (default is "ccw")
            hp_D.stop_depot_angle_orientation = "cw"
        
        # ===== DECOMPOSE INSTANCE =====
        # Add iteration counter to vrp_object name. Needed for unique route names
        vrp_object_iter.name = f"{unique_vrp_object_name}+nbI{nb_iter}+{hp_D.cluster_strategy}-{hp_D.stop_depot_angle_orientation}+C{number_of_clusters}+sol-{hp_RP.subproblem_type}"
        # 
        runtime_dict["set_routing_n_no_of_cluster_params"] += (datetime.now() - start_set_routing_n_no_of_cluster_params).seconds
        # QUICKFIX to avoid "break of method" when an error occurs
        try:
            start_decompose_n_veh_assignment = datetime.now()
            vrp_object_iter, subproblem_df, runtime_D = decompose_vrp_object(
                vrp_object=vrp_object_iter,
                hp_D=hp_D,
                seed=gs.seed,
                dec_prec=gs.dec_prec,
                route_df = route_df_for_clustering
            )    
            # Assign fleet to subproblems (only relevant for LIMITED fleet VRP variants, i.e., HFVRP(TW))
            vrp_object_iter, veh_type_cluster_dict, multiple_assignments, runtime_vt_to_c = assign_vehicles_to_subproblems(
                vrp_object_iter,
                hp_gurobi=hp_gurobi,
                fthr=hp_P.fthr,
                multiple_stop_to_cluster_assignments_possible=hp_D.multiple_stop_to_cluster_assignments_possible,
                assignment_rule = hp_D.vehicle_to_cluster_assignment_rule
            )
            vrp_object_iter.veh_type_cluster_dict = veh_type_cluster_dict
            # Track runtime of decomposition step
            total_runtime += runtime_D + runtime_vt_to_c
            runtime_dict["decompose_n_veh_assignment"] += (datetime.now() - start_decompose_n_veh_assignment).seconds
            # ===== GENERATE ROUTES FOR ROUTE POOL =====       
            start_routing = datetime.now()
            if "PCVRP" == hp_RP.subproblem_type:
                hp_RP.subproblem_type += "_" + hp_RP.prizing_strategies[random.randint(0, len(hp_RP.prizing_strategies)-1)]
                if random.random() < 0.5:             
                    # change order of veh_type_df rows (mainly needed for HVRP instances; allow more variablity in PCVRP routing)
                    vrp_object_iter.veh_type_df = vrp_object_iter.veh_type_df.iloc[::-1]

            route_pool_sub, runtime_pool_creation = vrp_object_iter.create_route_pool(
                gs=gs,
                hp_gurobi=hp_gurobi,
                hp_I=hp_I,
                hp_R=hp_R,
                hp_RP = hp_RP
            )
            end_routing = (datetime.now() - start_routing).seconds
            if "GSPI" == hp_RP.subproblem_type:
                runtime_dict["route_GSPI"] += end_routing
            if "HVRP" == hp_RP.subproblem_type:
                runtime_dict["route_HVRP"] += end_routing
            elif "PC" in hp_RP.subproblem_type:
                runtime_dict["route_PC"] += end_routing
            elif "VRP_indiv" == hp_RP.subproblem_type:
                runtime_dict["route_VRP_indiv"] += end_routing

            # Track runtime of route_pool postprocessing
            start_time_route_pool_postprocess = datetime.now()
            route_pool.update(route_pool_sub)
            # remove duplicate routes (vehicle type and route (ordered))
            route_pool = util_remove_duplicate_route_vt_pairs_in_route_pool(route_pool)
            sol_dict_scp = {
                "SCP": {
                    "route_dict": route_pool,
                    "runtime": runtime_pool_creation_total,
                    "route_dict_matches_df_format": False,
                }
            }
            instance_sol_pool = create_sol_obj(vrp_object_iter, gs, {}, sol_dict_scp)
            runtime_route_pool_postprocess = round((datetime.now() - start_time_route_pool_postprocess).seconds, 2)
            # Track cumulative runtime to create routes for route pool
            runtime_pool_creation_total += runtime_pool_creation
            # Add tim to create routes to total runtime in every iteration
            total_runtime += runtime_pool_creation + runtime_route_pool_postprocess
            runtime_dict["route_pool_post_processing"] += (datetime.now() - start_time_route_pool_postprocess).seconds
            # ===== SOLVE SET COVERING PROBLEM =====
            start_solve_SCP = datetime.now() 
            route_dict_scp, multiple_visited_stops, runtime_solving_SCP, is_feasible = (
                get_optimal_route_dict_from_root_pool_using_scp(
                    vrp_object=vrp_object_iter,
                    route_pool_df=instance_sol_pool.solutions.SCP.route_df,
                    warm_start_sol=None,
                    hp_SCP=hp_SCP
                )
            )
            # Add solving time Gurobi SCP to the total runtime of route pool creation and finding solution to SCP
            sol_dict_scp["SCP"]["runtime"] = sol_dict_scp["SCP"]["runtime"] + runtime_solving_SCP

            # Track runtime of the effective/net (optimization) time to solve the SCP
            total_runtime += runtime_solving_SCP
            # Set bool variable to True if Gurobi needs extensive computation time to find the optimum solution
            if runtime_solving_SCP >= hp_RP.Gurobi_SCP_runtime_threshold_before_pruning_route_pool:
                clean_route_pool = True
            runtime_dict["solve_SCP"] += (datetime.now() - start_solve_SCP).seconds
            # ===== TRANSFORM SCP INTO SPP to create ======
            # I.e., find (feasible) solution to the COMPLETE/ORIGINAL vrp_object
            start_time_scp_to_spp = datetime.now()
            # Resolve multiple visited stops by least savings heuristic approach
            if multiple_visited_stops:
                route_dict_spp = spp_from_scp_least_savings(vrp_object_iter, route_dict_scp, multiple_visited_stops)
                route_pool.update(route_dict_spp)
            else:
                route_dict_spp = deepcopy(route_dict_scp)
            # TRY (!) to repair SPP if it is not feasible, i.e., if "pendulum" routes are in the solution route dict, i.e., route_dict_spp
            if not is_feasible:
                route_dict_spp = try_to_include_excluded_stops(vrp_object_iter, route_dict_spp, verbose=gs.verbose)                       
                route_pool.update(route_dict_spp)
            #else:
                # Remove artificial pendulum tours from vrp_object_iter if the route_dict_spp is a FEASIBLE solution to the COMPLETE/ORIGINAL vrp_object
            #    vrp_object_iter.veh_type_df = vrp_object_iter.veh_type_df[vrp_object_iter.veh_type_df["veh_type"] != "pendulum"]
            # Track time of SCP to SPP transformation step
            runtime_spp_postprocess = round((datetime.now() - start_time_scp_to_spp).seconds, 2)
            total_runtime += runtime_spp_postprocess
            runtime_dict["scp_to_spp"] += (datetime.now() - start_time_scp_to_spp).seconds
            # ===== TRY LS TO OVERAL/COMPLETE SOLUTION =====
            start_improve_complete_HVRP_sol = datetime.now()
            # I.e., Applies pyvrp.ls to route_dict_spp. If it is infeasible, this might also repair the solution
            # Apply improvement step to overall solution found by SPP
            route_dict_I, runtime_I = vrp_object_iter.improve(
                route_dict=route_dict_spp,
                hp_I=hp_I,
                pruned_list_stop_neigh=[],
                seed=gs.seed,
                dec_prec=gs.dec_prec
            )
            if route_dict_I != route_dict_spp:
                # add improved routes to route pool
                route_pool.update(route_dict_I)
            total_runtime += runtime_I
            runtime_dict["improve_complete_HVRP_sol"] += (datetime.now() - start_improve_complete_HVRP_sol).seconds
            # ===== POSTPROCESSING TO STORE KPIS OF ITERATION =====
            start_post_processing_iteration = datetime.now()
            sol_dict = {
                "SPP" : {
                    "route_dict": route_dict_spp,
                    "runtime":runtime_spp_postprocess,
                    "route_dict_matches_df_format": False,
                }
            }
            sol_dict["I"] = {
                "route_dict": route_dict_I,
                "runtime": runtime_I,
                "route_dict_matches_df_format": False,
                }
            static_hps = {
                "hp_gurobi": hp_gurobi,
                "hp_I": hp_I,
                "hp_M": hp_M,
                "hp_P": hp_P,
                "hp_R": hp_R,
                "hp_SCP": hp_SCP,
            }
            vrp_object_sol = create_sol_obj(vrp_object_iter, gs, static_hps, sol_dict)
        #"""
        except Exception as e:
            logger.info(f"{vrp_object_iter.name} failed: because of {e}; Continue with next iter")
            continue
        # ===== PREPARE NEXT ITERATION =====
        nb_iter += 1
        # Store current best solution found so far
        if vrp_object_sol.solutions.I.total_costs < best_solution_value:
            current_best_sol_object = deepcopy(vrp_object_sol)
            best_solution_value = vrp_object_sol.solutions.I.total_costs
            nb_iter_no_imprv = 0
            current_best_sol_object.total_runtime_until_convg = round(total_runtime,2)
            # create a list of clusters based on the largest "K" valuable to add to the SPP route pool
            max_int = vrp_object_sol.solutions.SPP.route_df['route_id'].apply(
                lambda s: int(s.split("+C")[1].split("+")[0]) if "pend" not in s else 1
            ).max()
            # Set min avg. cluster size for "intensification phase" to 5% (so max no of clusters always 20)
            max_int = min(max_int, 20)
            cluster_range_in_current_bks_sol = list(range(1, max_int+1))
        else:
            nb_iter_no_imprv += 1
        # Truncate route_pool if it got too large for an efficient solving by SCP Gurobi model
        if clean_route_pool:
            logging.info(f"{vrp_object.name.lower()} - Route pool of size {len(route_pool)} cannot be solved efficiently in SCP. Reset to current best solution route dict")
            try:
                route_pool = deepcopy(current_best_sol_object.solutions.I.route_dict)
            except Exception as e:
                logging.info(f"{e}: no feasible solution available reset route_pool to empty dict")
                route_pool = {}
        runtime_dict["post_rpocessing_iteration"] += (datetime.now() - start_post_processing_iteration).seconds
        true_total_runtime = (datetime.now() - true_total_start_time).seconds
    # Add complete runtime to object that stores all data about the best solution found by the matheuristic
    current_best_sol_object.total_runtime = round(total_runtime,2)
    runtime_dict["true_total"] = true_total_runtime
    runtime_dict["iterations"] = nb_iter
    return vrp_object, current_best_sol_object, runtime_dict