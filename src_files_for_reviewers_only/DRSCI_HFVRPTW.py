from datetime import datetime
import math
import os


from core.config import load_general_settings, load_hyperparameters_from_yaml_file, Hyperparameters
from core.solving.matheuristic import solve_with_matheuristic
from core.utils import set_file_directory
from core.vrp import VRP_OBJECT

def set_list_of_clusters_based_on_strategy(
    vrp_object: "VRP_OBJECT", hp_M: "Hyperparameters"
):
    if hp_M.max_clusters_setting == "limit_max_routes":
        max_no_clusters = int(
            # Size of instance divided by average number of stops in route (based on mean demand and capacity)
            vrp_object.stop_df.demand.sum()
            / (vrp_object.veh_type_df.capacity.min())
        )
    elif hp_M.max_clusters_setting == "limit_min_routes":
        # Sum of demand divided by the
        max_no_clusters = int(
            vrp_object.stop_df.demand.sum() / vrp_object.veh_type_df.capacity.max()
        )
    if hp_M.no_of_cluster_list_strategy == "pick_random":
        list_no_of_clusters = list(range(1, max_no_clusters + 1))
        return list_no_of_clusters
    elif hp_M.no_of_cluster_list_strategy == "ordered_2k":
        limit_2k = int(math.log2(vrp_object.dimension))
        list_no_of_clusters = [
            2**k for k in range(limit_2k + 1) if 2**k <= max_no_clusters
        ]
        return list_no_of_clusters


if __name__ == "__main__":
    project_name = "Matheuristic"
    on_server_repo_name = "matheuristic_hvrps_lagrange"
    
    gs = load_general_settings()
    hp_D = load_hyperparameters_from_yaml_file(project_name, "hp_D")
    hp_gurobi = load_hyperparameters_from_yaml_file(project_name, "hp_gurobi")
    hp_I = load_hyperparameters_from_yaml_file(project_name, "hp_I")
    hp_M = load_hyperparameters_from_yaml_file(project_name, "hp_M")
    hp_P = load_hyperparameters_from_yaml_file(project_name, "hp_P")
    hp_R = load_hyperparameters_from_yaml_file(project_name, "hp_R")
    hp_RP = load_hyperparameters_from_yaml_file(project_name, "hp_RP")
    hp_SCP = load_hyperparameters_from_yaml_file(project_name, "hp_SCP")
    
    max_rt_multilpier = 10

    dataset = "kerscher_et_al_hfvrptw"
    vrp_type = "HFVRPTW"
    instance_name = "c1_10_1+vt3+fcd-2+ccr-0.8+lva-0.9"
    file_name = f"{instance_name}.json"
    (
        local_path_to_instances,
        _,
        path_to_write_sol_json,
        path_to_write_sol_raw_csv,
        log_files,
    ) = set_file_directory(
        dataset=dataset,
        problem_type=vrp_type,
        output_dir="matheuristic_hvrps",
    )
    if gs.read_json:
        local_path_to_instances = os.path.join(local_path_to_instances, "instance_json")





    vrp_object = VRP_OBJECT(
        name=instance_name, type=vrp_type
    ).load_instance(
        os.path.join(local_path_to_instances, file_name),
        read_json=gs.read_json,
    )

    vrp_object.project_name = project_name


    hp_RP.subproblem_type_list = [
        "GSPI",
        "HVRP",
        "PCVRP",
        "VRP_indiv",
    ]
    hp_M.list_no_of_clusters = set_list_of_clusters_based_on_strategy(
        vrp_object=vrp_object, hp_M=hp_M
    )
    hp_M.maximum_runtime = int(vrp_object.dimension * max_rt_multilpier)
        
    vrp_object, vrp_object_sol, rt_dict = solve_with_matheuristic(
        vrp_object,
        gs=gs,
        hp_D_initial=hp_D,
        hp_gurobi=hp_gurobi,
        hp_I = hp_I,
        hp_M = hp_M,
        hp_P = hp_P,
        hp_R_initial=hp_R,
        hp_RP_initial=hp_RP,
        hp_SCP=hp_SCP
    )
    vrp_object_sol.sol_method = "DRSCI"

        
    file_name=f'{datetime.now().strftime("%Y%m%d%H%M%S")}__{instance_name}_result'
    # save the solution as a json file
    vrp_object_sol.write_to_json(
        path_to_write_sol_json,
        init_file_name=file_name,
        create_file_name=False
    )
