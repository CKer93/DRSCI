import numpy as np
from typing import TYPE_CHECKING

from core.solving.solve_vt_to_clusters_with_gurobi import veh_type_to_clusters

if TYPE_CHECKING:
    from pandas import DataFrame
    from core.vrp import VRP_OBJECT
    from core.config import Hyperparameters

def assign_vehicles_to_subproblems(vrp_object: "VRP_OBJECT", **kwargs):
    """"""
    # List of cluster ids:
    cluster_idx_list = list(vrp_object.stop_df.c_idx.unique())
    if "-1" in cluster_idx_list:
        cluster_idx_list.remove("-1")
    # Number of clusters
    num_clusters = len(cluster_idx_list)

    if "FSM" in vrp_object.type:
        stops_in_cluster_dict = vrp_object.stop_df.c_idx.value_counts().to_dict()
        veh_type_cluster_dict = {}
        for veh_type in vrp_object.veh_type_df.veh_type.to_list():
            veh_type_cluster_dict[veh_type] = {}
            for c_idx in cluster_idx_list:
                veh_type_cluster_dict[veh_type][c_idx] = stops_in_cluster_dict[c_idx]
        return vrp_object, veh_type_cluster_dict, False, 0

    assignment_rule = kwargs.get("assignment_rule", None)
    if assignment_rule == "balanced_remainder_random":
        return balanced_remainder_random(vrp_object, cluster_idx_list, num_clusters, kwargs["seed"])
    elif assignment_rule == "best_demand_capacity_fit":
        return best_demand_capacity_fit(vrp_object, cluster_idx_list, num_clusters,)
    elif assignment_rule== "based_on_solution_n_rb_clustering":
        return route_based_remainder_random(vrp_object, cluster_idx_list, num_clusters, kwargs["route_df"], kwargs["seed"])
    elif assignment_rule == "based_on_MIP_with_Gurobi":
        return based_on_MIP_with_Gurobi(
            vrp_object,
            kwargs["hp_gurobi"],
            kwargs["fthr"],
            kwargs.get("multiple_stop_tp_cluster_assignments_possible", False)
        )
    




def balanced_remainder_random(
    vrp_object: "VRP_OBJECT", cluster_idx_list: list, num_clusters: int, seed : int
) -> dict:
    """"""
    # Create a dictionary to store the allocation results
    veh_type_cluster_dict = {}
    # Loop through each vehicle type and distribute to clusters
    for _, row in vrp_object.veh_type_df.iterrows():
        # Calculate base allocation and remainder
        base_allocation = int(row["no_of_available"]) // num_clusters
        remainder = int(row["no_of_available"]) % num_clusters

        # Allocate base vehicles to each cluster
        base_distribution = [base_allocation] * num_clusters

        # Distribute the remainder randomly across the clusters
        np.random.seed(seed)
        remainder_indices = np.random.choice(num_clusters, remainder, replace=False)
        for idx in remainder_indices:
            base_distribution[idx] += 1

        veh_type_cluster_dict[row.veh_type] = dict(zip(cluster_idx_list, base_distribution))

    return vrp_object, veh_type_cluster_dict, False, 0

def best_demand_capacity_fit(vrp_object: "VRP_OBJECT", cluster_idx_list: list, num_clusters: int,):
    """"""
    # Dictionary that stores the vehicle type and an iterator as key and the veh_type and the capacity as values (nested dictionary)
    veh_type_dict = {}
    for _, row in vrp_object.veh_type_df.iterrows():
        for i in range(int(row["no_of_available"])):
            if row["veh_type"] == "1": # this veh_type is unlimitedly available in the SAP datasets
                continue
            key = f"vt{row['veh_type']}-{i}"  # Create a unique key
            veh_type_dict[key] = {"veh_type": row["veh_type"], "capacity": row["capacity"]}

    # Dictionary that stores the unserved demand of the clusters
    cluster_demand_dict = vrp_object.stop_df.groupby('c_idx')['demand'].sum().to_dict()
    if "-1" in cluster_demand_dict.keys():
        del cluster_demand_dict['-1']
    # Create a dictionary to store the allocation results
    veh_type_cluster_dict = {}
    for row_id, veh_type in enumerate(vrp_object.veh_type_df["veh_type"]):
        veh_type_cluster_dict[veh_type] = {}
        for c_idx in cluster_idx_list:
            if veh_type == "1":
                veh_type_cluster_dict[veh_type][c_idx] = int(vrp_object.dimension / num_clusters)
            else:
                veh_type_cluster_dict[veh_type][c_idx] = 0
            
    
    for veh_key, values in veh_type_dict.items():
        # Get cluster idx of the cluster that has the biggest uncovered demand remaining
        c_idx_max_demand = max(cluster_demand_dict, key=cluster_demand_dict.get)
        # Assign the vehicle to that cluster
        veh_type_cluster_dict[values["veh_type"]][c_idx_max_demand] += 1
        # Update the uncovered demand in the cluster_demand_dict
        cluster_demand_dict[c_idx_max_demand] = cluster_demand_dict[c_idx_max_demand] - values["capacity"]

    return vrp_object, veh_type_cluster_dict, False, 0

def route_based_remainder_random(vrp_object: "VRP_OBJECT", cluster_idx_list: list, num_clusters: int, route_df: "DataFrame", seed: int):
    # Dictionary that stores the vehicle type and an iterator as key and the veh_type and the capacity as values (nested dictionary)
    veh_type_cluster_dict = {}
    for row_id, veh_type in enumerate(vrp_object.veh_type_df["veh_type"]):
        veh_type_cluster_dict[veh_type] = {}
        for c_idx in cluster_idx_list:
            if veh_type == "1":
                veh_type_cluster_dict[veh_type][c_idx] = int(vrp_object.dimension / num_clusters)
            else:
                veh_type_cluster_dict[veh_type][c_idx] = 0

    for route_id, route in route_df.iterrows():
        if route.veh_type == '1':
            continue
        veh_type_cluster_dict[route.veh_type][str(route.c_idx)] += 1

    # assign remainer of limited fleet 
    for row_id, veh_type in vrp_object.veh_type_df.iterrows():
        if veh_type.veh_type == '1':
            continue
        no_of_unused_veh = veh_type.no_of_available - sum(veh_type_cluster_dict[veh_type.veh_type].values())
        # Calculate base allocation and remainder
        base_allocation = no_of_unused_veh // num_clusters
        remainder = no_of_unused_veh % num_clusters

        # Allocate base vehicles to each cluster
        if base_allocation > 0:
            for cluster in veh_type_cluster_dict[veh_type.veh_type].keys():
                veh_type_cluster_dict[veh_type.veh_type][cluster] += base_allocation 

        # Distribute the remainder randomly across the clusters
        if remainder > 0:
            np.random.seed(seed)
            remainder_indices = np.random.choice(num_clusters, remainder, replace=False)
            for idx in remainder_indices:
                veh_type_cluster_dict[veh_type.veh_type][str(idx)] += 1
    return vrp_object, veh_type_cluster_dict, False, 0

def based_on_MIP_with_Gurobi(
        vrp_object: "VRP_OBJECT",
        hp_gurobi: "Hyperparameters",
        fthr: float = 1.1,
        multiple_stop_tp_cluster_assignments_possible: bool = False
    ):
    # create a dictionary to split the vehicle fleet to the clusters
    fixed_stop_idx_to_c_idx_assignments_dict = {}
    for row_idx, row_values in vrp_object.stop_df[
        vrp_object.stop_df.u_c > fthr
    ].iterrows():
        if not row_values.c_idx == "-1":
            fixed_stop_idx_to_c_idx_assignments_dict[row_idx] = row_values.c_idx
    stop_cluster_dict, veh_type_cluster_dict, multiple_assignments, runtime = veh_type_to_clusters(
            vrp_object,
            fixed_stop_idx_to_c_idx_assignments_dict,
            hp_gurobi=hp_gurobi,
            multiple_stop_to_cluster_assignments_possible = multiple_stop_tp_cluster_assignments_possible
        )
    # Update the dataframe column 'c_idx' with values from the dictionary
    if not multiple_stop_tp_cluster_assignments_possible:
        vrp_object.stop_df.loc[stop_cluster_dict.keys(), "c_idx"] = (
            vrp_object.stop_df.index.map(stop_cluster_dict)
        )
    return vrp_object, veh_type_cluster_dict, multiple_assignments, runtime