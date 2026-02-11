from datetime import datetime
import gurobipy as gp
from gurobipy import GRB

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.vrp import VRP_OBJECT


def prepare_data_for_gurobi_model_vt_to_c(
        vrp_object,
):
    # Dictionary that stores the stop_idx has key and its demand as value
    stop_demand_dict = vrp_object.stop_df.demand.to_dict()
    # Remove depot from customer_demand_dict
    del stop_demand_dict[vrp_object.map_stop_id_idx[vrp_object.depot_identifier]]

    # Set of unique cluster labels but ignoring the 'dummy' cluster for the depot (label always '-1')
    cluster_idx_set = set(vrp_object.stop_df.c_idx.unique())
    cluster_idx_set.discard('-1')
    # Dictionary that stores the c_idx as key and its medoid as value
    cluster_medoid_dict = {}
    for c in cluster_idx_set:
        df_c = vrp_object.stop_df[vrp_object.stop_df.c_idx == c]
        cluster_medoid_dict[c] = df_c[df_c.u_c == 1].index[0]

    # Dictionary that stores the vehicle type and an iterator as key and the veh_type and the capacity as values (nested dictionary)
    veh_type_dict = {}
    for _, row in vrp_object.veh_type_df.iterrows():
        for i in range(row["no_of_available"]):
            key = f"vt{row['veh_type']}-{i}"  # Create a unique key
            veh_type_dict[key] = {"veh_type": row["veh_type"], "capacity": row["capacity"]}

    return stop_demand_dict, cluster_medoid_dict, veh_type_dict


def process_gurobi_result_data_vt_to_c(model, x, y, runtime, multiple_stop_to_cluster_assignments_possible):
    # Check if an optimal solution was found
    if model.Status == 3:
        raise ValueError(
            "The vehicle type allocation Gurobi model is infeasible. Check the input route_dict if it is sufficient."
        )
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        #if model.status == GRB.TIME_LIMIT:
            #print(f"\nGap to lower bound: {model.MIPGap * 100:.2f}%")
        # update customer to cluster assignments:
        multiple_assignments = False
        if multiple_stop_to_cluster_assignments_possible:
            stop_cluster_dict = {0 : ["-1"]}
        else:
            stop_cluster_dict = {0 : "-1"}
        for i_c in x:
            if x[i_c].X > 0.5:
                stop_idx = int(i_c.split("_")[0])
                cluster_idx = i_c.split("_")[1]
                if multiple_stop_to_cluster_assignments_possible:
                    if stop_cluster_dict.get(stop_idx, False):
                        stop_cluster_dict[stop_idx].append(cluster_idx)
                        multiple_assignments = True
                    else:
                        stop_cluster_dict[stop_idx] = [cluster_idx]                    
                else:
                    stop_cluster_dict[stop_idx] = cluster_idx

        veh_type_cluster_dict = {}
        for m_k_c in y:
            if y[m_k_c].X > 0.5:
                veh_type_id = m_k_c.split("-")[0].split('vt')[1]
                c_idx = m_k_c.split("_")[1]
                if veh_type_cluster_dict.get(veh_type_id, False):
                    if veh_type_cluster_dict[veh_type_id].get(c_idx, False):
                        veh_type_cluster_dict[veh_type_id][c_idx] += 1
                    else:
                        veh_type_cluster_dict[veh_type_id].update({c_idx : 1 })
                else:
                    veh_type_cluster_dict.update({veh_type_id : {c_idx : 1 }})
    else:
        raise ValueError("No feasible solution found in the Gurobi model to assign vehicles to subproblems.")
    return stop_cluster_dict, veh_type_cluster_dict, multiple_assignments, runtime

def veh_type_to_clusters(
        vrp_object: "VRP_OBJECT",
        fixed_stop_idx_to_c_idx_assignments_dict: dict,
        hp_gurobi: "Hyperparameters",
        multiple_stop_to_cluster_assignments_possible: bool = False
    ):
    stop_demand_dict, cluster_medoid_dict, veh_type_dict = prepare_data_for_gurobi_model_vt_to_c(
        vrp_object,
    )

    # Create the optimization model
    model = gp.Model("veh_types_to_clusters")

    # Decision variables: x_ic if stop i is assigned to cluster c
    x = {}
    for i in stop_demand_dict.keys():
        if i in fixed_stop_idx_to_c_idx_assignments_dict.keys():
            for c in cluster_medoid_dict.keys():
                if fixed_stop_idx_to_c_idx_assignments_dict[i] == c:
                    x[f"{i}_{c}"] = model.addVar(
                        lb=1,
                        ub=1,
                        vtype=GRB.BINARY,
                        name=f"{i}_{c}"
                    )
                else:
                    x[f"{i}_{c}"] = model.addVar(
                        lb=0,
                        ub=0,
                        vtype=GRB.BINARY,
                        name=f"{i}_{c}"
                    )
        else:
            for c in cluster_medoid_dict.keys():
                x[f"{i}_{c}"] = model.addVar(
                    lb=0,
                    ub=1,
                    vtype=GRB.BINARY,
                    name=f"{i}_{c}"
                )

    y = {}
    for k in veh_type_dict.keys():
        for c in cluster_medoid_dict.keys():
            y[f"{k}_{c}"] = model.addVar(
                lb=0,
                ub=1,
                vtype=GRB.BINARY,
                name=f"{k}_{c}"
            )

    # Constraints
    # Every custmer is assigned to at least one cluster
    if multiple_stop_to_cluster_assignments_possible:
        for i in stop_demand_dict.keys():        
            model.addConstr(
                gp.quicksum(
                    x[f"{i}_{c}"] for c in cluster_medoid_dict.keys()
                )
                >= 1,
                name=f"Assign_{i}_to_c",
            )
    # Every custmer is assigned to exactly one cluster
    else:
        for i in stop_demand_dict.keys():        
            model.addConstr(
                gp.quicksum(
                    x[f"{i}_{c}"] for c in cluster_medoid_dict.keys()
                )
                == 1,
                name=f"Assign_{i}_to_c",
            )


    # A vehicle is at maximum assigned to one cluster
    for k in veh_type_dict.keys():
        model.addConstr(
            gp.quicksum(
                y[f"{k}_{c}"] for c in cluster_medoid_dict.keys()
            )
            <= 1,
            name=f"Asign_{k}_to_c",
        )
    
    # logic constraint vehicle capacity fulfills customer demands assigned to cluster
    for c in cluster_medoid_dict.keys():
        model.addConstr(
            gp.quicksum(
                x[f"{i}_{c}"] * stop_demand_dict[i] for i in stop_demand_dict.keys() 
            ) -
            gp.quicksum(
                y[f"{k}_{c}"] * veh_type_dict[k]['capacity'] for k in veh_type_dict.keys() 
            )
            <= 0,
            name=f"Demand_capa_feas_{c}"
            )
        
    # Objective function: Minimize dissimilarity between stops in cluster and cluster-medoid
    model.setObjective(
        gp.quicksum(
                x[f"{i}_{c}"] * vrp_object.similarity_matrix[i-1][cluster_medoid_dict[c]-1]
                for i in stop_demand_dict.keys() for c in cluster_medoid_dict.keys()
        ),
        GRB.MINIMIZE,
    )

    model.update()
    # model parameters
    model.params.OutputFlag = 0
    if hp_gurobi.verbose:
        model.params.OutputFlag = 1
    model.params.Threads = hp_gurobi.num_threads
    # set time limit
    model.params.TimeLimit = hp_gurobi.maximum_runtime
    start_time = datetime.now()
    model.optimize()
    runtime = (datetime.now() - start_time).seconds
    return process_gurobi_result_data_vt_to_c(model, x, y, runtime, multiple_stop_to_cluster_assignments_possible)

