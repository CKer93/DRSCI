import ast
from copy import deepcopy
from datetime import datetime
import gurobipy as gp
from gurobipy import Model, GRB
import logging
from typing import TYPE_CHECKING

from core.evaluation import calculate_variable_costs

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.vrp import VRP_OBJECT

logger = logging.getLogger(__name__)

def build_GSP_model_gurobi(
    fleet_dict: dict,
    giant_tours_idx_excl_depot: list,
    gt_idx_excl_depot: list,
    subroute_dict: dict,
) -> tuple[object, dict]:
    # Initialize model
    model = Model("giant_tour_partitioning")

    # Generate set of feasible x_u,v,k variables
    feas_subroute_veh_pairs = []
    for subtour_tuple, subtour_values in subroute_dict.items():
        demand = subtour_values['demand']
        for k_id, k_values in fleet_dict.items():
            if demand <= k_values['capacity']:
                feas_subroute_veh_pairs.append((subtour_tuple, k_id))
    x = model.addVars(
        feas_subroute_veh_pairs,
        vtype=GRB.BINARY,
        name="x"
    )
    # Objective function: Minimize total cost of used subtours and the assigned vehicles
    model.setObjective(
        gp.quicksum(
            (
                fleet_dict[k]["fixed_costs"]
                + fleet_dict[k]["variable_costs"] * subroute_dict[u_v]["var_costs"]
            )
            * x[u_v, k]
            for (u_v, k) in x
        ),
        GRB.MINIMIZE
    )
    
    # Constraint: Exactly one segment that start at the first customer of the giant tour must be selected
    model.addConstrs(
        gp.quicksum(
            x[(1, v), k] 
            for k in fleet_dict.keys()
            for v in giant_tour_idx_excl_depot
            if ((1, v), k) in x.keys()
        ) == 1
        for giant_tour_idx_excl_depot in map(tuple, giant_tours_idx_excl_depot)
        if 1 in giant_tour_idx_excl_depot
    )


    # Constraint: Exactly one segment that ends at the last customer of the giant tour must be selected
    n =  gt_idx_excl_depot[-1]
    model.addConstrs(
        gp.quicksum(
            x[(u, n), k]
            for k in fleet_dict
            for u in giant_tour_idx_excl_depot
            if ((u, n), k) in x
        )
        == 1
        for giant_tour_idx_excl_depot in map(tuple, giant_tours_idx_excl_depot)
        if n in giant_tour_idx_excl_depot
    )

    # Constraint: Seamless partitioning, the next route starts at the next customer where the last tour ended and returned to the depot
    model.addConstrs(
        gp.quicksum(
            x[(u, v), k] 
            for k in fleet_dict
            for u in gt_idx_excl_depot
            if ((u, v), k) in x
        )
        ==
        gp.quicksum(
            x[(v+1, u), k] 
            for k in fleet_dict
            for u in gt_idx_excl_depot[v:]
            if ((v+1, u), k) in x
        )
        for v in gt_idx_excl_depot[:-1]
    )
    # Constraint: Use at maximum every vehicle once
    model.addConstrs(
        gp.quicksum(
        x[(u_v), k] 
        for u_v in subroute_dict.keys() 
        if ((u_v), k) in x
        ) 
        <= 1
        for k in fleet_dict.keys()
    )
    model.update()
    #print(f"Runtime update gurobimodel: {(datetime.now()-start_time).seconds}")
    return model, x

def generate_fleet_dict(
    vrp_object: "VRP_OBJECT"
):
    # Create dictionary that stores the vehicle type and an iterator as key and the veh_type, the capacity, and var_unit_costs as values (nested dictionary)
    fleet_dict = {}
    if "FSM" in vrp_object.type or "HFVRPTW" == vrp_object.type:
        total_customer_demand = vrp_object.stop_df.demand.sum()
    for _, row in vrp_object.veh_type_df.iterrows():
        if "FSM" in vrp_object.type:
            if "TW" in vrp_object.type:
                no_of_available_veh = int((total_customer_demand * 2) / row["capacity"])
            else:
                no_of_available_veh = int((total_customer_demand * 1.2) / row["capacity"])
        else:
            if "HFVRPTW" == vrp_object.type and row["no_of_available"] == vrp_object.dimension:
                no_of_available_veh = int((total_customer_demand * 1) / row["capacity"])
            else:
                no_of_available_veh = row["no_of_available"]
        for i in range(no_of_available_veh):
            key = f"vt{row['veh_type']}-{i}"  # Create a unique key
            fleet_dict[key] = {
                "veh_type": row["veh_type"],
                "capacity": row["capacity"],
                "fixed_costs": row['costs'],
                "variable_costs": row['variable_distance_unit_costs']
            }
    return fleet_dict

def generate_giant_tours(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_R: "Hyperparameters"
    ) -> tuple[dict, float, dict]:
    hp_R_gt = deepcopy(hp_R)
    hp_R_gt.max_runtime = int(vrp_object.dimension * hp_R.max_runtime_dimension_multiplier)
    hp_R_gt.nb_iter_without_improvement = 20000
    # Create copy of original vrp_object and modify fleet to ONE uncapacitated vehicle
    vrp_object_giant_tour = deepcopy(vrp_object)
    vrp_object_giant_tour.veh_type_df = vrp_object_giant_tour.veh_type_df[:1]
    vrp_object_giant_tour.veh_type_df.capacity = vrp_object.stop_df.demand.sum() * 2
    vrp_object_giant_tour.veh_type_df.no_of_available = vrp_object.dimension

    # Generate single TSP tour using PyVRP
    route_dict, runtime, solver_meta_data = vrp_object_giant_tour.solve(gs, hp_R_gt)
    giant_tours = []
    giant_tours_idx_excl_depot = []
    gt_idx_excl_depot = []

    if len(route_dict) > 0:
        # Concat routes (including depot)
        gt_id = 0
        for dict_values in route_dict.values():
            giant_tours.append(dict_values['route'])
            if gt_id == 0:
                start_idx = 1
            else:
                start_idx = giant_tours_idx_excl_depot[gt_id-1][-1]+1
            giant_tours_idx_excl_depot.append(list(range(start_idx, start_idx+len(giant_tours[gt_id])-2)))
            gt_idx_excl_depot += giant_tours_idx_excl_depot[gt_id]

            gt_id += 1        

    # Return the route_dict, i.e., single entry that is the giant tour, and the solver runtime
    return giant_tours, giant_tours_idx_excl_depot, gt_idx_excl_depot, runtime, route_dict


def generate_subtour_dict(
    giant_tours: list,
    giant_tours_idx_excl_depot: list,
    vrp_object: "VRP_OBJECT",
    exclude_proven_infeasible: bool = True,  
    exclude_low_utilized: bool = True,    
): 
    if exclude_low_utilized:
        min_demand_routes = vrp_object.veh_type_df.capacity.min()*0.5
    if exclude_proven_infeasible:
        max_capacity = vrp_object.veh_type_df.capacity.max()
    # Create numpy array for demands for fast array-based lookups
    demands_array = vrp_object.stop_df["demand"].to_numpy()  
    # Generate subtour dictionary with stop ids, variable costs (veh type independent), and demand
    subroute_dict = {}
    for gt_id, giant_tour_idx_excl_depot in enumerate(giant_tours_idx_excl_depot):
        len_giant_tour_idx_excl_depot = len(giant_tour_idx_excl_depot)
        # -----------------------------------------------------
        # 1) Pre-map giant_tour_idx_excl_depot to stop_ids once
        # -----------------------------------------------------
        giant_tour_stop_idx = [vrp_object.map_stop_id_idx[s] for s in giant_tours[gt_id][1:-1]]
        # -----------------------------------------------------
        # 2) Build a prefix sum of demands for this route
        #    so demand of sub-route [u_id ... v_id] is:
        #    prefix_dem[v_id+1] - prefix_dem[u_id]
        # -----------------------------------------------------
        prefix_dem = [0] * (len(giant_tour_stop_idx) + 1)
        for i in range(len(giant_tour_stop_idx)):
            prefix_dem[i + 1] = prefix_dem[i] + demands_array[giant_tour_stop_idx[i]]

        # -----------------------------------------------------
        # 3) Nested loop to consider all (u_id, v_id) sub-slices
        # -----------------------------------------------------
        for u_id in range(len_giant_tour_idx_excl_depot):
            for v_id in range(u_id, len_giant_tour_idx_excl_depot):
                # Reconstruct the subroute in terms of actual stop IDs
                subroute = [vrp_object.depot_identifier] + giant_tours[gt_id][u_id+1:v_id+2] + [vrp_object.depot_identifier]
                # Convert those to integer indices for cost calculation
                subroute_idx = [vrp_object.map_stop_id_idx[i] for i in subroute]
                # sub-route is from u_id to v_id in the *excl_depot* route
                demand = prefix_dem[v_id + 1] - prefix_dem[u_id]
                if exclude_proven_infeasible and demand > max_capacity:
                    continue
                if exclude_low_utilized and demand < min_demand_routes:
                    continue
                var_costs = calculate_variable_costs(
                    vrp_object.distance_matrix, 
                    subroute_idx,
                    veh_type_unit_costs=1.0,
                )
                # The dictionary key is (u, v), where u,v are the original stop IDs
                u_stop_id = giant_tour_idx_excl_depot[u_id]
                v_stop_id = giant_tour_idx_excl_depot[v_id]
                # First and last customer of subroute are the unique key in the dictionary
                subroute_dict[(u_stop_id, v_stop_id)] = {
                    'subroute': subroute,
                    'var_costs': var_costs,
                    'demand': demand
                } 
    return subroute_dict


def giant_tours_n_partitioning_MILP(
        vrp_object: "VRP_OBJECT",
        gs: "Hyperparameters",
        hp_R: "Hyperparameters",
        hp_gurobi: "Hyperparameters",
        hp_I: "Hyperparameters"
):
    # Generate giant tour
    giant_tours, giant_tours_idx_excl_depot, gt_idx_excl_depot, rt_gt, giant_tour_infeas_dict = generate_giant_tours(
        vrp_object,
        gs,
        hp_R,
    )
    # Return 'empty' sol_dict if no giant tour could be found (must be at least a pendulum tour between the depot and a single customer)
    if len(giant_tours) < 1:
        return {"Infeas_GS": {"route_dict": {}, "runtime": rt_gt, "route_dict_matches_df_format": False}}
    # Create subroute tuples based on start and end customer
    subroute_dict = generate_subtour_dict(
        giant_tours,
        giant_tours_idx_excl_depot,
        vrp_object,
        # exclude_proven_infeasible=False
    )
    # Create fleet dict with every available vehicle and its meta-data
    fleet_dict = generate_fleet_dict(
        vrp_object
    )
    # Build Gurobi model
    model, x = build_GSP_model_gurobi(
        fleet_dict,
        giant_tours_idx_excl_depot,
        gt_idx_excl_depot,
        subroute_dict
    )
    # Set parameters to prioritize finding a feasible solution quickly
    model.params.OutputFlag = 0
    if hp_gurobi.verbose:
        model.params.OutputFlag = 1
    model.params.Threads = hp_gurobi.num_threads
    model.params.TimeLimit = hp_gurobi.maximum_runtime

    # Optimize model
    start_time = datetime.now()
    model.optimize()
    rt_gurobi = (datetime.now() - start_time).seconds
    if model.Status == GRB.INFEASIBLE:
        return {
            "GS": {"route_dict": giant_tour_infeas_dict, "runtime": rt_gt, "route_dict_matches_df_format": False},
            "Infeas_GSP": {"route_dict": {}, "runtime": rt_gt+rt_gurobi, "route_dict_matches_df_format": False}
            } 


    route_dict = {}
    subroute_veh_tuples_used = [var for var in model.getVars() if var.X > 0.5]
    for subroute_veh_tuple_used in subroute_veh_tuples_used:
        subroute_tuple = ast.literal_eval(subroute_veh_tuple_used.VarName[2:-1].split(",vt")[0])
        veh_id = subroute_veh_tuple_used.VarName[2:-1].split(",")[-1]

        veh_type = fleet_dict[veh_id]['veh_type']
        route_id = vrp_object.name + "+" + veh_id
        route = subroute_dict[subroute_tuple]['subroute']
        # extract tour segment and create route_dict.
        route_dict[route_id] = {
            'veh_id': veh_id,
            'veh_type': veh_type,
            'route_id': route_id,
            'route': route
        }
    route_dict_improved, rt_I = vrp_object.improve(
        route_dict,
        hp_I,
        [],
        gs.seed,
        gs.dec_prec
    )

    sol_dict = {
        "GS": {"route_dict": giant_tour_infeas_dict, "runtime": rt_gt, "route_dict_matches_df_format": False},
        "GSP": {"route_dict": route_dict, "runtime": rt_gurobi, "route_dict_matches_df_format": False},
        "GSPI": {"route_dict": route_dict_improved, "runtime": rt_I, "route_dict_matches_df_format": False},
    }
    return sol_dict