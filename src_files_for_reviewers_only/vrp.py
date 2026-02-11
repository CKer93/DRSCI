import copy
import logging
import numpy as np
import os
import pandas as pd
import re
import time
from typing import TYPE_CHECKING

from core.clustering.clustering import vertex_based_clustering, route_based_clustering
from core.data_manipulation.data_processing import (
    data_cleansing_vrplib_bks_solution,
    filter_df_based_on_row_idxs,
    filter_dm_based_on_row_idxs,
    read_instance_vrplib_format,
    read_solution_vrplib_format,
)
from core.evaluation import create_sol_obj
from core.improving.ls_with_pyvrp import (
    run_pyvrp_local_search,
    transform_route_dict_to_list_of_pyvrp_routes,
)
from core.solving.solve_with_pyvrp import (
    solve_with_pyvrp_hgs,
    prepare_data_for_pyvrp,
    format_pyvrp_output,
)
from core.solving.solve_bin_packing_with_gurobi import (
    solve_bin_packing_model_with_gurobi,
    prepare_data_for_gurobi_model_bin_packing,
    process_gurobi_result_data_bin_packing,
)
from core.utils import (
    util_write_object_to_json,
    util_read_object_from_json,
    util_create_subproblem_df,
)
import core.solving.root_pool_strategies as rps

if TYPE_CHECKING:
    from core.config import Hyperparameters

logger = logging.getLogger(__name__)

class VRP_OBJECT:
    """
    Represents a Vehicle Routing Problem (VRP) instance.

    This class provides methods to initialize and load data for a VRP instance,
    supporting different problem types such as CVRP, VRPTW, and HVRPTW.
    """

    def __init__(self, name : str = "", type: str = "") -> "VRP_OBJECT":
        """
        Initializes the VRP_OBJECT instance with a name and type.

        Args:
            name (str): The name of the VRP instance.
            type (str): The type of problem (e.g., "HVRPTW", "VRPTW").

        Returns:
            VRP_OBJECT: An initialized VRP_OBJECT instance.

        Note:
            No information is needed, when the vrp object is loaded from a .json file.
        """
        self.name = name
        self.type = type

    def create_sub_vrp(
        self,
        list_of_idx_to_keep: list,
        sup_prob_name_add_on: str,
        update_veh_availablity: bool = True,
    ) -> "VRP_OBJECT":
        """
        Creates a sub-VRP object based on a specified cluster.

        This method generates a sub-VRP object by extracting a subset of stops that belong
        to the specified cluster. The new VRP object will include only the stops in the
        specified cluster and will update relevant attributes accordingly.

        Args:
            cluster_name (str): The name of the cluster to create the sub-VRP from.

        Returns:
            VRP_OBJECT: A new VRP object containing only the stops from the specified cluster.

        Example:
            >>> vrp = VRP_OBJECT()
            >>> sub_vrp = vrp.create_sub_vrp("cluster_1")
            >>> print(sub_vrp.name)

        Notes:
            - The method deep copies the original VRP object and then removes unnecessary attributes.
            - The sub-VRP object will have its stop DataFrame and distance matrix filtered
            based on the specified cluster.
            - The depot is included in the sub-VRP object and placed at the first index.

        """
        sub_vrp_object = copy.deepcopy(self)

        # Delete unnecessary attributes for the sub-VRP bks_route_dict and similarity_matrix if they exist
        if hasattr(sub_vrp_object, "bks_sol"):
            del sub_vrp_object.bks_sol
        if hasattr(sub_vrp_object, "similarity_matrix"):
            del sub_vrp_object.similarity_matrix

        # Update the name of the sub-VRP object to reflect the cluster
        if "+c" in sup_prob_name_add_on:
            sub_vrp_object.name = re.sub(r"(\+C[^+]*)", r"\1" + sup_prob_name_add_on, self.name)
            # sub_vrp_object.name = self.name.split("+C")[0] + "+C" + self.name.split("+C")[1].split("+")[0] + sup_prob_name_add_on + self.name.split("+C")[1][len(self.name.split("+C")[1].split("+")[0]):]

        # Check if the depot is in the list_of_idx_to_keep
        depot_idx = sub_vrp_object.map_stop_id_idx[sub_vrp_object.depot_identifier]
        if depot_idx not in list_of_idx_to_keep:
            # Add it if its missing:
            list_of_idx_to_keep.insert(0, depot_idx)
        # Filter the stop DataFrame based on the cluster name
        sub_vrp_object.stop_df = filter_df_based_on_row_idxs(
            self.stop_df, list_of_idx_to_keep
        )

        # Update the dimension attribute of the sub-VRP object (excl. the depot)
        sub_vrp_object.dimension = len(sub_vrp_object.stop_df.stop_id)-1

        # reset index of shrinked stop_df to filter the correct distance matrix values
        sub_vrp_object.stop_df = sub_vrp_object.stop_df.reset_index(drop=True)
        sub_vrp_object.distance_matrix = filter_dm_based_on_row_idxs(
            self.distance_matrix, list_of_idx_to_keep
        )

        # Update the no_of_available vehicles based on the demand share of the sub_vrp_object relative to the original vrp_object

        if update_veh_availablity:
            # check if it is an homogeneous fleet
            if sub_vrp_object.veh_type_df.shape[0] == 1:
                # This is the DRI paper logic: The vehicles are assigned to sub-clusters based on the demand to ensure feasibility in terms of fleet size.
                sub_vrp_object.veh_type_df.no_of_available = max(
                    1,
                    int(
                        np.floor(
                            self.veh_type_df.no_of_available
                            * sub_vrp_object.stop_df.demand.sum()
                            / self.stop_df.demand.sum()
                        )
                    )
                )
            else:
                cluster_idx = sup_prob_name_add_on.split("+c")[1]

                # Iterate over the rows of the dataframe using index for direct access
                for row_id in sub_vrp_object.veh_type_df.index:
                    row_values = sub_vrp_object.veh_type_df.loc[row_id]
                    if self.veh_type_cluster_dict[row_values["veh_type"]].get(
                        cluster_idx, False
                    ):
                        # Update the dataframe directly using .loc
                        sub_vrp_object.veh_type_df.loc[row_id, "no_of_available"] = (
                            self.veh_type_cluster_dict[row_values["veh_type"]][cluster_idx]
                        )
                    else:
                        # Delete the row if no vehicle of that type was assigned to that subcluster
                        sub_vrp_object.veh_type_df.drop(row_id, inplace=True)

        # Update the mapping of indices to stop IDs and vice versa
        sub_vrp_object.map_idx_stop_id = dict(
            zip(sub_vrp_object.stop_df.index, sub_vrp_object.stop_df.stop_id)
        )
        sub_vrp_object.map_stop_id_idx = dict(
            zip(sub_vrp_object.stop_df.stop_id, sub_vrp_object.stop_df.index)
        )

        return sub_vrp_object

    def create_route_pool(
            self,
            gs: "Hyperparameters",
            hp_gurobi: "Hyperparameters",
            hp_I: "Hyperparameters",
            hp_R: "Hyperparameters",
            hp_RP: "Hyperparameters",
        ):
        """"""
        if len(self.veh_type_cluster_dict) > 1:
            # initialize variables and deep copy the instance attributes for resetting them after the decomposition
            vrp_object = copy.deepcopy(self)
            route_dict_long = {}
            runtime_R_t = 0
            for cluster_name in vrp_object.stop_df.c_idx.unique():
                if cluster_name == "-1":
                    continue
                # Create a list of indices to keep based on the cluster name in the cluster column
                list_of_idx_to_keep = vrp_object.stop_df[
                    vrp_object.stop_df["c_idx"] == cluster_name
                ].index.tolist()
                vrp_object_sub = vrp_object.create_sub_vrp(
                    list_of_idx_to_keep=list_of_idx_to_keep,
                    sup_prob_name_add_on=f"+c{cluster_name}",
                    update_veh_availablity=True,
                )
                
                if vrp_object_sub.veh_type_df.shape[0] == 0:
                    logger.info(f'no data in veh_type_df for {vrp_object_sub.name}')
                
                route_dict_long_sub, runtime_R_t_sub = rps.create_routes(
                    vrp_object=vrp_object_sub,
                    gs=gs,
                    hp_gurobi=hp_gurobi,
                    hp_I=hp_I,
                    hp_R=hp_R,
                    hp_RP = hp_RP
                )
                route_dict_long.update(route_dict_long_sub)
                runtime_R_t += runtime_R_t_sub

        else:
            vrp_object = copy.deepcopy(self)
            runtime_available_sub = int(
                hp_R.max_runtime / vrp_object.veh_type_df.shape[0]
            )
            route_dict_long_sub, runtime_R_t_sub = rps.create_routes(
                vrp_object_sub, runtime_available_sub
            )
            route_dict_long.update(route_dict_long_sub)
            runtime_R_t += runtime_R_t_sub

        return route_dict_long, runtime_R_t

    def decompose(
        self,
        feature_array: np.ndarray,
        hp_D: "Hyperparameters",
        seed: int,
        dec_prec: int = 4,
        **kwargs
    ) -> tuple[pd.DataFrame, float]:
        """
        Clusters stops in a VRP instance based on a specified metric.

        This function clusters the stops of a Vehicle Routing Problem (VRP) instance
        using a specified clustering method and metric. It supports various similarity
        metrics and clustering methods, and returns the VRP object with updated stop
        cluster indices.

        Notes:
            - The function removes the depot from the clustering process by excluding the
            first stop in the feature array.
            - The best clustering configuration is selected based on the specified clustering
            metric.

        """
        start_time = time.time()
        if hp_D.cluster_strategy == 'vbc':
            labels, u_max = vertex_based_clustering(
                feature_array=feature_array,                    
                similarity_matrix=self.similarity_matrix,
                hp_D=hp_D,
                seed=seed
            )
        elif hp_D.cluster_strategy == 'rbc':
            feature_array = np.stack(kwargs["route_df"].cog_feature_array)
            labels, u_max = route_based_clustering(
                route_df=kwargs["route_df"],
                stop_df=self.stop_df,
                feature_array=feature_array,
                hp_D=hp_D,
                seed=seed
            )
            
        runtime = round(time.time() - start_time, dec_prec)
        # Insert -1 for the depot to labels and u_max and add them to the vrp object stop_df
        self.stop_df["c_idx"] = np.insert(labels, 0, -1).astype(str)
        self.stop_df["u_c"] = np.insert(u_max, 0, -1).astype(float)

        # Create a subproblem_df with the cluster information
        subproblem_df = util_create_subproblem_df(self.stop_df, dec_prec)

        return subproblem_df, runtime

    def improve(
        self,
        route_dict: dict,
        hp_I: "Hyperparameters",
        pruned_list_stop_neigh: list,
        seed: int,
        dec_prec: int = 4
    ) -> tuple[dict, float]:
        pyvrp_model = prepare_data_for_pyvrp(self)
        pyvrp_model_data = pyvrp_model.data()
        # add route data of dr solution including veh_type to correctly calculate the incumbent costs
        routes_data = transform_route_dict_to_list_of_pyvrp_routes(
            self.map_stop_id_idx, pyvrp_model_data, route_dict
        )

        pyvrp_sol, runtime_I, sol_has_improved = run_pyvrp_local_search(
            pyvrp_model_data,
            routes_data,
            hp_I,
            pruned_list_stop_neigh,
            seed,
            dec_prec
        )
        if sol_has_improved:
            # transform pyvrp_sol_obj to route_dict
            return format_pyvrp_output(self, pyvrp_model, pyvrp_sol, format_after_improvement=True), runtime_I
        else:
            return route_dict, runtime_I

    def load_instance(
        self,
        path_to_data: str,
        read_json: bool = False,
    ) -> "VRP_OBJECT":
        """
        Loads the instance data from a file.

        Args:
            path_to_data (str): Path to the instance file.
            read_json (bool): If True, the instance data is read from a JSON file. Defaults to False.

        Returns:
            VRP_OBJECT: The instance object with the data loaded from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the data cannot be loaded properly due to format issues.

        Examples:
            >>> vrp = VRP_OBJECT("example_instance")
            >>> vrp.load_instance("/path/to/data")

        Notes:
            If `read_json` is set to True, the method expects a JSON file format.
            Otherwise, it loads data in the Solomon/Gehring Homberger benchmark format
            for VRPTW problem types.
        """
        if read_json:
            return util_read_object_from_json(self, path_to_data)
        if self.type == "VRP":
            # Load the instance data from a file in the Solomon/ Gehring Homberger benchmark format.
            instance_data_dict = read_instance_vrplib_format(
                os.path.join(path_to_data, "original", self.name + ".vrp")
            )
            self.stop_df = pd.DataFrame(
                {
                    "x_coord": instance_data_dict["node_coord"][:, 0],
                    "y_coord": instance_data_dict["node_coord"][:, 1],
                    "demand": instance_data_dict["demand"],
                }
            )
            self.name = instance_data_dict["name"]
            self.distance_matrix = instance_data_dict["edge_weight"]
        elif self.type == "VRPTW":
            # Load the instance data from a file in the Solomon/ Gehring Homberger benchmark format.
            instance_data_dict = read_instance_vrplib_format(
                os.path.join(path_to_data, "original", self.name + ".TXT")
            )
            self.spatial_char = self.name.split("_")[0][:-1]
            self.veh_char = self.name.split("_")[0][-1:]
            self.temp_char = self.name.split("_")[-1]
            self.stop_df = pd.DataFrame(
                {
                    "x_coord": instance_data_dict["node_coord"][:, 0],
                    "y_coord": instance_data_dict["node_coord"][:, 1],
                    "demand": instance_data_dict["demand"],
                    "tw_start": instance_data_dict["time_window"][:, 0],
                    "tw_end": instance_data_dict["time_window"][:, 1],
                    "service_time": instance_data_dict["service_time"],
                }
            )
        else:
            raise ValueError(
                f"The problem type {self.type} is not yet supported for loading data."
            )
        instance_solution_dict = read_solution_vrplib_format(
            os.path.join(path_to_data, "solution", self.name + ".sol")
        )
        self.name = instance_data_dict["name"]
        self.distance_matrix = instance_data_dict["edge_weight"]

        # Add the stop ID based on the index
        self.stop_df["stop_id"] = self.stop_df.index.astype(str)
        # create a dicts that map the stop_id to the index of the stop_df
        self.map_idx_stop_id = dict(zip(self.stop_df.index, self.stop_df.stop_id))
        self.map_stop_id_idx = dict(zip(self.stop_df.stop_id, self.stop_df.index))

        self.depot_identifier = self.stop_df[self.stop_df["demand"] == 0][
            "stop_id"
        ].values[0]
        self.dimension = len(self.stop_df) - 1  # excluding the depot from the dimension
        self.bks_route_dict = data_cleansing_vrplib_bks_solution(
            instance_solution_dict["routes"], self.depot_identifier
        )
        self.veh_base_capacity = instance_data_dict["capacity"]
        # Create vehicle_type_df
        self.veh_type_df = pd.DataFrame(
            {
                "veh_type": "1",
                "capacity": self.veh_base_capacity,
                "costs": 0,
                "no_of_available": instance_data_dict["vehicles"],
                # calculate the share on the complete fleet based no_of_available
                "fleet_share": 1.0,
            },
            index=[0],
        )
        return self

    def route(
        self,
        gs: "Hyperparameters",
        hp_R: "Hyperparameters"
    ):
        # split runtime based on the number of stops per cluster in the instances
        share_of_customers_in_clusters_dict = dict(
            self.stop_df.c_idx.value_counts(normalize=True)
        )
        total_runtime_available = copy.deepcopy(hp_R.max_runtime)
        total_runtime = 0
        total_route_dict = {}
        list_of_raw_result_objs = []
        for cluster_name in self.stop_df.c_idx.unique():
            if cluster_name == "-1":
                continue
            # Create a list of indices to keep based on the cluster name in the cluster column
            list_of_idx_to_keep = self.stop_df[
                self.stop_df["c_idx"] == cluster_name
            ].index.tolist()
            instance_sub = self.create_sub_vrp(
                list_of_idx_to_keep=list_of_idx_to_keep,
                sup_prob_name_add_on=f"+c{cluster_name}",
                update_veh_availablity = True
            )
            # update runtime based on the share of customers in the cluster and the total runtime available (round down to the nearest integer)
            hp_R.max_runtime = int(
                total_runtime_available
                * share_of_customers_in_clusters_dict[cluster_name]
            )
            route_dict, runtime, raw_result_obj = instance_sub.solve(gs, hp_R)
            # stagg route dict and runtime
            total_route_dict.update(route_dict)
            total_runtime += runtime
            list_of_raw_result_objs.append(raw_result_obj)
        # reset runtime to the total runtime available after the split and all suboroblems have been routed
        hp_R.max_runtime = total_runtime_available
        return total_route_dict, total_runtime, list_of_raw_result_objs

    def resolve_capacity_infeasibility(self):
        """"""
        # get indices of self.stop_df where the value in the demand column is smaller than the capacity of the vehicle type in self.veh_type_df.capacity
        list_of_idx_to_keep = self.stop_df[
            self.stop_df.demand <= self.veh_type_df.capacity.max()
        ].index.tolist()
        # remove idx of the depot based on the depot_identifier and map_idx_stop_id
        list_of_idx_to_keep.remove(self.map_stop_id_idx[self.depot_identifier])
        # create a subproblem based on the indices of the customers with demand smaller than the capacity of the vehicle type
        self = self.create_sub_vrp(
            list_of_idx_to_keep = list_of_idx_to_keep,
            sup_prob_name_add_on="",
            update_veh_availablity=False
        )
        return self

    def setup_heterogeneous_fleet(
        self,
        no_of_vehicle_types: int,
        sum_avail_large_veh: float = 0.9,
        avail_ratio=None,
        xs_fleet_size: str = "pendulum",
        veh_avail_buffer: float = 0.1,
        capa_interval_str="double",
        capa_cost_ratio: float = 0.8,
        fleet_scenario: str = "f_v_even",
        fix_var_multiplier=2.0,
    ) -> "VRP_OBJECT":

        variable_costs = self.bks_sol.var_costs
        # Create vehicle_type_df
        veh_types = [str(x) for x in range(1, no_of_vehicle_types + 1)]
        # ----------------------------------------------
        # Calculate vehicle capacities
        if capa_interval_str != "double":
            Warning(
                "The capa_interval_str must be 'double' for now; setting it to 'double'"
            )
        if capa_interval_str == "double":
            capa_interval = (
                np.ceil(self.veh_base_capacity * 0.5),
                self.veh_base_capacity,
            )
        veh_capacities = []
        for i in range(no_of_vehicle_types):
            veh_capacities.append(
                np.ceil(
                    capa_interval[0]
                    + i
                    * (capa_interval[1] - capa_interval[0])
                    / (no_of_vehicle_types - 1)
                )
            )
        # ----------------------------------------------
        if xs_fleet_size not in ["pend", "bks_homo", "min_buffer"]:
            raise ValueError(
                f"The xs_fleet_size must be one of ['pend', 'bks_homo', 'min_buffer']; not {xs_fleet_size}"
            )
        if xs_fleet_size == "pend":
            veh_availabilities = [len(self.stop_df[1:])]
        elif xs_fleet_size == "bks_homo":
            veh_availabilities = [len(self.bks_sol.route_dict)]
        elif xs_fleet_size == "min_buffer":
            veh_availabilities = [
                max(
                    1,
                    np.ceil(
                        (
                            self.stop_df.demand.sum()
                            * (1 + veh_avail_buffer - sum_avail_large_veh)
                        )
                        / veh_capacities[0]
                    ),
                )
            ]

        avail_ratio = [
            round(
                veh_capacities[0] * veh_availabilities[0] / self.stop_df.demand.sum(), 4
            )
        ]
        # Calculate vehicle availabilities
        if avail_ratio is None or len(avail_ratio) != no_of_vehicle_types:
            # linearly distribute the availability of large vehicles
            for i in range(1, no_of_vehicle_types):
                avail_ratio.append(
                    round(sum_avail_large_veh / (no_of_vehicle_types - 1), 2)
                )

        for i in range(1, no_of_vehicle_types):
            veh_availabilities.append(
                int(
                    max(
                        1,
                        np.ceil(
                            (self.stop_df.demand.sum() * avail_ratio[i]) / veh_capacities[i]
                        ),
                    )
                )
            )
        # ----------------------------------------------
        # Calculate vehicle costs
        no_of_routes_in_bks = len(self.bks_sol.route_dict)
        if fleet_scenario == "f_v_even":
            veh_base_costs = np.ceil(variable_costs / no_of_routes_in_bks)
        elif fleet_scenario == "f_dominant":
            veh_base_costs = np.ceil(
                fix_var_multiplier * variable_costs / no_of_routes_in_bks
            )
        elif fleet_scenario == "v_dominant":
            veh_base_costs = np.ceil(
                variable_costs / (no_of_routes_in_bks * fix_var_multiplier)
            )
        else:
            raise ValueError(
                f"The fleet scenario must be one of ['f_v_even', 'f_dominant', 'v_dominant']; not {self.fleet_scenario}"
            )
        cost_interval = (
            veh_base_costs / (capa_interval[1] / capa_interval[0] * capa_cost_ratio),
            veh_base_costs,
        )
        veh_costs = []
        for i in range(no_of_vehicle_types):
            veh_costs.append(
                np.ceil(
                    cost_interval[0]
                    + i
                    * (cost_interval[1] - cost_interval[0])
                    / (no_of_vehicle_types - 1)
                )
            )
        # Set variable_unit_costs to 1 for now
        variable_distance_unit_costs = list(np.ones(no_of_vehicle_types))
        variable_duration_unit_costs = list(np.zeros(no_of_vehicle_types))

        # Create vehicle_type_df
        self.veh_type_df = pd.DataFrame(
            {
                "veh_type": veh_types,
                "capacity": veh_capacities,
                "costs": veh_costs,
                "no_of_available": veh_availabilities,
                # calculate the share on the complete fleet based no_of_available
                "fleet_share": np.round(
                    np.array(veh_availabilities) / sum(veh_availabilities), 4
                ),
                "variable_distance_unit_costs": variable_distance_unit_costs,
                "variable_duration_unit_costs": variable_duration_unit_costs,
                "fix_capacity_unit_costs": np.round(
                    np.array(veh_costs) / np.array(veh_capacities), 2
                )
            }
        )
        # Add rank_fcuc and rank_vcuc
        self.veh_type_df["rank_fcuc"] = self.veh_type_df["fix_capacity_unit_costs"].rank(ascending=True, method="dense").astype(int)
        self.veh_type_df["rank_vduc"] = self.veh_type_df["variable_distance_unit_costs"].rank(ascending=True, method="dense").astype(int)

        # Store the vehicle type information as a dictionary to self
        self.veh_type_params = {
            "no_of_types": no_of_vehicle_types,
            "veh_cap_interval_str": capa_interval_str,
            "veh_avail_ratio": avail_ratio,
            "xs_fleet_size_str": xs_fleet_size,
            "veh_avail_buffer": veh_avail_buffer,
            "veh_capa_cost_ratio": capa_cost_ratio,
            "veh_base_costs": veh_base_costs,
            "sum_avail_large_veh": sum_avail_large_veh,
            "fleet_scenario": fleet_scenario,
            "fix_var_multiplier": fix_var_multiplier,
        }
        self.type = "HVRPTW"
        return self

    def solve(
        self,
        gs: "Hyperparameters",
        hp_R: "Hyperparameters"
    ) -> dict:
        """ """
        if hasattr(hp_R, "max_runtime") and hp_R.max_runtime < hp_R.min_runtime_per_pyvrp_call:
            hp_R.max_runtime = hp_R.min_runtime_per_pyvrp_call
        if hp_R.solver == "pyvrp":
            return solve_with_pyvrp_hgs(self, gs, hp_R)
        elif hp_R.solver == "vsr":
            from core.solving.solve_with_vsr import solve_with_vsr
            return solve_with_vsr(self, gs, hp_R)
        else:
            return None

    def transform_to_prize_collecting_problem(
        self, prize_setting_strategy: str
    ) -> "VRP_OBJECT":
        """"""
        # initialize the required column to False and the prize column to 0.0 to all stops for modeling the prize collecting problem
        self.stop_df["required"] = False
        self.stop_df["prize"] = 0.0
        depot_id = self.map_stop_id_idx[self.depot_identifier]
        max_variable_unit_costs = self.veh_type_df[
                    "variable_distance_unit_costs"
                ].max()
        max_fixed_costs = self.veh_type_df["costs"].max()

        if prize_setting_strategy == "constant":
            penalty_factor = np.max(self.distance_matrix[depot_id, :]) + np.max(self.distance_matrix[:, depot_id])
            self.stop_df["prize"] = penalty_factor  * max_variable_unit_costs + max_fixed_costs
        else:
            if "pendcosts" in prize_setting_strategy.lower():
                # Vectorize pendulum costs of every stop: i.e., sum of: distance_matrix[depot_id][stop] + distance_matrix[stop][depot_id]
                arr = (
                    self.distance_matrix[depot_id, self.stop_df.index]
                    + self.distance_matrix[self.stop_df.index, depot_id]
                )
                # Now multiply by costs and store in a 'prize' column
                # Make sure your self.stop_df.index is numeric or can be used to index into distance_matrix
                self.stop_df["prize"] = arr * max_variable_unit_costs + max_fixed_costs

            # If 'capaconsump' is in the strategy, vector-scale the prize for the non-depot rows
            if "capaconsump" in prize_setting_strategy.lower():
                max_cap = self.veh_type_df["capacity"].max()
                # Suppose index=0 is depot row, so we apply capacity scaling to the other rows:
                max_capa_div_demand = max_cap / self.stop_df.loc[self.stop_df.index != depot_id, "demand"]

                # Normalizing the 'scores' column to [0, 1]
                if not max_capa_div_demand.max() == max_capa_div_demand.min():
                    max_capa_div_demand_norm = (max_capa_div_demand - max_capa_div_demand.min()) / (max_capa_div_demand.max() - max_capa_div_demand.min())
                    self.stop_df.loc[self.stop_df.index != depot_id, "prize"] *= (
                        1 + max_capa_div_demand_norm
                    )
            """
            if "fuzzy" in prize_setting_strategy.lower():
                for stop in self.stop_df.index:
                    self.stop_df.at[stop, "prize"] = (
                        self.stop_df.at[stop, "prize"] * self.stop_df.at[stop, "u_c"]
                    )
            else:
                raise ValueError(
                    f"The prize setting strategy {prize_setting_strategy} is not yet supported."
                )
            #"""
        return self

    def write_to_json(
        self, path_to_write_json: str, init_file_name: str = None
    ) -> None:
        """
        Stores the instance data in a JSON file.

        Args:
            path_to_write_json (str): The path to the JSON file.
            init_file_name (str): The initial file name.

        Returns:
            None
        """
        util_write_object_to_json(
            self, path_to_write_json, sol_object=False, init_file_name=init_file_name
        )
