import logging
import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria, NoImprovement, FirstFeasible
from pyvrp.search import NeighbourhoodParams
from pyvrp.solve import SolveParams
import numpy as np
import time
from typing import TYPE_CHECKING
from warnings import warn

from core.data_manipulation.data_processing import data_cleansing_route

# pyvrp params
from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyParams
from pyvrp.Population import PopulationParams

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.vrp import VRP_OBJECT

logger = logging.getLogger(__name__)


def set_stopping_criteria(hps: "Hyperparameters") -> "MultipleCriteria":
    """
    Configures and returns a set of stopping criteria for PyVRP.

    This function takes an object of the class Hyperparameters and creates
    a list of corresponding criteria objects, which are used to determine when
    PyVRP should terminate.

    Args:
        hps (Hyperparameters): An object of the class Hyperparameters.
            Possible keys include:
            - "max_runtime" (int): Maximum runtime in seconds.
            - "nb_iter_without_improvement" (int): Maximum iterations without improvement.
            - "max_iterations" (int): Maximum number of iterations.

    Returns:
        MultipleCriteria: An object encapsulating the defined stopping criteria.

    Raises:
        Warning: If no stopping criteria are specified, defaults to a criterion
        of 20,000 iterations without improvement.

    Notes:
        The function defaults to a "NoImprovement" criterion with 20,000 iterations
        if no specific criteria are provided.
    """
    # Initialize an empty list to store stopping criteria
    stopping_criteria = []

    # Check and add a runtime criterion if specified in the dictionary
    if hasattr(hps, "max_runtime"):
        stopping_criteria.append(MaxRuntime(int(hps.max_runtime)))

    # Check and add a no-improvement criterion if specified in the dictionary
    if hasattr(hps, "nb_iter_without_improvement"):
        stopping_criteria.append(NoImprovement(hps.nb_iter_without_improvement))

    # Check and add a max-iterations criterion if specified in the dictionary
    if hasattr(hps, "max_iterations"):
        stopping_criteria.append(NoImprovement(hps.max_iterations))

    # If no criteria were set, use a default no-improvement criterion
    if len(stopping_criteria) == 0:
        warn(
            "No stopping criterion was set. Defaulting to 20,000 iterations without improvement."
        )
        stopping_criteria.append(NoImprovement(20000))

    # Return a MultipleCriteria object that combines all the specified criteria
    return MultipleCriteria(stopping_criteria)


def solve_with_pyvrp_hgs(
    instance: "VRP_OBJECT", gs: "Hyperparameters", hp_R: "Hyperparameters"
) -> tuple[dict, float, object]:
    """
    Solves a VRP instance using the PyVRP HGS solver with specified hyperparameters.

    This function prepares the model data, sets stopping criteria, configures solver
    parameters, and executes the HGS (Hybrid Genetic Search) algorithm to find a solution
    for the Vehicle Routing Problem (VRP).

    Notes:
        - Ensure that the `instance` is compatible with the PyVRP library.
        - The stopping criteria should be carefully chosen based on problem size and
          desired precision.
    """
    # Set the stopping criteria for the solver
    stopping_criterion = set_stopping_criteria(hps=hp_R)
    # Prepare model data with specified multiplier
    model = prepare_data_for_pyvrp(instance, gs.multiplier)

    # Configure solver parameters based on pruning preference
    if not hp_R.pruning:
        # Use a granular neighborhood parameter based on the instance dimension
        neighbourhood_params = NeighbourhoodParams(nb_granular=instance.dimension)
    else:
        # Default solver parameters without pruning
        neighbourhood_params = NeighbourhoodParams()
    solver_params = SolveParams(neighbourhood=neighbourhood_params)
    # Track solver runtime
    start_time = time.time()

    # Solve the model using the configured parameters
    result = model.solve(
        stop=stopping_criterion,
        seed=gs.seed,
        display=hp_R.verbose,
        params=solver_params,
    )

    # Calculate total runtime
    runtime = time.time() - start_time

    if not result.best.is_feasible():
        start_time_feas = time.time()
        stopping_criterion_feas = MultipleCriteria(
            [
                FirstFeasible(),
                MaxRuntime(
                    max(
                        1,
                        min(
                            int(instance.dimension * 0.33)
                            ,5
                        )
                    )
                )
            ]
            )
        result_feas = model.solve(
            stop=stopping_criterion_feas,
            seed=gs.seed,
            display=hp_R.verbose,
            params=solver_params,
        )
        runtime_feas = time.time() - start_time_feas
        runtime += runtime_feas
        if not result_feas.best.is_feasible():
            return {}, round(runtime, 2), result
        else:
            result = result_feas
    if result.best.num_clients() < 1:
        return {}, round(runtime, 2), result

    # Format and return the result along with runtime
    return format_pyvrp_output(instance, model, result.best, format_after_improvement=False), round(runtime, 2), result


def prepare_data_for_pyvrp(instance: "VRP_OBJECT", multiplier: int = 100):
    """
    Prepares data for the PyVRP model based on the given instance data.

    This function configures the PyVRP model by adding depots, clients,
    edges, and vehicle types based on the provided VRP instance data.

    Args:
        instance (VRP_OBJECT): The instance object containing the VRP data.
        multiplier (int): A scaling factor applied to coordinates and time windows.
            Defaults to 100.

    Returns:
        Model: A PyVRP model object with the prepared data.

    Example:
        >>> instance = load_vrp_instance("path/to/instance")
        >>> model = prepare_data_for_pyvrp(instance, multiplier=100)
        >>> print(model)

    Notes:
        - The multiplier is used to scale coordinates and time-related values
          to avoid precision issues.
        - Ensure the instance object is properly initialized with depot, clients,
          and vehicle type data before calling this function.
    """

    def add_a_client_to_model(model, row, multiplier):
        """ """
        x = row.x_coord 
        y = row.y_coord 
        delivery = row.demand
        name = row.stop_id

        if use_time_windows:
            tw_early = row.tw_start * multiplier
            tw_late = row.tw_end * multiplier
            service_duration = row.service_time * multiplier
        else:
            tw_early = 0
            tw_late = np.iinfo(np.int64).max
            service_duration = 0

        if use_prize_collecting:
            required = row.required
            prize = row.prize * multiplier * multiplier
        else:
            required = True
            prize = 0

        model.add_client(
            x=x,
            y=y,
            delivery=delivery,
            tw_early=tw_early,
            tw_late=tw_late,
            service_duration=service_duration,
            required=required,
            prize=prize,
            name=name,
        )

    def add_a_vehicle_type_to_model(model, row, multiplier):
        """ " """
        veh_type = row.veh_type
        num_available = int(row.no_of_available)
        capacity = int(row.capacity)
        fixed_cost = int(row.costs * multiplier * multiplier)
        unit_distance_cost = int(row.variable_distance_unit_costs * multiplier)

        if use_time_windows:
            tw_early = int(depot_row["tw_start"] * multiplier)
            tw_late = int(depot_row["tw_end"] * multiplier)
            unit_duration_cost = int(row.variable_duration_unit_costs * multiplier)
        else:
            tw_early = 0
            tw_late = np.iinfo(np.int64).max
            unit_duration_cost = 0

        model.add_vehicle_type(
            name=veh_type,
            num_available=num_available,
            capacity=capacity,
            fixed_cost=fixed_cost,
            unit_distance_cost=unit_distance_cost,
            unit_duration_cost=unit_duration_cost,
            tw_early=tw_early,
            tw_late=tw_late,
        )

    model = Model()
    # Add depot to the model
    d_id = instance.depot_identifier
    depot_row = instance.stop_df.loc[instance.stop_df["stop_id"] == d_id].iloc[0]
    model.add_depot(
        x=depot_row["x_coord"] * multiplier,
        y=depot_row["y_coord"] * multiplier,
    )

    # -------------------------------------------------
    # 1) Check once whether we have time windows or prize collecting
    # -------------------------------------------------
    use_time_windows = "TW" in instance.type
    use_prize_collecting = "PC" in instance.type

    # -------------------------------------------------
    # Add clients
    # -------------------------------------------------
    client_stops = instance.stop_df[instance.stop_df["stop_id"] != d_id]
    for row in client_stops.itertuples(index=False):
        add_a_client_to_model(model, row, multiplier)

    # -------------------------------------------------
    # Add edges (distance + duration)
    # -------------------------------------------------
    # Add edges to the model based on the distance matrix
    locations = [*model._depots, *model._clients]
    # check if instance.duration_matrix exists
    if not hasattr(instance, "duration_matrix"):
        # create a copy of the distance matrix to use as duration matrix
        instance.duration_matrix = instance.distance_matrix.copy()

    dist_matrix_scaled = instance.distance_matrix * multiplier
    dur_matrix_scaled = instance.duration_matrix * multiplier
    add_edge_bind = model.add_edge
    for f_idx, frm in enumerate(locations):
        for t_idx, to in enumerate(locations):
            add_edge_bind(
                frm,
                to,
                distance=int(dist_matrix_scaled[f_idx, t_idx]),
                duration=int(dur_matrix_scaled[f_idx, t_idx]),
            )
    # delete the duration matrix to avoid confusion if it is a copy of the distance matrix
    if instance.distance_matrix is instance.duration_matrix:
        del instance.duration_matrix

    # -------------------------------------------------
    # Add vehicle types
    # -------------------------------------------------
    for row in instance.veh_type_df.itertuples(index=False):
        add_a_vehicle_type_to_model(model, row, multiplier)

    return model


def format_pyvrp_output(instance: "VRP_OBJECT", pyvrp_model: object, pyvrp_solution: object, format_after_improvement: bool = False) -> dict:
    """
    Formats the output of the PyVRP model into a dictionary containing routes.

    This function processes the result from the PyVRP solver and organizes
    the solution into a dictionary format, including vehicle types and routes.

    Args:
        instance (VRP_OBJECT): The instance object containing the VRP data.
        model (Model): The PyVRP model object used for solving.
        result (Result): The PyVRP result object containing the solution.

    Returns:
        dict: A dictionary containing the routes of the solution, indexed by vehicle index.
            Each entry includes the vehicle type and the list of stop IDs in the route.

    Example:
        >>> instance = load_vrp_instance("path/to/instance")
        >>> model = prepare_data_for_pyvrp(instance)
        >>> result = model.solve(...)
        >>> formatted_output = format_pyvrp_output(instance, model, result)
        >>> print(formatted_output)

    Notes:
        - Ensure the result is feasible before calling this function.
        - The function assumes that the depot identifier and vehicle types are correctly
          defined in the model.
    """
    result_route_dict = {}
    # Check if the result is feasible
    if pyvrp_solution.is_feasible():
        # Iterate over each vehicle's route in the best solution
        for veh_idx, route in enumerate(pyvrp_solution.routes()):
            # Retrieve the vehicle type for the current route
            r_veh = pyvrp_model._vehicle_types[route.vehicle_type()].name

            # Cleanse the route by ensuring depot inclusion
            # transform the route to a list of stop IDs based on instance.map_idx_stop_id
            route_stop_id = [instance.map_idx_stop_id[stop] for stop in route.visits()]
            r_v = data_cleansing_route(route_stop_id, instance.depot_identifier)

            # Create unique identifier of route based on how it was created
            if format_after_improvement:
                route_key = f"{instance.name}_I+vt{r_veh}-{str(veh_idx)}"
            else:
                route_key = f"{instance.name}+vt{r_veh}-{str(veh_idx)}"

            # Add the route to the result dictionary, keyed by vehicle index
            result_route_dict[route_key] = {
                "veh_id": f"vt{r_veh}-{str(veh_idx)}",
                "route_id": route_key,
                "veh_type": r_veh,
                "route": r_v,
            }
    return result_route_dict
