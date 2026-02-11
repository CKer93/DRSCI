import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters


def caculate_similarity_matrix(
    hp_D: "Hyperparameters",
    stop_df: pd.DataFrame,
    dec_prec: int = 4,
) -> tuple[np.array, np.array]:
    """
    Calculates the similarity matrix for stops based on the specified similarity metric.

    This function computes a similarity matrix for stops in a VRP instance using different
    similarity metrics such as coordinates, angles to the depot, or a hybrid of both. The
    similarity matrix is calculated using the specified precision and optionally weighted
    for hybrid metrics.

    Args:
        stop_df (pd.DataFrame): A DataFrame containing the stop information, including
            coordinates and other relevant features.
        travel_time_matrix (np.array): A 2D numpy array representing the travel time matrix between stops. Relevant for the std metric to calculate the penalties.
        veh_capacity (int): The vehicle capacity for the VRP instance. It is used for the std metric.
        hp_dict (dict, optional): A dictionary of hyperparameters for similarity calculation.
            The dictionary can include:
            - 'similarity_metric_type' (str): The type of similarity metric ('coords',
              'dptangl', or 'hybrid').
            - 'hybrid_angle_weight' (str, optional): The weight for the angle component
              in the hybrid similarity metric. If not provided, it will be calculated as
              the mean of the coordinates.

    Returns:
        tuple: A tuple containing:
            - similarity_matrix (np.array): A 2D numpy array representing the similarity
              matrix between stops, excluding the depot.
            - feature_data (np.array): A 2D numpy array of the feature data used in the
              similarity calculation, excluding the depot.

    Example:
        >>> import pandas as pd
        >>> stop_df = pd.DataFrame({
        ...     'x_coord': [0, 1, 2],
        ...     'y_coord': [0, 1, 2]
        ... })
        >>> travel_time_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        >>> veh_capacity = 20
        >>> hp_dict = {
        ...     'similarity_metric_type': 'coords',
        ... }
        >>> similarity_matrix, features = caculate_similarity_matrix(stop_df, hp_dict)
        >>> print(similarity_matrix)

    Notes:
        - The function assumes that the first stop in the DataFrame is the depot and excludes
          it from the similarity matrix.
        - The `get_angles` function is used to calculate angles to the depot for 'dptangl'
          and 'hybrid' metrics.
        - For hybrid metrics, normalization and weighting of angles can be adjusted.
    """

    similarity_metric_type = hp_D.similarity_metric_type
    similarity_measure = hp_D.similarity_measure
    if hp_D.hybrid_angle_weight not in ["balanced", "coords_dominant", "dptangl_dominant"]:
        print("Invalid hybrid angle weight. Using 'balanced' by default.")
        hybrid_angle_weight = "balanced"
    else:
        hybrid_angle_weight = hp_D.hybrid_angle_weight
        
    if similarity_metric_type == "coords":
        feature_data = stop_df[["x_coord", "y_coord"]]
    elif similarity_metric_type in ["dptangl", "hybrid"]:
        # Calculate angles to the depot
        node_depot_angles = get_angles(
            instance_node_df=stop_df, arc_rot=hp_D.stop_depot_angle_orientation, dec_prec=dec_prec
        )
        stop_df["node_depot_angle"] = node_depot_angles
        feature_data = stop_df[["node_depot_angle"]]

        if similarity_metric_type == "hybrid":
            if hybrid_angle_weight == "balanced":
                hybrid_angle_weight = np.round(
                    stop_df[["x_coord", "y_coord"]].mean().mean(), dec_prec
                )
            elif hybrid_angle_weight == "coords_dominant":
                hybrid_angle_weight = (
                    np.round(stop_df[["x_coord", "y_coord"]].mean().mean(), dec_prec)
                    / 2
                )
            else:
                hybrid_angle_weight = (
                    np.round(stop_df[["x_coord", "y_coord"]].mean().mean(), dec_prec)
                    * 2
                )
            stop_df["node_depot_angle"] = node_depot_angles * hybrid_angle_weight
            feature_data = stop_df[["x_coord", "y_coord", "node_depot_angle"]]
    else:
        raise ValueError(
            "Invalid similarity metric type. Choose 'coords', 'dptangl', or 'hybrid'."
        )
    # Calculate the similarity matrix and round to the specified precision
    similarity_matrix = np.round(cdist(feature_data, feature_data), dec_prec)

    # Exclude the depot (first row and column) from the similarity matrix
    return similarity_matrix[1:, 1:], np.array(feature_data[1:])


def get_angles(
    instance_node_df: pd.DataFrame, arc_rot: str = "ccw", dec_prec: int = 4
) -> np.array:
    """
    Calculates the angles between the depot and each point in the instance node DataFrame.

    This function computes the angles between the depot and other points using their
    coordinates. The angles are calculated using the `true_arctan` function, and the
    results are returned as a numpy array.

    Args:
        instance_node_df (pd.DataFrame): A DataFrame containing the coordinates of the depot
            and other points. It must include 'x_coord' and 'y_coord' columns.
        arc_rot (str, optional): The direction of the arc rotation ('ccw' for counter-clockwise
            or 'cw' for clockwise). Defaults to 'ccw'.

    Returns:
        np.array: An array of angles between the depot and each point in radians, rounded to
        the specified precision.

    Example:
        >>> import pandas as pd
        >>> instance_node_df = pd.DataFrame({
        ...     'x_coord': [0, 1, 2],
        ...     'y_coord': [0, 1, 2]
        ... })
        >>> angles = get_angles(instance_node_df)
        >>> print(angles)

    Notes:
        - The depot is assumed to be the first point (index 0) in the DataFrame.
        - The `true_arctan` function is used to calculate the angles, which handles both
          clockwise and counter-clockwise rotations.
    """
    angles_to_depot = []
    for i in range(instance_node_df.shape[0]):
        angle = round(
            true_arctan(
                instance_node_df["x_coord"].iloc[i]
                - instance_node_df["x_coord"].iloc[0],
                instance_node_df["y_coord"].iloc[i]
                - instance_node_df["y_coord"].iloc[0],
                arc_rot=arc_rot,
            ),
            dec_prec,
        )
        angles_to_depot.append(angle)

    return np.array(angles_to_depot)


def true_arctan(x: float, y: float, arc_rot: str = "ccw") -> float:
    """
    Calculates the arctangent of y/x with consideration for rotation direction.

    This function computes the arctangent of the given coordinates, adjusting for
    clockwise or counter-clockwise rotation as specified.

    Args:
        x (float): The x-coordinate difference.
        y (float): The y-coordinate difference.
        arc_rot (str, optional): The direction of the arc rotation ('ccw' for counter-clockwise
            or 'cw' for clockwise). Defaults to 'ccw'.

    Returns:
        float: The angle in radians between the positive x-axis and the point (x, y).

    Example:
        >>> angle = true_arctan(1, 1)
        >>> print(angle)
        >>> angle = true_arctan(1, 1, arc_rot='cw')
        >>> print(angle)

    Notes:
        - The function uses numpy's arctan2 to calculate the angle, which returns values
          in the range [-π, π].
        - The direction of rotation affects how the angle is adjusted.
    """
    if arc_rot == "ccw":
        return np.arctan2(y, x)
    else:
        if y > 0:
            return np.arctan2(y, x) - np.pi
        else:
            return np.arctan2(y, x) + np.pi
