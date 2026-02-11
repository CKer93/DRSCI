from copy import deepcopy
from math import ceil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Hyperparameters
    from core.vrp import VRP_OBJECT

def custom_fcm_clustering(
    feature_arry: np.array,
    similarity_matrix: np.array,
    no_of_clusters: int,
    seed: int
) -> tuple:
    """
    Performs Fuzzy C-Means (FCM) clustering on the given feature array.

    This function applies the Fuzzy C-Means (FCM) clustering algorithm to the input
    feature array and similarity matrix, identifying the specified number of clusters.
    The function uses a seeded approach for initialization and returns cluster labels
    and the highest degree of membership values for each data point.

    Args:
        feature_arry (np.array): A 2D numpy array where each row represents a data point
            and each column represents a feature.
        similarity_matrix (np.array): A 2D numpy array representing the similarity matrix
            between data points.
        no_of_clusters (int): The number of clusters to form.
        seed (int, optional): The random seed for initialization. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - labels (np.array): A 1D numpy array with cluster labels for each data point.
            - u_max (np.array): A 1D numpy array with the highest degree of membership values
              for each data point.

    Example:
        >>> import numpy as np
        >>> feature_array = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        >>> similarity_matrix = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        >>> labels, u_max = custom_fcm_clustering(feature_array, similarity_matrix, 2, seed=42)
        >>> print(labels)
        >>> print(u_max)

    Notes:
        - The function uses the cmedoids method from a clustering library to perform FCM.
        - The similarity matrix should reflect the pairwise similarities between data points.
        - The resulting labels and membership values provide insight into the clustering structure.
    """
    # Perform Fuzzy C-Means clustering
    cntr, u, u0, d, jm, p, fpc = cmedoids(
        feature_array=feature_arry.T,
        similarity_matrix=similarity_matrix,
        c=no_of_clusters,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=seed,
        init="kmeans++_seed_medoids"
    )

    # Initialize labels and u_max arrays with -1
    labels = np.array([-1 for _ in range(len(feature_arry))])
    u_max = np.array([-1 for _ in range(len(feature_arry))]).astype(float)

    # Assign cluster labels and highest degree of membership values
    for i in range(len(feature_arry)):
        labels[i] = np.argsort(u[:, i])[-1]
        u_max[i] = np.round(np.max(u[:, i]), 3)

    return labels, u_max

def route_based_clustering(
    route_df : pd.DataFrame,
    stop_df : pd.DataFrame,
    feature_array : np.array,
    hp_D : "Hyperparameters",
    seed : int,
):
    # ensure that there are more routes than number of clusters
    labels_routes = (
        KMeans(n_clusters=min(hp_D.number_of_clusters, len(feature_array)), random_state=seed)
        .fit(feature_array)
        .labels_
    )
    # add cluster labels to route_df
    df = deepcopy(route_df)
    df["c_idx"] = labels_routes.astype(str)
    # Group by `c_idx` and concatenate lists
    grouped_routes = df.groupby("c_idx")["route"].apply(lambda x: [item for sublist in x for item in sublist[1:-1]]).to_dict()
    # Assign c_idx to stops based on grouped_routes
    for c_idx, stops in grouped_routes.items():
        stop_df.loc[stop_df["stop_id"].isin(stops), "c_idx"] = c_idx
    labels_vertices = stop_df["c_idx"].iloc[1:]
    # Create a dummy u_max array with 1s for K-Means
    u_max = np.ones(len(labels_vertices), dtype=float)
    return labels_vertices, u_max

def set_number_of_cluster_range(
    vrp_object: "VRP_OBJECT",
    hps_object: "Hyperparameters"
) -> list:
    """
    Determines a range of cluster counts for a VRP instance.

    This function calculates the range of the number of clusters based on
    the average number of stops per cluster, using the specified number of cluster setting strategy.
    It returns a list of possible cluster counts within the calculated range.

    Args:
        vrp_object (VRP_OBJECT): An instance of VRP_OBJECT containing the VRP data,
            including the total number of stops (dimension).
        hps_object (Hyperparameters): An object of Hyperparameters containg the required parameters.

    Returns:
        list: A list of integers representing the possible number of clusters within the
              calculated range.

    Notes:
        - The function uses the ceil function to ensure the upper bound includes the ceiling
          of the division result.
        - The lower bound is set to a minimum of 2 to avoid having less than two clusters.

    """


    if hps_object.cluster_range_rule == "bounded":
        # Calculate the upper bound of the cluster range
        range_ub = ceil(vrp_object.dimension / hps_object.avg_stops_per_cluster_lb)
        # Calculate the lower bound of the cluster range, ensuring a minimum of 2 clusters
        range_lb = max(2, ceil(vrp_object.dimension / hps_object.avg_stops_per_cluster_ub))
        # Return the range of cluster counts as a list
        return list(range(range_lb, range_ub + 1))
    elif hps_object.cluster_range_rule == "expected_min_max_routes":
        # Calculate ub of the cluster range based on the maximum expected routes used
        max_cluster_range = min(
            int(
                vrp_object.stop_df.demand.sum() / vrp_object.veh_type_df.capacity.min()
            ),
            vrp_object.veh_type_df.no_of_available.sum(),
            int(vrp_object.dimension/hps_object.avg_stops_per_cluster_lb)
        )
        # Calculate the lb of the cluster range based on the theoretical minimum routes required
        min_cluster_range = int(
            vrp_object.stop_df.demand.sum() /vrp_object.veh_type_df.capacity.max(),
            )
        if max_cluster_range > min_cluster_range:
            return list(range(min_cluster_range, max_cluster_range+1))
        else:
            return [min_cluster_range]
    elif hps_object.cluster_range_rule == "theoretical_min_routes":
        # Calculate the lb of the cluster range based on the theoretical minimum routes required
        min_cluster_range = ceil(
            vrp_object.stop_df.demand.sum() / vrp_object.veh_type_df.capacity.max(),
            )
        return [min_cluster_range]

    else:
        raise ValueError(f"{hps_object.cluster_range_rule} is not defined.")

def vertex_based_clustering(
        feature_array : np.array,
        similarity_matrix : np.array,
        hp_D : "Hyperparameters",
        seed : int,
) -> tuple[np.array, np.array]:
    # Perform clustering based on the specified method
    number_of_clusters = hp_D.number_of_clusters
    # Perform K-Means clustering --> replace with K-Medoids
    labels = (
        KMeans(n_clusters=number_of_clusters, random_state=seed)
        .fit(feature_array)
        .labels_
    )
    # Create a dummy u_max array with 1s for K-Means
    u_max = np.ones(len(feature_array), dtype=float)
    return labels, u_max

