import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import MapClass

def method1():
    pass

def method2(semantic_map,feature_num,distance_type,n_clusters,merge_method="ward"):
    print("Calculating Distances")
    distances, valid_masks = generate_distance_matrix_subspace(semantic_map,feature_num,distance_type)
    print(distances)
    print("Clustering...")
    linked = linkage(distances, method=merge_method)
    clusters = fcluster(linked, t=n_clusters, criterion='maxclust')
    print("Done!")
    return clusters, valid_masks

def generate_distance_matrix_subspace(semantic_map,k,distance_type="projection"):
    mask_ids = []
    subspaces = []
    for img_id, local_masks in semantic_map.refined_prediction_masks.items():
        for mask_id in local_masks.keys():
            _,valid_indices = semantic_map.get_valid_keypoints(img_id,mask_id)
            data_points = semantic_map.descriptors[img_id][valid_indices].copy()
            if data_points.shape[0] < k:
                continue
            subspaces.append(get_subspace(data_points.T,k))
            mask_ids.append((img_id,mask_id))
    distances = np.zeros((len(subspaces), len(subspaces)))

    # Compute the pairwise distances only for the upper triangle (including diagonal)
    for i in range(len(subspaces)):
        for j in range(i, len(subspaces)):  # j starts from i to avoid redundant calculations
            if distance_type == "projection":
                dist = projection_distance(subspaces[i], subspaces[j])
            elif distance_type == "chordal_distance":
                dist = chordal_distance(subspaces[i], subspaces[j])
            distances[i, j] = dist
            distances[j, i] = dist
    return distances, mask_ids

def get_subspace(M,k):
    """
    Computes a k-dimensional orthonormal basis for the subspace spanned by the columns of M
    using Singular Value Decomposition (SVD).

    The function first centers the data in M by subtracting the mean of each row (feature),
    then performs SVD to identify the top-k principal components (left singular vectors),
    which form an orthonormal basis for the desired k-dimensional subspace.

    Parameters:
    ----------
    M : np.ndarray
        A 2D NumPy array of shape (d, n), where d is the dimensionality of each data point
        and n is the number of data points.
    k : int
        The number of leading components to extract (i.e., the dimensionality of the subspace).

    Returns:
    -------
    np.ndarray
        A (d, k) NumPy array whose columns form an orthonormal basis for the k-dimensional subspace.
    """

    M = M - np.mean(M, axis=1, keepdims=True)  # zero-mean across points
    # Perform SVD
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    # Take the top-k left singular vectors => basis for k-dimensional subspace
    return U[:, :k]

def projection_distance(U, V):
    """
    Computes the projection (or subspace) distance between two subspaces represented by orthonormal bases U and V.

    The distance is defined as the Frobenius norm of the difference between the projection matrices of the subspaces.
    This measure is invariant to the choice of basis and reflects how different the two subspaces are.

    Parameters:
    ----------
    U : np.ndarray
        A (d, k1) matrix whose columns form an orthonormal basis for a k1-dimensional subspace in R^d.
    V : np.ndarray
        A (d, k2) matrix whose columns form an orthonormal basis for a k2-dimensional subspace in R^d.

    Returns:
    -------
    float
        The Frobenius norm of the difference between the projection matrices of U and V.
    """
    P_U = U @ U.T
    P_V = V @ V.T
    return np.linalg.norm(P_U - P_V, 'fro')

def chordal_distance(U, V):
    """
    Computes the chordal distance between two subspaces represented by orthonormal bases U and V.

    The chordal distance is based on the principal angles between subspaces, and is defined as
    the square root of the sum of the squared sine values of those angles. It provides a measure 
    of similarity between subspaces, with a value of 0 indicating identical subspaces.

    Parameters:
    ----------
    U : np.ndarray
        A (d, k1) matrix with orthonormal columns representing a k1-dimensional subspace in R^d.
    V : np.ndarray
        A (d, k2) matrix with orthonormal columns representing a k2-dimensional subspace in R^d.

    Returns:
    -------
    float
        The chordal distance between the two subspaces.
    """
    M = U.T @ V
    # Use numpy.linalg.svd
    sigma = np.linalg.svd(M, compute_uv=False)
    sin_squared = 1 - sigma**2
    return np.sqrt(np.sum(sin_squared))

def compare_with_gt(object_dict, clusters, valid_masks):
    pred_dict = {}
    for object_label, masks_list in object_dict.items():
        for masks in masks_list:
            if masks in valid_masks:
                index = valid_masks.index(masks)
                if object_label not in pred_dict:
                    pred_dict[object_label] = []
                pred_dict[object_label].append(clusters[index])
    return pred_dict

def main():
    with open('../latest.pkl','rb') as f:
        semantic_map = pickle.load(f)
    object_dict = semantic_map.group_by_matched_label()
    sorted_object_dict = dict(sorted(object_dict.items()))
    clusters, valid_masks = method2(semantic_map,15,"projection",40)
    print(compare_with_gt(sorted_object_dict,clusters, valid_masks))

if __name__ == "__main__":
    main()
