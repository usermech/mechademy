import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
import pickle
import os
import sys
import networkx as nx
from collections import defaultdict, Counter
import community
from sklearn.cluster import SpectralClustering

from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MapClass

def method1():
    pass

def method2(semantic_map,feature_num,distance_type,n_clusters,merge_method="average"):
    distances, valid_masks = generate_distance_matrix_subspace(semantic_map,feature_num,distance_type)
    distances = row_wise_min_max_normalize(distances)
    distances = 0.5 * (distances + distances.T)

    linked = linkage(distances, method=merge_method)
    clusters = fcluster(linked, t=n_clusters, criterion='maxclust')
    return clusters, valid_masks

def method3(semantic_map, min_num_feature=15, score_threshold=0.3):
    graph = semantic_map.G
    masks_with_keypoints = {}
    for img_id, val in semantic_map.refined_prediction_masks.items():
        if img_id not in masks_with_keypoints.keys():
            masks_with_keypoints[img_id] = {}
        for mask_id, mask in val.items():
            _,ids = semantic_map.get_valid_keypoints(img_id,mask_id)
            if len(np.where(ids==1)[0]) >= min_num_feature:
                masks_with_keypoints[img_id][mask_id]= {"mask":mask,"valid_keypoints":np.where(ids==1)[0]}
    edges_list=[]
    for node0, node1, attrs in graph.edges(data=True):
        matches = attrs["matches"].cpu().numpy()
        lookup_table = {kpt: mask_id1 for mask_id1, mask_kpts1 in masks_with_keypoints[node1].items() for kpt in mask_kpts1["valid_keypoints"]}
        for mask_id0, mask_kpts0 in masks_with_keypoints[node0].items():
            mask0, valid_kps0 = mask_kpts0["mask"],mask_kpts0["valid_keypoints"]
            # make a new list if element in valid_indices is in matches01[:, 0] without iterating
            valid_indices = np.intersect1d(valid_kps0, matches[:, 0])
            # find the index of the valid_indices in matches01[:, 0]
            match_indices = np.where(np.isin(matches[:, 0], valid_indices))[0]
            corresponding_indices = matches[match_indices, 1].tolist()

            corresponding_objects = [(mask_id0, lookup_table.get(kpts1, None)) for kpts1 in corresponding_indices
                            if lookup_table.get(kpts1, None) is not None]
            # find the corresponding objects in objects1 if key is not in lookup_table, skip it
            object_feature_pair = [((mask_id0, lookup_table.get(kpts1, None)),(kpts0,kpts1), ) for kpts0, kpts1 in zip(valid_indices, corresponding_indices)
                                    if lookup_table.get(kpts1, None) is not None]
            try:
                corresponding_objects, corresponding_features = zip(*object_feature_pair)
                # make them lists
                corresponding_objects, corresponding_features = list(corresponding_objects), list(corresponding_features)
            except ValueError:
                corresponding_objects, corresponding_features = [], []
            if corresponding_objects:
                # Find the count for each pair
                pair_count = Counter(corresponding_objects)

                # take the maximum count and the corresponding pair
                max_pair, max_count = pair_count.most_common(1)[0]

                confidence = max_count / len(corresponding_objects)
                min_feat_count = min(len(masks_with_keypoints[node0][max_pair[0]]["valid_keypoints"]), len(masks_with_keypoints[node1][max_pair[1]]["valid_keypoints"]))
                avg_feat_count = (len(masks_with_keypoints[node0][max_pair[0]]["valid_keypoints"]) +len(masks_with_keypoints[node1][max_pair[1]]["valid_keypoints"])) / 2
                score = confidence * max_count**2 / (avg_feat_count * min_feat_count)
                # pair_scores.append((max_pair, score))
                if score >= score_threshold:
                    edges_list.append(((node0,max_pair[0]),(node1,max_pair[1])))
                    
    return edges_list

def method4(semantic_map,feature_num,distance_type):
    distances, valid_masks = generate_distance_matrix_subspace(semantic_map,feature_num,distance_type)
    gt_correspondences = semantic_map.gt_pred_correspondences.copy()
    edges_list = []
    for label in range(distances.shape[0]):
        cost = distances[label,:]
        non_zero_mean = cost[cost != 0].mean()
        cost[cost == 0] = non_zero_mean
        similar_labels = z_score_low_anomaly(cost)
        if similar_labels.shape[0]>0:
            for similar_label in similar_labels:
                target = valid_masks[label]
                edges_list.append((target,valid_masks[similar_label]))
    subgraphs = extract_subgraphs(edges_list)
    return filter_subgraphs(subgraphs,edges_list,gt_correspondences)
    
def filter_subgraphs(subgraphs,edges_list,gt_correspondences):
    true_labels, predicted_labels = [],[]
    cluster_count = 0
    for subgraph in subgraphs:
        if len(subgraph) < 3:
            continue
        elif len(subgraph) >= 100:
            true_labels,predicted_labels,cluster_count = partition_subgraph(subgraph,edges_list,gt_correspondences,true_labels, predicted_labels,cluster_count)
        else:
            for node in subgraph:
                true_labels.append(gt_correspondences[node[0]][node[1]])
                predicted_labels.append(cluster_count)
            cluster_count += 1

    return true_labels, predicted_labels 

def partition_subgraph(subgraph,edges_list,gt_correspondences, true_labels, predicted_labels,cluster_count):
    graph_edges = []
    for match in edges_list:
        if match[0] in subgraph and match[1] in subgraph:
            graph_edges.append(match)
    G = nx.Graph()
    G.add_edges_from(graph_edges)
    partition = community.best_partition(G)
    for node, label in partition.items():
        true_labels.append(gt_correspondences[node[0]][node[1]])
        predicted_labels.append(int(label+cluster_count))
    cluster_count += len(partition)
    return true_labels, predicted_labels,cluster_count




def row_wise_min_max_normalize(D, exclude_diagonal=True):

    """
    Normalizes each row of the distance matrix using min-max normalization.
    
    Parameters:
    - D (ndarray): NxN distance matrix.
    - exclude_diagonal (bool): Whether to exclude diagonal entries from min/max computation.

    Returns:
    - D_norm (ndarray): Row-normalized distance matrix.
    """
    D = np.array(D, dtype=float)  # Ensure it's a float array
    D_norm = np.zeros_like(D)

    for i in range(D.shape[0]):
        row = D[i]
        
        if exclude_diagonal:
            mask = np.arange(D.shape[1]) != i
            row_min = np.min(row[mask])
            row_max = np.max(row[mask])
        else:
            row_min = np.min(row)
            row_max = np.max(row)

        denom = row_max - row_min
        if denom == 0:
            D_norm[i] = 0  # or leave the row unchanged
        else:
            D_norm[i] = (row - row_min) / denom

        if exclude_diagonal:
            D_norm[i, i] = 0  # optionally keep diagonal at zero

    return D_norm

def cur_reduction(descriptors):
    # Take svd of the descriptors
    U, S, Vt = np.linalg.svd(descriptors, full_matrices=False)
    try:
        cut_off = (np.where(S < 0.1)[0][0])
        best_frobenius_norm = float('inf')
        best_row_indexes = None
        for j in range(1000):
            col_indexes = np.random.choice(descriptors.shape[1], cut_off, replace=False)
            row_indexes = np.random.choice(descriptors.shape[0], cut_off, replace=False)
            # Create the CUR matrix
            C = descriptors[:, col_indexes]

            R = descriptors[row_indexes, :]

            W = descriptors[np.ix_(row_indexes, col_indexes)]
            U = np.linalg.pinv(W) 

            # Compute the CUR matrix
            CUR = C @ U @ R

            # Compute the Frobenius norm of the difference between the CUR matrix and the original matrix
            frobenius_norm = np.linalg.norm(CUR - descriptors, 'fro')
            if frobenius_norm < best_frobenius_norm:
                best_frobenius_norm = frobenius_norm
                best_row_indexes = row_indexes
        reduced_descriptors = descriptors[best_row_indexes, :]
    except:
        reduced_descriptors = descriptors
    mean_vec = np.mean(reduced_descriptors, axis=0)
    return mean_vec


def generate_distance_matrix_subspace(semantic_map,k,distance_type="projection"):
    mask_ids = []
    subspaces = []
    for img_id, local_masks in semantic_map.refined_prediction_masks.items():
        for mask_id in local_masks.keys():
            _,valid_indices = semantic_map.get_valid_keypoints(img_id,mask_id)
            try:
                score = semantic_map.gt_pred_correspondences[img_id][mask_id][1]
            except:
                continue
            if score < 0.5:
                continue
            data_points = semantic_map.descriptors[img_id][valid_indices].copy()
            if data_points.shape[0] < k:
                continue
            subspaces.append(get_subspace(data_points.T,k))
            mask_ids.append((img_id,mask_id))
    distances = np.zeros((len(subspaces), len(subspaces)))

    # Compute the pairwise distances only for the upper triangle (including diagonal)
    for i in range(len(subspaces)):
        for j in range(i, len(subspaces)):  # j starts from i to avoid redundant calculations
            if i == j:
                dist = 0
            elif distance_type == "projection":
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

def extract_subgraphs(edges_list):
    G = nx.Graph()
    G.add_edges_from(edges_list)
    
    # Get disconnected subgraphs
    subgraphs = list(nx.connected_components(G))
    subgraphs = [subgraph for subgraph in subgraphs if len(subgraph) >= 3]

    return subgraphs

def extract_labels_from_subgraphs(subgraphs, gt_correspondences):
    true_labels, predicted_labels = [], []
    instances = 0
    for label, subgraph in enumerate(subgraphs):
        instances += len(subgraph)
        for img_id, mask_id in subgraph:
            try:
                gt_label = gt_correspondences[img_id][mask_id][0]
                true_labels.append(gt_label)
                predicted_labels.append(label)
            except KeyError:
                continue
    return true_labels, predicted_labels, instances

def z_score_low_anomaly(data, threshold=3.0):
    mean = np.nanmean(data)  # Computes the mean while ignoring NaNs
    std = np.nanstd(data)    # Computes the std while ignoring NaNs
    
    # If the standard deviation is zero, we can't compute the z-scores
    if std == 0:
        print("Warning: Standard deviation is zero, can't compute z-scores.")
        return np.array([])  # Return an empty array as no anomalies can be found
    
    # Calculate z-scores
    z_scores = (data - mean) / std  # No abs() so it just gets lower anomalies

    return np.where(z_scores < -threshold)[0]


def flatten_equivalents(equivalents):
    visited = {}
    cluster_id = 0
    for key in equivalents:
        if key not in visited:
            stack = [key]
            group = set()
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited[node] = cluster_id
                    group.add(node)
                    stack.extend(equivalents[node] - group)
            cluster_id += 1
    return visited

def merge_predicted_labels_zscore(subgraphs,semantic_map,predicted_labels,k=10,distance_type="projection",z_threshold=3.0):
    subspaces = []
    for label, subgraph in enumerate(subgraphs):
        descriptors = np.empty((0,256))
        for img_id, mask_id in subgraph:
            _, valid = semantic_map.get_valid_keypoints(img_id,mask_id)
            descriptor_local = semantic_map.descriptors[img_id][valid].copy()
            descriptors = np.vstack([descriptors,descriptor_local])
        if descriptors.shape[0]>=k:
            subspaces.append(get_subspace(descriptors.T,k))
        distances = np.zeros((len(subspaces), len(subspaces)))

    # Compute the pairwise distances only for the upper triangle (including diagonal)
    for i in range(len(subspaces)):
        for j in range(i, len(subspaces)):  # j starts from i to avoid redundant calculations
            if i == j:
                dist = 0
            elif distance_type == "projection":
                dist = projection_distance(subspaces[i], subspaces[j])
            elif distance_type == "chordal_distance":
                dist = chordal_distance(subspaces[i], subspaces[j])
            distances[i, j] = dist
            distances[j, i] = dist

    label_equivalents = defaultdict(set)    
    for label, subgraph in enumerate(subgraphs):
        cost = distances[label,:]
        filtered_cost = cost[cost != 0]
        if filtered_cost.size > 0:
            non_zero_mean = filtered_cost.mean()
        else:
            non_zero_mean = 0
        cost[cost == 0] = non_zero_mean
        similar_labels = z_score_low_anomaly(cost,z_threshold)
        if similar_labels.shape[0]>0:
            for similar_label in similar_labels:
                label_equivalents[label].add(similar_label)
                label_equivalents[similar_label].add(label)  # Ensure bidirectionality
    final_label_mapping = flatten_equivalents(label_equivalents)
    remapped_labels = np.array([
    final_label_mapping.get(label, label)
    for label in predicted_labels
    ])
    return remapped_labels



def main():
    '''
    with open('../latest.pkl','rb') as f:
        semantic_map = pickle.load(f)
    object_dict = semantic_map.group_by_matched_label()
    sorted_object_dict = dict(sorted(object_dict.items()))
    clusters, valid_masks = method2(semantic_map,15,"projection",40)
    print(compare_with_gt(sorted_object_dict,clusters, valid_masks))
    '''
    with open('../../temp/clean.pkl','rb') as f:
        semantic_map = pickle.load(f)

 
    scores_list = [round(i*0.1,1) for i in range(2,9)]
    feat_num_list = list(range(10,20,5))
    results = defaultdict(dict)
    gt_correspondences = semantic_map.gt_pred_correspondences.copy()
    """
    for feat_num in feat_num_list:
        for score in scores_list:
            print(f"Processing: feat_num={feat_num}, score={score}")
            edges_list = method3(semantic_map,feat_num,score)
            subgraphs = extract_subgraphs(edges_list)
            true_labels, predicted_labels,instances = extract_labels_from_subgraphs(subgraphs,gt_correspondences)
            results[(feat_num,score)]["ari_score"] = adjusted_rand_score(true_labels, predicted_labels)
            results[(feat_num,score)]["homogeneity"] = homogeneity_score(true_labels, predicted_labels)
            results[(feat_num,score)]["completeness"] = completeness_score(true_labels, predicted_labels)
            results[(feat_num,score)]["predicted_object_num"] = len(subgraphs)
            results[(feat_num,score)]["true_object_num"] = len(np.unique(true_labels))
            results[(feat_num,score)]["instances"] = instances
            
            merged_predicted_labels = merge_predicted_labels_zscore(subgraphs,semantic_map,predicted_labels,feat_num,"chordal_distance",2.8)
            results[(feat_num,score)]["ari_score2"] = adjusted_rand_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["homogeneity2"] = homogeneity_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["completeness2"] = completeness_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["predicted_object_num2"] = len(np.unique(merged_predicted_labels))

            merged_predicted_labels = merge_predicted_labels_zscore(subgraphs,semantic_map,predicted_labels,feat_num,"chordal_distance",3.2)
            results[(feat_num,score)]["ari_score3"] = adjusted_rand_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["homogeneity3"] = homogeneity_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["completeness3"] = completeness_score(true_labels, merged_predicted_labels)
            results[(feat_num,score)]["predicted_object_num3"] = len(np.unique(merged_predicted_labels))

             
    with open('./method3_merge_results.pkl','wb') as f:
        pickle.dump(results,f)
      """
    # edges_list = method3(semantic_map,15,0.2)
    # subgraphs = extract_subgraphs(edges_list)
    # true_labels, predicted_labels,instances = extract_labels_from_subgraphs(subgraphs,gt_correspondences)
    true_labels,predicted_labels = method4(semantic_map,10,"chordal_distance")
    print(f"ARI Score for method4 {adjusted_rand_score(true_labels, predicted_labels)}")
    print(f"Homogeneity Score for method4 {homogeneity_score(true_labels, predicted_labels)}")
    print(f"Completeness Score for method4 {completeness_score(true_labels, predicted_labels)}")
    print(f"Number of predicted objects: {len(np.unique(predicted_labels))}")
    print(f"Number of true objects: {len(np.unique(true_labels))}")
    print(f"Number of total masks:{len(true_labels)}")

    predicted_labels2,valid_masks = method2(semantic_map,15,"chordal_distance",len(np.unique(true_labels)))
    """
    true_labels2 = []
    for img_id, mask_id in valid_masks:
        try:
            gt_label = gt_correspondences[img_id][mask_id][0]
            true_labels2.append(gt_label)
        except KeyError:
            continue

    print(f"ARI Score for method2 with chordal distance {adjusted_rand_score(true_labels2, predicted_labels2)}")
    print(f"Homogeneity Score for method2 with chordal distance {homogeneity_score(true_labels2, predicted_labels2)}")
    print(f"Completeness Score for method2 with chordal distance {completeness_score(true_labels2, predicted_labels2)}")
    """
        
            
 

    

if __name__ == "__main__":
    main()
