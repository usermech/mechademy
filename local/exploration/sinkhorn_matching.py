import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import ot  # Python Optimal Transport (POT)

# -------------------------------
# Sinkhorn + Cost Computation
# -------------------------------
def normalize_features(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def compute_cost_matrix(A, B):
    return cdist(A, B, metric='cosine')

def compute_sinkhorn(A, B, epsilon=0.1):
    A = normalize_features(A)
    B = normalize_features(B)
    C = compute_cost_matrix(A, B)

    r = np.ones(A.shape[0]) / A.shape[0]
    c = np.ones(B.shape[0]) / B.shape[0]

    P = ot.sinkhorn(r, c, C, reg=epsilon)
    return P, C

# -------------------------------
# Similarity Metric: Low-Cost Mass
# -------------------------------
def compute_low_cost_mass(P, C, threshold=0.4):
    return np.sum(P * (C < threshold))

# -------------------------------
# Main Function: Compute Similarity Matrix
# -------------------------------
def compute_similarity_matrix(collections, epsilon=0.1, threshold=0.4):
    n = len(collections)
    similarity_matrix = np.ones((n, n))

    for i in tqdm(range(n)):
        for j in range(i, n):
            P, C = compute_sinkhorn(collections[i], collections[j], epsilon)
            score = compute_low_cost_mass(P, C, threshold)
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score  # ensure symmetry

    return similarity_matrix

def compute_cross_similarity_matrix(A_collection, B_collection, epsilon=0.1, threshold=0.4):
    n = len(A_collection)
    m = len(B_collection)
    similarity_matrix = np.ones((n, m))

    for i in tqdm(range(n)):
        for j in range(m):
            P, C = compute_sinkhorn(A_collection[i], B_collection[j], epsilon)
            score = compute_low_cost_mass(P, C, threshold)
            similarity_matrix[i, j] = score

    return similarity_matrix