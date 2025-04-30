import numpy as np

def method1():
    pass
def method2():
    pass 

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
