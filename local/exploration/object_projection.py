import numpy as np
import random

def wraparound_centroid(mask):
    """
    Calculate the centroid for a 360-degree image considering wraparound effects.
    
    Parameters:
    - mask: Binary mask of foreground objects.
    
    Returns:
    - (centroid_x, centroid_y): The centroid coordinates in 360-degree spherical space.
    """
    # Get the shape of the mask (height, width)
    height, width = mask.shape
    
    # Convert the mask into cylindrical coordinates (wraparound handling)
    # X-coordinates (longitude): range [-π, π], width maps to -π to π
    cylindrical_x = np.linspace(-np.pi, np.pi, width)  # Longitude (-π to π)

    # Find the indices of the foreground points (where mask == 1)
    foreground_indices = np.where(mask == 1)  # Returns a tuple of row and column indices
    
    # Extract the longitudes corresponding to the foreground points
    longitudes = cylindrical_x[foreground_indices[1]]  # We are interested in the column indices (x-axis)

    # Convert longitudes to x (cos) and y (sin) components for spherical averaging
    x_coords = np.cos(longitudes)
    y_coords = np.sin(longitudes)
    
    # Compute the centroid in the x and y components (circular averaging)
    centroid_x = np.arctan2(np.sum(y_coords), np.sum(x_coords))
    

    # Convert the centroid to image coordinates
    centroid_x = int((centroid_x + np.pi) / (2 * np.pi) * width)
    centroid_y = np.mean(np.where(mask == 1)[0])

    return np.array([[centroid_x, centroid_y]])


def least_squares_intersection(points, directions):
    """
    Computes the least squares intersection point of multiple 2D lines.

    Args:
    points (numpy.ndarray): Nx2 array of starting points.
    directions (numpy.ndarray): Nx2 array of direction vectors.

    Returns:
    numpy.ndarray: The least squares intersection point (2D).
    """
    N = len(points)
    
    # Construct the least squares system
    A = np.zeros((N, 2))  # Coefficient matrix
    b = np.zeros(N)        # Right-hand side vector
    
    for i in range(N):
        x_i, y_i = points[i]
        dx, dy = directions[i]

        # Normal vector to the direction (perpendicular)
        normal = np.array([-dy, dx])  

        # Equation: normal • (X - Pi) = 0
        A[i, :] = normal
        b[i] = np.dot(normal, points[i])  

    # Solve the least squares problem using pseudo-inverse
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    
    return X

def angle_difference(v, target_direction):
    """Compute the angular difference (in radians) between two vectors."""
    v = v / np.linalg.norm(v)
    target_direction = target_direction / np.linalg.norm(target_direction)
    cos_angle = np.clip(np.dot(v, target_direction), -1.0, 1.0)
    return np.arccos(cos_angle)


def intersect_two_lines(p1, v1, p2, v2):
    """
    Compute the intersection point of two lines in 2D.

    Each line is defined as:
        Line 1: p1 + t * v1
        Line 2: p2 + s * v2

    Args:
        p1, v1: Point and direction vector for the first line.
        p2, v2: Point and direction vector for the second line.

    Returns:
        intersection point as a NumPy array, or None if lines are parallel.
    """
    # Perpendicular to v2
    perp = np.array([-v2[1], v2[0]])

    # Check if lines are parallel
    denom = np.dot(perp, v1)
    if np.abs(denom) < 1e-8:
        return None  # Lines are parallel or coincident

    # Solve for t such that p1 + t*v1 intersects Line 2
    t = np.dot(perp, p2 - p1) / denom
    intersection = p1 + t * v1
    return intersection

def optimize_intersection_l2(inliers):
    """
    Refine the intersection point using least-squares minimization
    over the inlier lines defined by (p, v).
    """
    A = []
    b = []

    for p, v in inliers:
        # Normalize direction
        v = v / np.linalg.norm(v)
        # Compute projection matrix to the orthogonal complement of v
        I = np.eye(2)
        P = I - np.outer(v, v)
        A.append(P)
        b.append(P @ p)

    A = np.concatenate(A)
    b = np.concatenate(b)

    # Solve Ax = b in least-squares sense
    refined_intersection = np.linalg.lstsq(A, b, rcond=None)[0]
    return refined_intersection

def ransac_intersection(vectors, angle_threshold=np.radians(5), iterations=100):
    """
    Estimate the intersection point of multiple directional vectors using RANSAC,
    based on angular (radial) alignment.

    Args:
        vectors: List of tuples (p, v) where `p` is a point and `v` is a direction vector.
        angle_threshold: Max allowed angle difference (in radians) to be considered an inlier.
        iterations: Number of RANSAC iterations.

    Returns:
        best_intersection: Estimated intersection point.
        best_inliers: List of inlier (p, v) tuples.
        best_pair: Tuple of indices of the vector pair used to compute best_intersection.
    """
    best_intersection = None
    best_inliers = []
    best_error = float('inf')
    best_pair = None

    for _ in range(iterations):
        i1, i2 = random.sample(range(len(vectors)), 2)
        p1, v1 = vectors[i1]
        p2, v2 = vectors[i2]

        intersection = intersect_two_lines(p1,v1,p2,v2)
        if intersection is None:
            continue

        inliers = []
        total_error = 0.0
        for (p, v) in vectors:
            to_intersection = intersection - p
            if np.dot(to_intersection, v) <= 0:
                continue

            angle_diff = angle_difference(v, to_intersection)
            if angle_diff < angle_threshold:
                inliers.append((p, v))
                total_error += angle_diff

        if len(inliers) > len(best_inliers):
            best_intersection = intersection
            best_inliers = inliers
            best_error = total_error
            best_pair = (i1, i2)
        elif len(inliers) == len(best_inliers) and total_error < best_error:
            best_intersection = intersection
            best_inliers = inliers
            best_error = total_error
            best_pair = (i1, i2)
    best_intersection = optimize_intersection_l2(best_inliers)
    return best_intersection, best_inliers, best_pair
