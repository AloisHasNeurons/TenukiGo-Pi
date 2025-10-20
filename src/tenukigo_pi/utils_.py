"""
Fixed version of utils_.py

Key fixes:
1. Renamed remove_duplicates to removeDuplicates for consistency
2. Fixed are_similar to properly handle numpy array comparisons
3. Added explicit type conversions where needed
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def line_equation(x1, y1, x2, y2):
    """
    Calculates the slope and intercept (y = mx + b) for a line.

    Args:
        x1 (float): Start point x-coordinate.
        y1 (float): Start point y-coordinate.
        x2 (float): End point x-coordinate.
        y2 (float): End point y-coordinate.

    Returns:
        tuple: (slope, intercept)
               For vertical lines, slope is float('Inf') and intercept is x.
    """
    if x1 == x2:
        # Vertical line: y = x1
        slope = float('Inf')
        b = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        b = y1 - slope * x1
    return slope, b


def normalize_line_direction(lines):
    """
    Sorts the endpoints of each line so that (x1, y1) is always
    the "top-left-most" point.

    Args:
        lines (list): A list of lines, each as [x1, y1, x2, y2].

    Returns:
        list: The list of lines with normalized endpoints.
    """
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]
        if (x1 + y1) > (x2 + y2):
            # Swap endpoints
            lines[i] = [x2, y2, x1, y1]
    return lines


def are_similar(line1, line2, threshold=10):
    """
    Checks if two lines are similar based on a distance threshold.

    Args:
        line1 (np.array or tuple): First line [x1, y1, x2, y2].
        line2 (np.array or tuple): Second line [x1, y1, x2, y2].
        threshold (int, optional): Maximum allowed pixel distance.
                                   Defaults to 10.

    Returns:
        bool: True if all endpoints are within the threshold.
    """
    # Convert to numpy arrays if needed
    line1 = np.array(line1)
    line2 = np.array(line2)
    return np.all(np.abs(line1 - line2) <= threshold)


def removeDuplicates(lines):
    """
    Groups similar lines and averages them to remove duplicates.

    Args:
        lines (list or np.array): List of lines [x1, y1, x2, y2] to be filtered.

    Returns:
        np.array: A filtered array of averaged lines.
    """
    if len(lines) == 0:
        return np.array([])

    # Convert to list if needed
    lines_list = lines.tolist() if isinstance(lines, np.ndarray) else list(lines)

    grouped_lines = {}
    for line in lines_list:
        x1, y1, x2, y2 = line
        found = False
        for key in list(grouped_lines.keys()):  # Convert to list to avoid dict size change during iteration
            if are_similar(np.array(key), np.array(line)):
                grouped_lines[key] = grouped_lines[key] + [line]
                found = True
                break
        if not found:
            grouped_lines[(x1, y1, x2, y2)] = [line]

    final_lines = []
    for key in grouped_lines.keys():
        final_lines.append(np.mean(grouped_lines[key], axis=0))

    return np.array(final_lines).astype(int)


def is_vertical(x1, y1, x2, y2):
    """
    Checks if a line is (mostly) vertical.

    Args:
        x1 (float): Start point x-coordinate.
        y1 (float): Start point y-coordinate.
        x2 (float): End point x-coordinate.
        y2 (float): End point y-coordinate.

    Returns:
        bool: True if the line is vertical.
    """
    return abs(x1 - x2) < 50 and abs(y1 - y2) > 50


def intersect(line1, line2):
    """
    Finds the (x, y) intersection point of two lines.

    Args:
        line1 (list): [x1, y1, x2, y2] for the first line.
        line2 (list): [x1, y1, x2, y2] for the second line.

    Returns:
        np.array: [x, y] coordinates of the intersection.
    """
    slope1, b1 = line_equation(*line1)
    slope2, b2 = line_equation(*line2)
    if slope1 == float('Inf'):
        # Line 1 is vertical
        x = b1
        y = slope2 * x + b2
    elif slope2 == float('Inf'):
        # Line 2 is vertical
        x = b2
        y = slope1 * x + b1
    else:
        # Standard case
        x = (b2 - b1) / (slope1 - slope2)
        y = slope1 * x + b1
    return np.array([int(np.round(x)), int(np.round(y))])


def map_intersections(intersections, board_size=19):
    """
    Creates a dictionary mapping (x, y) pixel coordinates to (col, row)
    board indices (0-18).

    Args:
        intersections (np.array): Array of (x, y) intersection coordinates.
        board_size (int, optional): Size of the board. Defaults to 19.

    Returns:
        dict: A map where key=(x, y) and value=(col, row).
    """
    # Sort intersections primarily by y-coordinate (row), then x (col)
    sorted_indices = np.lexsort((intersections[:, 0], intersections[:, 1]))
    cleaned_intersections = intersections[sorted_indices]
    cleaned_intersections = cleaned_intersections.tolist()

    board_map = {}
    for j in range(0, board_size):  # j is the row index
        # Get the 19 intersections for this row
        row_points = cleaned_intersections[:board_size]
        cleaned_intersections = cleaned_intersections[board_size:]

        # Sort this row by x-coordinate to get column order
        row_points.sort(key=lambda x: x[0])

        for i in range(board_size):  # i is the column index
            if row_points:
                # Map the (x, y) tuple to its (col, row) index
                board_map[tuple(row_points.pop(0))] = (i, j)

    return board_map


def detect_intersections(cluster_1, cluster_2, image):
    """
    Detects intersection points between two clusters of lines.

    Args:
        cluster_1 (np.array): Array of vertical lines [x1, y1, x2, y2].
        cluster_2 (np.array): Array of horizontal lines [x1, y1, x2, y2].
        image (np.array): Image used for boundary checking.

    Returns:
        np.array: An array of (x, y) intersection points.
    """
    intersections = []

    # Get image dimensions - handle both regular arrays and other types
    if hasattr(image, 'shape'):
        if callable(image.shape):
            img_height, img_width = image.shape()[:2]
        else:
            img_height, img_width = image.shape[:2]
    else:
        raise ValueError(f"Image object has no 'shape' attribute. Type: {type(image)}")

    for v_line in cluster_1:
        for h_line in cluster_2:
            inter = intersect(v_line, h_line)

            # Explicitly get intersection coordinates
            inter_x = int(inter[0])
            inter_y = int(inter[1])

            # Check if x and y are within the valid range [0, width-1] and [0, height-1]
            if (0 <= inter_x < img_width) and (0 <= inter_y < img_height):
                # If valid, add as an integer tuple
                intersections.append((inter_x, inter_y))
            else:  # Optional debug print
                print(f"DEBUG: Intersection {inter} rejected. Image size: {img_width}x{img_height}")

    return np.array(intersections)


def calculate_distances(lines):
    """
    Calculate distances between consecutive lines in a sorted list.

    Args:
        lines (np.array): Array of lines [x1, y1, x2, y2].

    Returns:
        list: A list of distances.
    """
    distances = []
    for i in range(len(lines) - 1):
        dist_start = np.linalg.norm(lines[i + 1][:2] - lines[i][:2])
        dist_end = np.linalg.norm(lines[i + 1][2:] - lines[i][2:])
        distances.append((dist_start + dist_end) / 2)
    return distances


def find_common_distance(distances, target_distance=30):
    """
    Finds the most common grid spacing distance using DBSCAN clustering.

    Args:
        distances (list): List of distances between grid lines.
        target_distance (float, optional): The expected grid spacing.

    Returns:
        tuple: (mean_distance, clustered_distances)
    """
    # Reshape distances into a column vector for sklearn
    distances_reshaped = np.array(distances).reshape((-1, 1))

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1, min_samples=1)
    labels = dbscan.fit_predict(distances_reshaped)

    means = []
    label_indices = []

    # Calculate means for each cluster
    for label in np.unique(labels):
        means.append(np.mean(distances_reshaped[labels == label]))
        label_indices.append(label)

    # Find the cluster mean closest to our target distance (e.g., 30px)
    index = np.argmin(np.abs(np.array(means) - target_distance))
    chosen_label = label_indices[index]

    return means[index], distances_reshaped[labels == chosen_label]


def is_approx_multiple(value, base, threshold):
    """
    Checks if a value is approximately a multiple of a base, +/- a threshold.

    Args:
        value (float): The value to check (e.g., 61.2).
        base (float): The base multiple (e.g., 30).
        threshold (float): The allowed error (e.g., 2.0).

    Returns:
        bool: True if value is approx. a multiple of base.
    """
    if value < base:
        return (base - value) < threshold
    # Checks if value is just under a multiple (e.g., 59.8)
    check_under = abs((value % base) - base) < threshold
    # Checks if value is just over a multiple (e.g., 60.2)
    check_over = abs(value % base) < threshold
    return check_under or check_over


def restore_and_remove_lines(lines, distance_threshold=10):
    """
    Restores missing grid lines and removes spurious lines.

    Iterates through a sorted list of lines, checks the spacing,
    and adds new lines if a gap is an approx. multiple of the
    mean distance. Removes lines that are too close.

    Args:
        lines (np.array): A sorted array of lines [x1, y1, x2, y2].
        distance_threshold (float, optional): Error margin. Defaults to 10.

    Returns:
        np.array: The corrected array of lines.
    """
    if len(lines) == 0:
        return lines

    # Sort by x-coord (for vertical lines) or y-coord (for horizontal)
    lines = np.sort(lines, axis=0)
    distances = calculate_distances(lines)

    if len(distances) <= 1:
        return lines

    mean_distance, _ = find_common_distance(distances)
    restored_lines = []

    i = 0
    while i < len(lines) - 1:
        # Calculate spacing between current line and next line
        dist_start = np.linalg.norm(lines[i + 1][:2] - lines[i][:2])
        dist_end = np.linalg.norm(lines[i + 1][2:] - lines[i][2:])
        spacing = (dist_start + dist_end) / 2

        if is_approx_multiple(spacing, mean_distance, distance_threshold):
            if spacing >= mean_distance:
                # Gap is a multiple of mean_distance, fill it
                num_missing_lines = round(spacing / mean_distance) - 1
                for j in range(1, num_missing_lines + 1):
                    if is_vertical(*lines[i]):
                        x1 = lines[i][0] + j * mean_distance
                        y1 = lines[i][1]
                        x2 = lines[i][2] + j * mean_distance
                        y2 = lines[i][3]
                    else:
                        x1 = lines[i][0]
                        y1 = lines[i][1] + j * mean_distance
                        x2 = lines[i][2]
                        y2 = lines[i][3] + j * mean_distance
                    restored_lines.append([x1, y1, x2, y2])
        else:
            # Spacing is invalid, remove the next line
            lines = np.delete(lines, i + 1, axis=0)
            i -= 1  # Re-check from the same index
        i += 1

    if restored_lines:
        lines = np.append(lines, np.array(restored_lines, dtype=int), axis=0)

    # Re-sort the final list
    lines = np.sort(lines, axis=0)
    return lines


def non_max_suppression(boxes, overlap_thresh=0.5):
    """
    Applies non-maximum suppression (NMS) to remove redundant bounding boxes.

    Args:
        boxes (np.array): Array of boxes [x1, y1, x2, y2].
        overlap_thresh (float, optional): Overlap threshold to merge boxes.

    Returns:
        np.array: The filtered array of boxes.
    """
    if len(boxes) == 0:
        return np.array([])

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find coordinates of intersection rectangle
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute overlap ratio
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indices that overlap too much
        delete_indices = np.concatenate(([last],
                                         np.where(overlap > overlap_thresh)[0])
                                        )
        idxs = np.delete(idxs, delete_indices)

    return boxes[pick].astype("int")


def detect_lines(model_results, perspective_matrix):
    """
    Identifies and clusters all line intersections from model results.

    Args:
        model_results (ultralytics.Results): YOLO detection results.
        perspective_matrix (np.array): Matrix for perspective transform.

    Returns:
        tuple: (cluster_vertical, cluster_horizontal)
               Both are arrays of 19 lines.
    """
    # Get all detected intersection points (corners, edges, inner)
    empty_intersections = get_key_points(model_results, 3, perspective_matrix)
    empty_corner = get_key_points(model_results, 4, perspective_matrix)
    empty_edge = get_key_points(model_results, 5, perspective_matrix)

    arrays = [arr for arr in [empty_intersections, empty_corner, empty_edge]
              if arr.size > 0]

    if not arrays:
        raise Exception("No intersection points detected!")

    all_intersections = np.concatenate(arrays, axis=0)

    # --- Detect Vertical Lines ---
    # Sort by x-coordinate to prepare for clustering
    all_intersections_x = all_intersections[:, 0].reshape((-1, 1))
    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_x)
    cluster_labels = kmeans.labels_
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    sorted_unique_labels = unique_labels[np.argsort(label_counts)[::-1]]

    lines_equations = []
    lines_points_length = []
    cluster_vertical = []

    for label in sorted_unique_labels:
        mask = (cluster_labels == label)
        line_points = all_intersections[mask]

        if len(line_points) > 2:
            # Fit x as a function of y (since it's a vertical line)
            slope, intercept = np.polyfit(line_points[:, 1],
                                          line_points[:, 0], 1)
            # Create a line from y=0 to y=600
            line_ = [intercept, 0, slope * 600 + intercept, 600]
            lines_equations.append([slope, intercept])
            lines_points_length.append(len(line_points))
        else:
            # Not enough points, extrapolate from average of other lines
            if not cluster_vertical:
                raise Exception("BoardDetection: Cannot reconstruct all "
                                "vertical lines.")
            elif len(line_points) < 1:
                raise Exception(f"BoardDetection: Cannot reconstruct vertical "
                                f"line at point {line_points}")
            else:
                x1, y1 = line_points[0]
                slope_avg = np.average(np.array(lines_equations)[:, 0],
                                       weights=lines_points_length, axis=0)
                intercept = x1 - slope_avg * y1
                line_ = [intercept, 0, slope_avg * 600 + intercept, 600]
                lines_equations.append([slope_avg, intercept])
                lines_points_length.append(len(line_points))

        cluster_vertical.append(line_)

    cluster_vertical = normalize_line_direction(cluster_vertical)
    cluster_vertical = np.sort(np.array(cluster_vertical), axis=0).astype(int)

    # --- Detect Horizontal Lines ---
    # Sort by y-coordinate for clustering
    all_intersections_y = all_intersections[:, 1].reshape((-1, 1))
    kmeans = KMeans(n_clusters=19, n_init=10)
    kmeans.fit(all_intersections_y)
    cluster_labels = kmeans.labels_
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    sorted_unique_labels = unique_labels[np.argsort(label_counts)[::-1]]

    lines_equations = []
    lines_points_length = []
    cluster_horizontal = []

    for label in sorted_unique_labels:
        mask = (cluster_labels == label)
        line_points = all_intersections[mask]

        if len(line_points) > 2:
            # Fit y as a function of x (standard horizontal line)
            slope, intercept = np.polyfit(line_points[:, 0],
                                          line_points[:, 1], 1)
            line_ = [0, intercept, 600, slope * 600 + intercept]
            lines_equations.append([slope, intercept])
            lines_points_length.append(len(line_points))
        else:
            # Not enough points, extrapolate
            if not cluster_horizontal:
                raise Exception("BoardDetection: Cannot reconstruct all "
                                "horizontal lines.")
            elif len(line_points) < 1:
                raise Exception(f"BoardDetection: Cannot reconstruct line "
                                f"at point {line_points}")
            else:
                x1, y1 = line_points[0]
                slope_avg = np.average(np.array(lines_equations)[:, 0],
                                       weights=lines_points_length, axis=0)
                intercept = y1 - slope_avg * x1
                line_ = [0, intercept, 600, slope_avg * 600 + intercept]
                lines_equations.append([slope_avg, intercept])
                lines_points_length.append(len(line_points))

        cluster_horizontal.append(line_)

    cluster_horizontal = normalize_line_direction(cluster_horizontal)
    cluster_horizontal = np.sort(np.array(cluster_horizontal),
                                 axis=0).astype(int)

    return (np.array(cluster_vertical).reshape((-1, 4)),
            np.array(cluster_horizontal).reshape((-1, 4)))


def get_corners_inside_box(corners_boxes, board_box):
    """
    Filters a list of boxes to find those with at least one corner
    inside a main bounding box.

    Args:
        corners_boxes (np.array): Array of [x1, y1, x2, y2] boxes.
        board_box (tuple or np.array): A single (x1, y1, x2, y2) bounding box.

    Returns:
        np.array: The filtered array of boxes.
    """
    # Ensure board_box is a numpy array
    board_box = np.array(board_box)
    x1, y1, x2, y2 = board_box

    square_x1 = corners_boxes[:, 0]
    square_y1 = corners_boxes[:, 1]
    square_x2 = corners_boxes[:, 2]
    square_y2 = corners_boxes[:, 3]

    # Check if any corner is inside the board_box
    condition = (
        ((square_x1 >= x1) & (square_x1 <= x2) &
         (square_y1 >= y1) & (square_y1 <= y2)) |
        ((square_x2 >= x1) & (square_x2 <= x2) &
         (square_y1 >= y1) & (square_y1 <= y2)) |
        ((square_x1 >= x1) & (square_x1 <= x2) &
         (square_y2 >= y1) & (square_y2 <= y2)) |
        ((square_x2 >= x1) & (square_x2 <= x2) &
         (square_y2 >= y1) & (square_y2 <= y2))
    )
    return corners_boxes[condition]


def get_corners(results, padding=None):
    """
    Extracts the four corner-centers of the board from YOLO results.

    Args:
        results (ultralytics.Results): YOLO detection results.
        padding (int, optional): Padding to add/subtract from corners.

    Returns:
        np.array: 4x2 array of [x, y] corner coordinates in order:
                  [top-left, top-right, bottom-right, bottom-left].
    """
    # Class 2 == "corner"
    corner_boxes = np.array(results[0].boxes.xyxy[results[0].boxes.cls == 2])

    if len(corner_boxes) < 4:
        raise Exception(f"Incorrect number of corners! "
                        f"Detected {len(corner_boxes)} corners")

    corner_boxes_nms = non_max_suppression(corner_boxes)

    # Class 1 == "board"
    model_board_edges = results[0].boxes.xyxy[results[0].boxes.cls == 1][0]
    corner_boxes = get_corners_inside_box(corner_boxes_nms,
                                          np.array(model_board_edges))

    if len(corner_boxes) != 4:
        raise Exception(f"Incorrect number of corners! Detected "
                        f"{len(corner_boxes)} after NMS/filtering.")

    # Get centers of the corner boxes
    corner_centers = ((corner_boxes[:, [0, 1]] +
                       corner_boxes[:, [2, 3]]) / 2)

    # Sort corners into [top-left, top-right, bottom-right, bottom-left]
    corner_centers = corner_centers[corner_centers[:, 1].argsort()]
    upper = corner_centers[:2][corner_centers[:2][:, 0].argsort()]
    lower = corner_centers[2:][corner_centers[2:][:, 0].argsort()[::-1]]
    corner_centers = np.concatenate((upper, lower)).astype(np.float32)

    if padding is not None:
        corner_centers[0] += np.array([-padding, -padding])  # Top-left
        corner_centers[1] += np.array([padding, -padding])   # Top-right
        corner_centers[2] += np.array([padding, padding])    # Bottom-right
        corner_centers[3] += np.array([-padding, padding])   # Bottom-left

    return corner_centers


def get_key_points(results, class_id, perspective_matrix, output_edge=600):
    """
    Extracts and transforms key points (like stones) from YOLO results.

    Args:
        results (ultralytics.Results): YOLO detection results.
        class_id (int): The class ID to extract (e.g., 0 for black, 6 for
                        white).
        perspective_matrix (np.array): Matrix to warp the points.
        output_edge (int, optional): Max coord value for filtering.

    Returns:
        np.array: Array of (x, y) transformed key points.
    """
    key_points = results[0].boxes.xywh[
        results[0].boxes.cls == class_id
    ].reshape((-1, 4))

    if key_points.size > 0:
        # Get center points (x, y) from (x, y, w, h)
        key_points = np.array(key_points[:, [0, 1]])
        # Apply perspective transform
        key_points_transf = cv2.perspectiveTransform(
            key_points.reshape((1, -1, 2)), perspective_matrix
        ).reshape((-1, 2))
        # Filter points outside the warped 600x600 image
        return key_points_transf[
            (key_points_transf[:, 0:2] >= 0).all(axis=1) &
            (key_points_transf[:, 0:2] <= output_edge).all(axis=1)
        ]

    return np.array(key_points)


def line_distance(line1, line2):
    """
    Calculates the average Euclidean distance between two line segments.

    Args:
        line1 (np.array): [x1, y1, x2, y2]
        line2 (np.array): [x1, y1, x2, y2]

    Returns:
        float: The average distance between endpoints.
    """
    # Ensure both are numpy arrays
    line1 = np.array(line1)
    line2 = np.array(line2)
    dist_start = np.linalg.norm(line1[:2] - line2[:2])
    dist_end = np.linalg.norm(line1[2:] - line2[2:])
    return (dist_start + dist_end) / 2


def average_distance(lines):
    """
    Calculates the average distance between consecutive lines in a list.

    Args:
        lines (list or np.array): A list of line segments.

    Returns:
        float: The average distance.
    """
    if len(lines) < 2:
        return 0.0

    distances = [line_distance(lines[i + 1], lines[i])
                 for i in range(len(lines) - 1)]
    return np.average(distances)


def add_lines_in_the_edges(lines, line_type):
    """
    Adds missing 1st or 19th grid lines if they weren't detected.

    Args:
        lines (np.array): Array of detected lines.
        line_type (str): "vertical" or "horizontal".

    Returns:
        np.array: The array of lines, possibly with new lines added.
    """
    if len(lines) not in [17, 18]:
        # Only try to fix if 1 or 2 lines are missing
        return lines

    mean_distance = average_distance(lines)
    appended = False

    if line_type == "vertical":
        left_border = np.array([0, 0, 0, 600])
        right_border = np.array([600, 0, 600, 600])

        if line_distance(lines[0], left_border) > mean_distance:
            # Missing line on the left
            x1 = lines[0][0] - mean_distance
            y1 = lines[0][1]
            x2 = lines[0][2] - mean_distance
            y2 = lines[0][3]
            lines = np.append(lines, [[x1, y1, x2, y2]], axis=0)
            appended = True
        if line_distance(lines[-1], right_border) > mean_distance:
            # Missing line on the right
            x1 = lines[-1][0] + mean_distance
            y1 = lines[-1][1]
            x2 = lines[-1][2] + mean_distance
            y2 = lines[-1][3]
            lines = np.append(lines, [[x1, y1, x2, y2]], axis=0)
            appended = True

        if appended:
            lines = lines[lines[:, 0].argsort()]  # Re-sort by x
        else:
            print("No missing edges in the vertical lines")

    elif line_type == "horizontal":
        top_border = np.array([0, 0, 600, 0])
        bottom_border = np.array([0, 600, 600, 600])

        if line_distance(lines[0], top_border) > mean_distance:
            # Missing line at the top
            x1 = lines[0][0]
            y1 = lines[0][1] - mean_distance
            x2 = lines[0][2]
            y2 = lines[0][3] - mean_distance
            lines = np.append(lines, [[x1, y1, x2, y2]], axis=0)
            appended = True
        if line_distance(lines[-1], bottom_border) > mean_distance:
            # Missing line at the bottom
            x1 = lines[-1][0]
            y1 = lines[-1][1] + mean_distance
            x2 = lines[-1][2]
            y2 = lines[-1][3] + mean_distance
            lines = np.append(lines, [[x1, y1, x2, y2]], axis=0)
            appended = True

        if appended:
            lines = lines[lines[:, 1].argsort()]  # Re-sort by y
        else:
            print("No missing edges in the horizontal lines")
    else:
        print("Please specify a valid line type ('vertical' or 'horizontal')")

    return lines.astype(int)
