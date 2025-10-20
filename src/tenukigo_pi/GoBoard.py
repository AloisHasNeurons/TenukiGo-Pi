import math
import copy
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Assumes a 'utils_.py' file exists in the same directory
# This file must contain all the following helper functions:
# get_corners, detect_lines, removeDuplicates, restore_and_remove_lines,
# add_lines_in_the_edges, get_key_points, detect_intersections,
# map_intersections
from .utils_ import (
    get_corners,
    detect_lines,
    removeDuplicates,
    restore_and_remove_lines,
    add_lines_in_the_edges,
    get_key_points,
    detect_intersections,
    map_intersections
)


class GoBoard:
    """
    Manages board detection and state extraction from a single camera frame.

    This class uses a YOLO model to find the board, applies perspective
    correction, detects grid lines and stones, and maps stones to their
    correct 19x19 intersection.

    Attributes:
        model (YOLO): The loaded YOLO object detection model.
        frame (np.array): The last raw frame received.
        transformed_image (np.array): The frame after perspective warping.
        annotated_frame (np.array): The raw frame with YOLO detections drawn.
        state (np.array): A 19x19x2 numpy array representing the board state.
                          Channel 0 = black stones, Channel 1 = white stones.
        padding (int): Padding used during perspective transformation.
    """

    def __init__(self, model_path):
        """
        Initializes the GoBoard detector.

        Args:
            model_path (str): File path to the YOLO model (e.g., 'model.pt').
        """
        self.model = YOLO(model_path)
        self.frame = None
        self.transformed_image = None
        self.annotated_frame = None
        self.state = None
        self.padding = 30
        self.perspective_matrix = None
        self.map = None

    def state_to_array(self):
        """
        Converts the internal 19x19x2 state into a simple 19x19 array.

        Returns:
            np.array: A 19x19 array where 0=empty, 1=black, 2=white.

        Raises:
            ValueError: If the state has not been set by processing a frame.
        """
        if self.state is None:
            raise ValueError(
                "The board state is not set. Process a frame first."
            )

        # Create a 2D array with the same shape as the board (19x19)
        board_array = np.zeros((19, 19), dtype=int)

        # Assign 1 for black stones and 2 for white stones
        board_array[self.state[:, :, 0] == 1] = 1
        board_array[self.state[:, :, 1] == 1] = 2

        return board_array

    def get_state(self):
        """
        Get a deep copy of the current 19x19x2 board state.

        Returns:
            np.array: A deep copy of the current board state.
        """
        return copy.deepcopy(self.state)

    def apply_perspective_transformation(self, double_transform=False):
        """
        Warps the input frame to get a flat, top-down view of the board.

        Args:
            double_transform (bool): If True, applies the transform twice,
                                     once to find the board and a second
                                     time to get the final clean image.
        """
        if double_transform:
            # Extract corners from the detection results
            input_points = get_corners(self.results, self.padding)

            # Define output points for perspective transformation
            output_edge = 600 + self.padding * 2
            out_pts = np.array([[0, 0], [output_edge, 0],
                                [output_edge, output_edge], [0, output_edge]],
                               dtype=np.float32)

            # Perform perspective transformation on the frame
            perspective_matrix = cv2.getPerspectiveTransform(input_points,
                                                             out_pts)
            first_transformed_image = cv2.warpPerspective(self.frame,
                                                          perspective_matrix,
                                                          (output_edge,
                                                           output_edge))
            self.results = self.model(first_transformed_image, verbose=False)
        else:
            first_transformed_image = self.frame

        self.annotated_frame = self.results[0].plot(labels=False, conf=False)

        # Save the annotated frame to a file
        filename = f"debug_frame_{time.time()}.jpg"
        cv2.imwrite(filename, self.annotated_frame)
        print(f"Saved debug frame: {filename}")

        # Extract corners from the detection results
        input_points = get_corners(self.results, 0)

        # Define output points for perspective transformation
        output_edge = 600
        out_pts = np.array([[0, 0], [output_edge, 0],
                            [output_edge, output_edge], [0, output_edge]],
                           dtype=np.float32)

        # Perform perspective transformation on the frame
        self.perspective_matrix = cv2.getPerspectiveTransform(input_points,
                                                              out_pts)
        self.transformed_image = cv2.warpPerspective(first_transformed_image,
                                                     self.perspective_matrix,
                                                     (output_edge,
                                                      output_edge))

    def assign_stones(self, white_stones_transf, black_stones_transf,
                      transformed_intersections):
        """
        Assigns detected stones to the nearest grid intersection.

        Args:
            white_stones_transf (np.array): Transformed coords of white stones.
            black_stones_transf (np.array): Transformed coords of black stones.
            transformed_intersections (np.array): Coords of grid intersections.
        """

        self.map = map_intersections(transformed_intersections)
        self.state = np.zeros((19, 19, 2))

        for stone in white_stones_transf:
            nearest_corner = self.find_nearest_corner(
                transformed_intersections, stone
            )
            # Use the map to get (row, col) from (x, y) coordinates
            row = self.map[nearest_corner][1]
            col = self.map[nearest_corner][0]
            self.state[row, col, 1] = 1  # Channel 1 is for White

        for stone in black_stones_transf:
            nearest_corner = self.find_nearest_corner(
                transformed_intersections, stone
            )
            # Use the map to get (row, col) from (x, y) coordinates
            row = self.map[nearest_corner][1]
            col = self.map[nearest_corner][0]
            self.state[row, col, 0] = 1  # Channel 0 is for Black

    def find_nearest_corner(self, transformed_intersections, stone):
        """
        Finds the closest intersection to a given stone.

        Args:
            transformed_intersections (list): List of (x, y) intersection
                                              points.
            stone (tuple): (x, y) coordinates of the stone.

        Returns:
            tuple: The (x, y) coordinates of the nearest intersection.
        """
        nearest_corner = None
        closest_distance = float('inf')

        # Iterate through intersections to find the nearest one
        for inter in transformed_intersections:
            distance = math.dist(inter, stone)
            if distance < closest_distance:
                nearest_corner = tuple(inter)
                closest_distance = distance

        return nearest_corner

    def process_frame(self, frame):
        """
        Runs the full detection pipeline on a single frame.

        Args:
            frame (np.array): Input frame from the camera.
        """
        self.frame = frame
        self.results = self.model(self.frame, verbose=False, conf=0.15)
        self.apply_perspective_transformation(double_transform=False)

        # Detect vertical and horizontal lines
        vertical_lines, horizontal_lines = detect_lines(
            self.results, self.perspective_matrix
        )

        # Clean up detected lines
        vertical_lines = removeDuplicates(vertical_lines)
        horizontal_lines = removeDuplicates(horizontal_lines)
        vertical_lines = restore_and_remove_lines(vertical_lines)
        horizontal_lines = restore_and_remove_lines(horizontal_lines)
        vertical_lines = add_lines_in_the_edges(vertical_lines, "vertical")
        horizontal_lines = add_lines_in_the_edges(horizontal_lines, "horizontal")
        vertical_lines = removeDuplicates(vertical_lines)
        horizontal_lines = removeDuplicates(horizontal_lines)

        # Get stone key points
        black_stones = get_key_points(self.results, 0, self.perspective_matrix)
        white_stones = get_key_points(self.results, 6, self.perspective_matrix)

        # Filter lines to be within the 600x600 transformed image
        # CRITICAL FIX: Ensure we're working with numpy arrays
        vertical_lines = np.array(vertical_lines)
        horizontal_lines = np.array(horizontal_lines)

        # Create boolean masks for filtering
        v_lines_le_600 = (vertical_lines <= 600).all(axis=1)
        v_lines_ge_0 = (vertical_lines >= 0).all(axis=1)
        h_lines_le_600 = (horizontal_lines <= 600).all(axis=1)
        h_lines_ge_0 = (horizontal_lines >= 0).all(axis=1)

        cluster_1 = vertical_lines[v_lines_le_600 & v_lines_ge_0]
        cluster_2 = horizontal_lines[h_lines_le_600 & h_lines_ge_0]

        if len(cluster_1) != 19 or len(cluster_2) != 19:
            raise Exception(
                f"Incorrect number of lines detected: {len(cluster_1)} "
                f"vertical, {len(cluster_2)} horizontal"
            )

        # Detect intersections
        intersections = detect_intersections(cluster_1, cluster_2,
                                             self.transformed_image)

        if len(intersections) == 0:
            raise Exception("No intersections were found!")
        if len(intersections) != 361:
            print(f"Warning: Only {len(intersections)}/361 intersections "
                  "found.")

        self.assign_stones(white_stones, black_stones, intersections)
