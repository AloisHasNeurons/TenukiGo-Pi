"""
Motion-triggered processing for Go board analysis.
Detects when something changes on the board and triggers analysis.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detects motion in the board region to trigger analysis."""

    def __init__(self,
                 threshold: int = 25,
                 min_changed_pixels: int = 100,
                 stability_frames: int = 5):
        """
        Args:
            threshold: Pixel difference threshold for motion detection
            min_changed_pixels: Minimum pixels changed to trigger
            stability_frames: Frames to wait after motion stops before analyzing
        """
        self.threshold = threshold
        self.min_changed_pixels = min_changed_pixels
        self.stability_frames = stability_frames

        self.previous_frame = None
        self.board_region = None
        self.frames_stable = 0
        self.motion_detected = False

    def set_board_region(self, corners: np.ndarray, padding: int = 50):
        """
        Set the region of interest (board area) from detected corners.

        Args:
            corners: 4 corner points [(x,y), (x,y), (x,y), (x,y)]
            padding: Extra pixels around board to monitor
        """
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        x_min = max(0, int(x_coords.min()) - padding)
        x_max = int(x_coords.max()) + padding
        y_min = max(0, int(y_coords.min()) - padding)
        y_max = int(y_coords.max()) + padding

        self.board_region = (x_min, y_min, x_max, y_max)
        logger.info(f"Board region set to: {self.board_region}")

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Detect if there's motion in the frame and if board is stable.

        Args:
            frame: Current video frame (BGR)

        Returns:
            tuple: (should_analyze, motion_currently_detected)
                - should_analyze: True if board is stable after motion
                - motion_currently_detected: True if motion is happening now
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop to board region if set
        if self.board_region is not None:
            x1, y1, x2, y2 = self.board_region
            gray = gray[y1:y2, x1:x2]

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # First frame - just store it
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, False

        # Compute absolute difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Count changed pixels
        changed_pixels = np.sum(thresh > 0)

        # Update previous frame
        self.previous_frame = gray

        # Detect motion
        motion_now = changed_pixels > self.min_changed_pixels

        if motion_now:
            self.motion_detected = True
            self.frames_stable = 0
            return False, True  # Motion detected, don't analyze yet
        else:
            # No motion in this frame
            if self.motion_detected:
                # We had motion before, count stable frames
                self.frames_stable += 1

                if self.frames_stable >= self.stability_frames:
                    # Board is stable after motion - analyze now!
                    self.motion_detected = False
                    self.frames_stable = 0
                    logger.info("Board stable after motion - triggering analysis")
                    return True, False
                else:
                    # Still waiting for stability
                    return False, False
            else:
                # No motion, and we weren't tracking any
                return False, False
