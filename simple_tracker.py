#!/usr/bin/env python3
"""
PennAiR Software Team Challenge 2024
Part 6: Object Tracking Enhancement

This script implements basic object tracking between frames to maintain
consistent shape identification throughout video processing.
"""

import cv2
import numpy as np
import argparse
import os

class SimpleTracker:
    """
    Basic object tracker using distance-based matching between frames
    """
    
    def __init__(self, max_distance=100):
        self.max_distance = max_distance
        self.tracked_objects = {}  # id -> (center, frame_count, contour)
        self.next_id = 1
    
    def update(self, contours, centers):
        """
        Update tracker with new detections
        
        Args:
            contours: List of detected contours
            centers: List of detected centers
            
        Returns:
            List of track IDs
        """
        if not self.tracked_objects:
            # First frame - assign IDs to all detections
            for contour, center in zip(contours, centers):
                self.tracked_objects[self.next_id] = (center, 0, contour)
                self.next_id += 1
            return list(self.tracked_objects.keys())
        
        # Find closest matches for existing tracks
        matched_centers = set()
        updated_tracks = {}
        
        for track_id, (old_center, frame_count, old_contour) in self.tracked_objects.items():
            best_match_idx = None
            best_distance = float('inf')
            
            for i, center in enumerate(centers):
                if center in matched_centers:
                    continue
                
                # Calculate Euclidean distance between centers
                distance = np.sqrt((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                # Update existing track
                center = centers[best_match_idx]
                contour = contours[best_match_idx]
                updated_tracks[track_id] = (center, frame_count + 1, contour)
                matched_centers.add(center)
            else:
                # Track lost - keep for a few frames
                if frame_count < 5:
                    updated_tracks[track_id] = (old_center, frame_count + 1, old_contour)
        
        # Add new tracks for unmatched detections
        for i, (contour, center) in enumerate(zip(contours, centers)):
            if center not in matched_centers:
                updated_tracks[self.next_id] = (center, 0, contour)
                self.next_id += 1
        
        self.tracked_objects = updated_tracks
        return list(self.tracked_objects.keys())
    
    def get_tracked_objects(self):
        """
        Get current tracked objects
        
        Returns:
            Dictionary of track_id -> (center, frame_count, contour)
        """
        return self.tracked_objects

def detect_shapes(image):
    """
    Detect shapes using the same algorithm from Parts 1-2
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range for grass background
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green (grass) background
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert mask to get non-grass areas (shapes)
    shape_mask = cv2.bitwise_not(grass_mask)
    
    # Clean up mask using morphological operations
    kernel = np.ones((5,5), np.uint8)
    shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, kernel)
    shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the shape mask
    contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    min_area = 1000
    max_area = 50000
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    return valid_contours

def find_centers(contours):
    """
    Calculate center points of contours
    """
    centers = []
    for contour in contours:
        # Calculate moments to find centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        else:
            # Fallback: use bounding rectangle center
            x, y, w, h = cv2.boundingRect(contour)
            centers.append((x + w//2, y + h//2))
    
    return centers

def draw_tracked_results(image, contours, centers, tracker):
    """
    Draw results with tracking information
    """
    result = image.copy()
    
    # Draw contour outlines in green
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Draw centers and tracking information
    tracked_objects = tracker.get_tracked_objects()
    
    for track_id, (center, frame_count, contour) in tracked_objects.items():
        if center in centers:
            # Draw center point
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            cv2.circle(result, center, 8, (0, 0, 255), 2)
            
            # Draw tracking ID
            cv2.putText(result, f"ID {track_id}", 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw frame count (tracking duration)
            cv2.putText(result, f"Frames: {frame_count}", 
                       (center[0] + 10, center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw area information
            area = cv2.contourArea(contour)
            cv2.putText(result, f"Area: {area:.0f}", 
                       (center[0] + 10, center[1] + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result

def process_video_with_tracking(video_path, output_path=None):
    """
    Process video with object tracking enhancement
    """
    print(f"=== Part 6: Object Tracking Enhancement ===")
    print(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps")
    print("Using distance-based object tracking between frames...")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracker
    tracker = SimpleTracker(max_distance=100)
    
    frame_count = 0
    print("Processing video frames with tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames")
        
        # Detect shapes in current frame
        contours = detect_shapes(frame)
        centers = find_centers(contours)
        
        # Update tracker with new detections
        track_ids = tracker.update(contours, centers)
        
        # Draw results with tracking information
        result_frame = draw_tracked_results(frame, contours, centers, tracker)
        
        # Write frame if output video requested
        if writer:
            writer.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('Object Tracking Demo', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Object tracking processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Results saved to: {output_path}" if output_path else "No output file specified")
    
    # Print final tracking statistics
    tracked_objects = tracker.get_tracked_objects()
    print(f"Final tracking results: {len(tracked_objects)} objects tracked")
    for track_id, (center, frame_count, contour) in tracked_objects.items():
        print(f"  Track {track_id}: Center at {center}, Tracked for {frame_count} frames")

def main():
    """Main function to run object tracking enhancement"""
    parser = argparse.ArgumentParser(description='PennAiR Challenge Part 6: Object Tracking Enhancement')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        process_video_with_tracking(args.video, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
