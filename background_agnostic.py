#!/usr/bin/env python3
"""
PennAiR Software Team Challenge 2024
Part 3: Background Agnostic Shape Detection

This script implements background agnostic detection using adaptive thresholding
to work with various background colors and textures.
"""

import cv2
import numpy as np
import argparse
import os

def detect_shapes_agnostic(image):
    """
    Background agnostic shape detection using adaptive thresholding
    """
    # Convert to grayscale for adaptive thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding (works on any background)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Clean up thresholded image using morphological operations
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def draw_results(image, contours, centers):
    """
    Draw detection results on image
    """
    result = image.copy()
    
    # Draw contour outlines in green
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Draw center points as red circles
    for i, center in enumerate(centers):
        # Draw center point
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        cv2.circle(result, center, 8, (0, 0, 255), 2)
        
        # Add shape label
        cv2.putText(result, f"Shape {i+1}", 
                   (center[0] + 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add area information
        area = cv2.contourArea(contours[i])
        cv2.putText(result, f"Area: {area:.0f}", 
                   (center[0] + 10, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def process_video_agnostic(video_path, output_path=None):
    """
    Process video with background agnostic detection
    """
    print(f"=== Part 3: Background Agnostic Detection ===")
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
    print("Using adaptive thresholding for background agnostic detection...")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames")
        
        # Apply background agnostic detection to current frame
        contours = detect_shapes_agnostic(frame)
        centers = find_centers(contours)
        
        # Draw results on frame
        result_frame = draw_results(frame, contours, centers)
        
        # Write frame if output video requested
        if writer:
            writer.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('Background Agnostic Processing', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Background agnostic processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Results saved to: {output_path}" if output_path else "No output file specified")

def main():
    """Main function to run background agnostic detection"""
    parser = argparse.ArgumentParser(description='PennAiR Challenge Part 3: Background Agnostic Detection')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        process_video_agnostic(args.video, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
