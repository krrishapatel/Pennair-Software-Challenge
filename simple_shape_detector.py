#!/usr/bin/env python3
"""
PennAiR Software Team Challenge 2024
Shape Detection Algorithm - Parts 1 & 2

This script implements:
- Part 1: Shape detection on static image with outline tracing and center marking
- Part 2: Same algorithm applied to video frame-by-frame
"""

import cv2
import numpy as np
import argparse
import os
from typing import List, Tuple, Optional

def detect_shapes(image):
    """
    Detect shapes in image using color-based segmentation
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

def process_image(image_path, output_path=None):
    """
    Process single image - Part 1
    """
    print(f"Processing image: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Detect shapes
    contours = detect_shapes(image)
    centers = find_centers(contours)
    
    # Draw results
    result = draw_results(image, contours, centers)
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Results saved to: {output_path}")
    
    # Print results
    print(f"Detected {len(contours)} shapes")
    for i, center in enumerate(centers):
        area = cv2.contourArea(contours[i])
        print(f"Shape {i+1}: Center at {center}, Area: {area:.0f}")
    
    return result

def process_video(video_path, output_path=None):
    """
    Process video frame-by-frame - Part 2
    """
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
        
        # Apply shape detection algorithm to current frame
        contours = detect_shapes(frame)
        centers = find_centers(contours)
        
        # Draw results on frame
        result_frame = draw_results(frame, contours, centers)
        
        # Write frame if output video requested
        if writer:
            writer.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('Video Processing', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Processed {frame_count} frames.")

def main():
    """Main function to run the shape detector"""
    parser = argparse.ArgumentParser(description='PennAiR Shape Detection Challenge - Parts 1 & 2')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, help='Output path for results')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        if args.image:
            print("=== Part 1: Static Image Processing ===")
            process_image(args.image, args.output)
        elif args.video:
            print("=== Part 2: Video Processing ===")
            process_video(args.video, args.output)
        else:
            print("Please provide either --image or --video argument")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
