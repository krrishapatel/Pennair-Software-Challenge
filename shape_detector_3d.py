#!/usr/bin/env python3
"""
PennAiR Software Team Challenge 2024
Part 4: 3D Coordinate Estimation

This script extends the shape detection algorithm to estimate 3D coordinates
(depth, x, y) of shape centers with respect to the camera using the provided
camera intrinsic matrix and known circle radius.
"""

import cv2
import numpy as np
import argparse
import os

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

def estimate_3d_coordinates(center_2d, camera_matrix, contour):
    """
    Estimate 3D coordinates using camera intrinsics and known circle radius
    
    Args:
        center_2d: 2D center coordinates (u, v) in pixels
        camera_matrix: Camera intrinsic matrix K
        contour: Shape contour for area calculation
    
    Returns:
        Tuple of (X, Y, Z) in meters relative to camera
    """
    # Camera intrinsic matrix from the challenge
    K = np.array([[2564.3186869, 0, 0],
                  [0, 2569.70273111, 0],
                  [0, 0, 1]])
    
    # Known circle radius: 10 inches = 0.254 meters
    real_radius_meters = 10 * 0.0254
    
    # Estimate pixel radius from contour area
    area = cv2.contourArea(contour)
    pixel_radius = np.sqrt(area / np.pi)
    
    # Calculate depth using similar triangles: Z = (f * R) / r
    # where f = focal length, R = real radius, r = pixel radius
    focal_length = K[0, 0]  # Use fx as focal length
    depth = (focal_length * real_radius_meters) / pixel_radius
    
    # Calculate X and Y in camera coordinates
    # (u - cx) / fx * Z = X
    # (v - cy) / fy * Z = Y
    cx = K[0, 2]  # Principal point x
    cy = K[1, 2]  # Principal point y
    fx = K[0, 0]  # Focal length x
    fy = K[1, 1]  # Focal length y
    
    u, v = center_2d
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    return X, Y, Z

def draw_3d_results(image, contours, centers, camera_matrix):
    """
    Draw results with 3D coordinate information
    """
    result = image.copy()
    
    # Draw contour outlines in green
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Draw centers and 3D information
    for i, (contour, center) in enumerate(zip(contours, centers)):
        # Estimate 3D coordinates
        X, Y, Z = estimate_3d_coordinates(center, camera_matrix, contour)
        
        # Draw center point
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        cv2.circle(result, center, 8, (0, 0, 255), 2)
        
        # Draw 3D coordinate information
        coord_text = f"Shape {i+1}: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m"
        cv2.putText(result, coord_text, 
                   (center[0] + 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw depth information
        depth_text = f"Depth: {Z:.3f}m"
        cv2.putText(result, depth_text, 
                   (center[0] + 10, center[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw pixel radius information
        area = cv2.contourArea(contour)
        pixel_radius = np.sqrt(area / np.pi)
        radius_text = f"Pixel Radius: {pixel_radius:.1f}"
        cv2.putText(result, radius_text, 
                   (center[0] + 10, center[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result

def process_video_3d(video_path, output_path=None):
    """
    Process video with 3D coordinate estimation
    """
    print(f"=== Part 4: 3D Coordinate Estimation ===")
    print(f"Processing video: {video_path}")
    
    # Camera intrinsic matrix from the challenge
    camera_matrix = np.array([[2564.3186869, 0, 0],
                             [0, 2569.70273111, 0],
                             [0, 0, 1]])
    
    print("Camera Matrix K:")
    print(camera_matrix)
    print("Known circle radius: 10 inches = 0.254 meters")
    print("Assumption: Flat surface")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps")
    print("Estimating 3D coordinates for each detected shape...")
    
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
        
        # Detect shapes in current frame
        contours = detect_shapes(frame)
        centers = find_centers(contours)
        
        # Draw 3D results
        result_frame = draw_3d_results(frame, contours, centers, camera_matrix)
        
        # Write frame if output video requested
        if writer:
            writer.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('3D Shape Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"3D coordinate estimation complete!")
    print(f"Processed {frame_count} frames")
    print(f"Results saved to: {output_path}" if output_path else "No output file specified")

def main():
    """Main function to run 3D coordinate estimation"""
    parser = argparse.ArgumentParser(description='PennAiR Challenge Part 4: 3D Coordinate Estimation')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Output path for processed video')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        process_video_3d(args.video, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
