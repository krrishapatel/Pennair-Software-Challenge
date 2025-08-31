#!/usr/bin/env python3
"""
PennAiR Challenge Part 5: Shape Detection Node
This node runs the shape detection algorithm on incoming images and publishes results
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge

class ShapeDetectorNode(Node):
    """
    Node that detects shapes in images and publishes detection results
    """
    
    def __init__(self):
        super().__init__('shape_detector_node')
        
        # Subscriber for incoming images
        self.image_sub = self.create_subscription(Image, 'camera_image', self.image_callback, 10)
        
        # Publishers for detection results
        self.detection_pub = self.create_publisher(String, 'shape_detections', 10)
        self.processed_image_pub = self.create_publisher(Image, 'processed_image', 10)
        
        # CV bridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        self.get_logger().info('Shape detector node started!')
        self.get_logger().info('Subscribing to: camera_image')
        self.get_logger().info('Publishing to: shape_detections, processed_image')
    
    def detect_shapes(self, image):
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
    
    def find_centers(self, contours):
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
    
    def draw_results(self, image, contours, centers):
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
    
    def image_callback(self, msg):
        """
        Process incoming image and publish detection results
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detect shapes using our algorithm
            contours = self.detect_shapes(cv_image)
            centers = self.find_centers(contours)
            
            # Draw results on image
            result_image = self.draw_results(cv_image, contours, centers)
            
            # Publish detection message with positions and outlines
            detection_msg = f"Detected {len(contours)} shapes at positions: {centers}"
            self.detection_pub.publish(String(data=detection_msg))
            
            # Publish processed image with detections drawn
            ros_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.processed_image_pub.publish(ros_image)
            
            # Log detection results
            self.get_logger().info(f'Published detection: {detection_msg}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    node = ShapeDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
