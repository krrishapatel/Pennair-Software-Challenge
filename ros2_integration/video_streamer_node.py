#!/usr/bin/env python3
"""
PennAiR Challenge Part 5: Video Streamer Node
This node streams input video as images on a ROS2 topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import argparse
import os

class VideoStreamerNode(Node):
    """
    Node that streams video frames as ROS2 Image messages
    """
    
    def __init__(self, video_path):
        super().__init__('video_streamer_node')
        
        # Publisher for video frames
        self.image_pub = self.create_publisher(Image, 'camera_image', 10)
        
        # CV bridge for converting between OpenCV and ROS images
        self.bridge = CvBridge()
        
        # Video path
        self.video_path = video_path
        
        # Timer for publishing frames
        self.timer = self.create_timer(0.033, self.publish_frame)  # ~30 fps
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.get_logger().info(f'Video streamer started: {self.width}x{self.height} @ {self.fps} fps')
        self.get_logger().info(f'Publishing frames on topic: camera_image')
    
    def publish_frame(self):
        """Publish current video frame"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert OpenCV image to ROS message
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                
                # Publish frame
                self.image_pub.publish(ros_image)
                
                # Log every 30 frames
                frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_number % 30 == 0:
                    self.get_logger().info(f'Published frame {frame_number}')
            else:
                # Video ended, restart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.get_logger().info('Video ended, restarting...')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    # Get video path from command line argument
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 video_streamer_node.py <video_path>")
        return 1
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return 1
    
    # Create and run node
    node = VideoStreamerNode(video_path)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    return 0

if __name__ == '__main__':
    exit(main())
