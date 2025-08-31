#!/usr/bin/env python3
"""
PennAiR Challenge Part 5: ROS2 Launch File
Launches both the video streamer and shape detector nodes
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch argument for video path
        DeclareLaunchArgument(
            'video_path',
            default_value='',
            description='Path to input video file'
        ),
        
        # Node 1: Video Streamer Node
        # This node streams the input video as images on the 'camera_image' topic
        Node(
            package='pennair_vision',
            executable='video_streamer_node',
            name='video_streamer_node',
            arguments=[LaunchConfiguration('video_path')],
            output='screen',
            parameters=[]
        ),
        
        # Node 2: Shape Detection Node  
        # This node runs the shape detection algorithm on incoming images
        # and publishes detections (positions, outlines) on ROS2 topics
        Node(
            package='pennair_vision',
            executable='shape_detector_node',
            name='shape_detector_node',
            output='screen',
            parameters=[]
        )
    ])

