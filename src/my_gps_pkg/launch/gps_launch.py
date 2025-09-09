from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nmea_navsat_driver',
            executable='nmea_serial_driver',
            name='nmea_driver_node',
            parameters=[
                {'port': '/dev/ttyACM0'},
                {'baud': 38400}, # This is the default, change if you configured a different one
                {'frame_id': 'gps_link'}
            ]
        )
    ])