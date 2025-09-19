from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'decision_making_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'lib'), glob('lib/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hhs',
    maintainer_email='hhs@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planner_node = decision_making_pkg.path_planner_node:main',
            'motion_planner_node = decision_making_pkg.motion_planner_node:main',
            'waypoint_follower_node = decision_making_pkg.waypoint_follower_node:main', # Add this line
            'path_planner_robust_node = decision_making_pkg.path_planner_robust_node:main',
        ],
    },
)