from setuptools import find_packages, setup

package_name = 'obstical_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ven6042s@hs-coburg.de',
    maintainer_email='ven6042s@hs-coburg.de',
    description='Obstacle detection using LiDAR, DetectNet, and camera fusion',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_obstacles = obstical_detection.detect_obs:main',
            'detect_in_range = obstical_detection.detect_obs_in_range:main',
            'cam_lidar_fusion = obstical_detection.cam_and_lidar:main',
            'cam_lidar = obstical_detection.cam_and_lidar_pointCluster:main',
            'fusion_cpm = obstical_detection.data_fusion_cpm:main',
            'rosbag_csv = obstical_detection.rosbag_to_csv:main',
            'pre_dict = obstical_detection.predict:main'
        ],
    },
)

