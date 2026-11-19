from setuptools import find_packages, setup

package_name = 'my_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mu-09',
    maintainer_email='dltjdwls2832@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'beep_node = my_package.beep_node:main',
            'depth_checker = my_package.depth_checker:main',
            'robot_capture = my_package.robot_capture:main',
            'nav_to_pose= my_package.nav_to_pose:main',
            'nav_through_pose= my_package.nav_through_pose:main',
            'follow_waypoints= my_package.follow_waypoints:main',
            'depth_to_3d= my_package.depth_to_3d:main',
            '3_2_e_click_approach= my_package.3_2_e_click_approach:main',
            'auto_follow= my_package.auto_follow:main',
            'rc_car_alert_viewer= my_package.rc_car_alert_viewer:main',
            'integration= my_package.integration:main',
            'person_follow= my_package.lidar_nav2_follow_robot9_person:main',
            'worker_follow= my_package.worker_follow_node:main',
            'worker_detect= my_package.worker_detect_node:main',
            'web_relay= my_package.web_relay_node:main',
            'console= my_package.terminal_console:main',   # 웹 대신 터미널 테스트
            'stop_report= my_package.stop_report:main',    # 추종 정지 원인 표

        ],
    },
)

