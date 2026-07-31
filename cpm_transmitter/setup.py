from setuptools import setup

package_name = 'cpm_transmitter'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HB',
    maintainer_email='hb@example.com',
    description='ETSI ITS CPM transmitter converting map coords to WGS84 and using UDP services',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'cpm_pub_node = cpm_transmitter.cpm_pub_node:main',
            'cpm_pub = cpm_transmitter.node_cpm:main',
            'viz_node = cpm_transmitter.viz_cpm:main',
            'sender = cpm_transmitter.cpm_udp_sender:main',
            'receiver = cpm_transmitter.cpm_udp_receiver:main',
            'cpf_node = cpm_transmitter.cooperative_fusion_node:main',
            'dynamic_cpf = cpm_transmitter.dynamic_node:main'
        ],
    },
)

