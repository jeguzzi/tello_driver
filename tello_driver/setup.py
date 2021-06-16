from setuptools import setup
import glob

package_name = 'tello_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'tellopy', 'tellopy._internal'],
    package_dir={'tellopy': 'src/TelloPy/tellopy', 'tello_driver': 'src/tello_driver',
                 'tellopy._internal': 'src/TelloPy/tellopy/_internal'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch')),
        ('share/' + package_name + '/cfg', glob.glob('cfg/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jerome Guzzi',
    maintainer_email='jerome@idsia.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tello_driver_node = tello_driver.tello_driver_node:main',
        ],
    },
)
