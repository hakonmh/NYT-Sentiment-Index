from setuptools import setup, find_packages
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

setup(
    name='newsindex',
    version='0.1.2',
    description='A daily sentiment index based on New York Times economic news.',
    python_requires='>=3.8',
    packages=find_packages(exclude=['tests', 'data']),
    install_requires=[
        'pandas>=1.4.0',
        'pynytimes',
        'torch',
        'transformers',
    ],
)
