from setuptools import setup, find_packages
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

setup(
    name='newspulse',
    version='0.1.0',
    description='A daily sentiment index based on New York Times economic news.',
    python_requires='>=3.8',
    packages=find_packages(['tests', 'data'])
)
