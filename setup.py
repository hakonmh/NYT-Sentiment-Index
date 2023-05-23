from setuptools import setup, find_packages
import os

current_folder = os.path.abspath(os.path.dirname(__file__))

setup(
    name='newspulse',
    version='0.0.1',
    description='A news sentiment index.',
    python_requires='>=3.8',
    packages=find_packages(['tests', 'data'])
)
