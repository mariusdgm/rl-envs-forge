from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_modules = f.read().splitlines()

setup(
    name='rl_envs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required_modules,
)