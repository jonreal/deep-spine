from setuptools import setup, find_packages

setup(
    name="deep_spine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'mujoco',
        'stable-baselines3'
    ],
    author="Jonathan Realmuto",
    author_email="jdrealmuto@gmail.com",
    description="A project for modeling spinal networks",
    url="",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
    ],
)
