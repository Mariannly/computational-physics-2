from setuptools import setup, find_packages

setup(
    name='orbits',
    version='0.1',
    description='A 2D two-body simulation of a planet orbiting a black' \
    'hole with or without relativistic corrections',
    author='Mariannly Marquez',
    author_email='mariannly.marquez@yachaytech.edu.ec',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'argparse',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'orbits=orbits.orbit:main',
        ], #this never worked! Don't know why
    },
    include_package_data=True,
)
