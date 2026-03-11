#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:59:14 2022

@author: mlampert
"""
from setuptools import setup, find_packages


setup(
    name = 'flap',
    packages = find_packages(),
    version = '1.0',
    license = 'MIT',
    description = 'Library developed focusing on NSTX GPI diagnostic data including correlation based velocimetry and structure segmentation.',
    author = 'Mate Lampert',
    author_email = 'mlampert@pppl.gov',
    url = 'https://github.com/thelampire/flap_nstx',
    keywords = [
        'fusion',
        'data analysis',
        'NSTX',
        'segmentation',
        'velocimetry'
    ],
    install_requires = [
        'h5py >= 3.6.0',
        'ipykernel >= 6.9.2',
        'lxml >= 4.8.0',
        'matplotlib >= 3.5.1',
        'numpy >= 1.22.3',
        'opencv-python >= 4.5.5',
        'pandas >= 1.4.1',
        'pickleshare >= 0.7.5',
        'scipy >= 1.8.0',
        'tornado >= 6.1',
        'pims >= 0.5',
        'scipy >= 1.7.5',
        'scikit-image >= 0.18.3',
        'scikit-learn >= 0.24.2',
        'imageio >= 2.9.0',
        'pandas >= 1.2.4',
        'shapely >= 2.1.2',
        'pims >= 0.7',
        'seaborn >= 0.13.2',
        'imageio >= 2.37.0',
        ''

    ],
    classifiers = [
        'Programming Language :: Python :: 3.9.10'
    ],
)