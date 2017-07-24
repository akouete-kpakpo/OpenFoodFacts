# -*- coding: utf-8 -*-
"""
"""

from setuptools import setup, find_packages

import logodetection

setup(
    name = 'logodetection',
    version = '0.1.0.dev',
    url = 'https://github.com/KevinKpakpo/OpenFoodFacts.git',
    license = 'http://www.fsf.org/licensing/licenses/agpl-3.0.html',
    author='Kévin Kpakpo',
    description='Répertoire pour détecter les logos sur les images de OpenFoodFacts.',
    #long_description=__doc__,
    #py_modules=['logodetection'],
    packages=find_packages(), 
    zip_safe=False,
    platforms='any',
    install_requires=['numpy','keras'],
    include_package_data=True,
)