# Always prefer setuptools over distutils

import setuptools  # this is the "magic" import
import os
from numpy.distutils.core import setup, Extension

#from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

__key__ = 'PACKAGE_VERSION'
__version__= os.environ[__key__] if __key__ in os.environ else '0.0.3'

setup(
    name='frdd-wofs-phi', 
    version=__version__,
    description='Official WoFS-Phi Repository', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-phi', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists',
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'python>=3.10',
    ],
    package_data={'wofs_phi' : ['*.txt']},

    packages=['wofs_phi'],  # Required

    python_requires='>=3.10, <4',
    package_dir={'wofs_ml_severe': 'wofs_ml_severe'},
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-phi/issues',
        'Source': 'https://github.com/NOAA-National-Severe-Storms-Laboratory/frdd-wofs-phi',
    },
)
