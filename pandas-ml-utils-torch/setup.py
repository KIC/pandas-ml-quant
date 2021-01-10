"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.2.0'

import os
from setuptools import setup, find_packages


setup(
   name="pandas-ml-utils-torch",
   version=__version__,
   author='KIC',
   author_email='',
   packages=find_packages(),
   scripts=[],
   url='https://github.com/KIC/pandas-ml-quant',
   license='MIT',
   description=__doc__,
   long_description=open('Readme.md').read(),
   long_description_content_type='text/markdown',
   install_requires=open("requirements.txt").read().splitlines() + [f"pandas-ml-common=={__version__}", f"pandas-ml-utils=={__version__}"],
   extras_require={
      "dev": open("dev-requirements.txt").read().splitlines(),
   },
   include_package_data=True,
   classifiers=[
      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.7',
   ],
   keywords=['pandas', 'ml', 'util', 'quant'],
)
