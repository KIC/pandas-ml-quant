"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.2.0'

import os
import re

from setuptools import setup, find_packages

url = 'https://github.com/KIC/pandas-ml-quant/pandas-ml-utils'


def fix_github_links(line):
   fixed_images = re.sub(r'(^\[ghi\d+]:\s+)', f'\\1{url}/raw/{__version__}/', line)
   fixed_location = re.sub(r'(^\[ghl\d+]:\s+)', f'\\1{url}/tree/{__version__}/', fixed_images)
   fixed_files = re.sub(r'(^\[ghf\d+]:\s+)', f'\\1{url}/blob/{__version__}/', fixed_location)
   return fixed_files


setup(
   name="pandas-ml-utils",
   version=__version__,
   author='KIC',
   author_email='',
   packages=find_packages(),
   scripts=[],
   url=url,
   license='MIT',
   description=__doc__,
   long_description='\n'.join([fix_github_links(l) for l in open('Readme.md').readlines()]),
   long_description_content_type='text/markdown',
   install_requires=open("requirements.txt").read().splitlines() + [f"pandas-ml-common=={__version__}"],
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
