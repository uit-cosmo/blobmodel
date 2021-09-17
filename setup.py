import os
from setuptools import setup

name = '2d_porpagating_blobs'

with open('README.md') as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(name=name,
      description='2d model for scrape-off layer',
      #long_description=long_description,
      #long_description_content_type='text/markdown',
      url='https://github.com/gregordecristoforo/2d_propagating_blobs/',
      author='Gregor Decristoforo',
      author_email='gregor.decristoforo@uit.no',
      license='GPL',
      version='1.0',
      packages=['model'],
      python_requires='>=3.5',
      #install_requires=['xarray>=0.11.2',
      #                  'scipy>=1.2.0',
      #                  'dask-image>=0.2.0'],
      #tests_require=['pytest',
      #:wq               'numpy'],
      #classifiers=[
      #  'Intended Audience :: Education',
      #  'Intended Audience :: Science/Research',
      #  'License :: OSI Approved :: MIT License',
      #  'Programming Language :: Python :: 3 :: Only',
      #  'Programming Language :: Python :: 3.5',
      #  'Programming Language :: Python :: 3.6',
      #  'Programming Language :: Python :: 3.7',
      #  'Topic :: Scientific/Engineering :: Visualization',
      #],
      zip_safe=False)