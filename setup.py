from setuptools import setup, find_packages
import sys, os

version = '0.1.0'

setup(name='pusher',
      version=version,
      description="",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='',
      author='',
      author_email='',
      url='',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
        'pyyaml'
      ],
      entry_points={
        'console_scripts': [
          'nn = neuralnet:entry',
        ]
      },
      
      )
