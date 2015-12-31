#!/usr/bin/env python3

from setuptools import setup, find_packages

version = '0.1.0'

from distutils.command.install import install as _install


class Install(_install):
    def run(self):
        _install.run(self)
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

setup(
    name='neuralnet',
    version=version,
    description="",
    long_description="""\
""",
    classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords='rnn,nn',
    author='Oscar Eriksson',
    author_email='oscar.eriks@gmail.com',
    url='',
    license='',
    cmdclass={'install': Install},
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'zope.interface',
        'pyyaml',
        'simplejson',
        'numpy',
        'nltk',
        'theano',
    ],
    setup_requires=['nltk'],
    entry_points={
        'console_scripts': [
            'neuralnet = neuralnet:entry',
        ]
    })
