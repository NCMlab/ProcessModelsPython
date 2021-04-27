#!/usr/bin/env python
import setuptools import setup

setup(
   name='process_models',
   version='1.0',
   description='Module to aid with bootstraping and mediation anaylsis',
   author='Jason Steffener, Ariel Barenboim, Jeremy Soong',
   author_email='jason.steffener@uottawa.ca, abare077@uottawa.ca, jsoon049@uottawa.ca',
   packages=['process_models'],  #same as name
   install_requires=['scikit-learn', 'pandas', 'nibabel','numpy'], #external packages as dependencies
)
