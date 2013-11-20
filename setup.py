import sys

if 'develop' in sys.argv:
    # use setuptools for develop, but nothing else
    from setuptools import setup
else:
    from distutils.core import setup

with open('README.rst') as file:
    long_description = file.read()

execfile('image_tools/__version__.py')

setup(name='image_tools',
      version=__version__,
      description='Image Toolkit: Fourier & other image tools.',
      long_description=long_description,
      author='Adam Ginsburg',
      author_email='adam.g.ginsburg@gmail.com',
      data_files=[],
      url='',
      packages=['fft_psd_tools','image_tools']
     )
