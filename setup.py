from setuptools import setup

setup(name='pyzpace',
      version='0.1',
      description='Zach Pace\'s astronomy-related python tools',
      url='http://github.com/zpace/python-personal',
      author='Zach Pace',
      author_email='zpace@astro.wisc.edu',
      license='MIT',
      packages=['pyzpace'],
      zip_safe=False,
      python_requires='>=3',
      install_requires=['numpy', 'scipy', 'matplotlib', 'astropy'])
