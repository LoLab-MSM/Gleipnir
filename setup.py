from distutils.core import setup

setup(name='gleipnir',
      version='0.24.0',
      description='Python toolkit for Nested Sampling.',
      author='Blake A. Wilson',
      author_email='blake.a.wilson@vanderbilt.edu',
      url='https://github.com/LoLab-VU/Gleipnir',
      packages=['gleipnir', 'gleipnir.pysb_utilities'],
      license='MIT',
      keywords=['nested sampling', 'calibration', 'model selection']
     )
