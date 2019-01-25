from setuptools import setup

setup(name='aidistillery',
      version='0.1',
      description='Distilling the progress in the field of artificial intelligence.',
      url='http://github.com/TheMTank/ai-distillery',
      author='The M Tank',
      author_email='themtank@lpag.de',
      license='MIT',
      packages=['aidistillery'],
      install_requires=[
            'numpy',
            'scipy',
            'sklearn',
            'matplotlib',
            'pandas',
            'spacy',
            'gensim==3.4.0',
            'fuzzywuzzy',
            'pytest'
      ],
      scripts=[
            'scripts/distill'
      ],
      zip_safe=False)
