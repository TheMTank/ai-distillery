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
            'fuzzywuzzy'
      ],
      scripts=[
            'bin/embed_doc2vec',
            'bin/embed_lsa',
            'bin/embed_word2vec',
            'bin/harvest_semanticscholar',
            'bin/extract_entities'
      ],
      zip_safe=False)
