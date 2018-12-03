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
            'scripts/embed_doc2vec',
            'scripts/embed_lsa',
            'scripts/embed_word2vec',
            'scripts/harvest_semanticscholar',
            'scripts/extract_entities',
            'scripts/index_tfidf'
      ],
      zip_safe=False)
