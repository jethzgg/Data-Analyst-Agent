from setuptools import setup, find_packages

setup(
    name='Aura-Data-Analyst',
    version='0.1.0',
    description='Data Analyst Agent for numerical and semantic analysis',
    author='Data Analyst Agent',
    packages=find_packages(),
    install_requires=[
        'polars>=1.0.0',
        'numpy>=1.26.0',
        'scikit-learn>=1.3.0',
        'chromadb>=0.4.0',
        'hdbscan>=0.8.33',
        'umap-learn>=0.5.3',
        'psycopg2-binary>=2.9.9',
        'google-genai>=0.2.0',
        'python-dotenv>=1.0.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
