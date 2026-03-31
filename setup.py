from setuptools import setup, find_packages

setup(
    name='Aura-Data-Analyst',
    version='0.1.0',
    description='Data Analyst Agent for numerical and semantic analysis',
    author='Data Analyst Agent',
    packages=find_packages(),
    install_requires=[
        'polars',
        'numpy',
        'scikit-learn',
        'hdbscan',
        'umap-learn',
        'chromadb',
        'google-genai',
        'python-dotenv',
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
