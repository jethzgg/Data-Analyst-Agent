from setuptools import setup, find_packages

setup(
    name='Data-Analyst',
    version='0.1.0',
    description='Data Analyst Agent for numerical and semantic analysis',
    author='Data Analyst Agent',
    packages=find_packages(),
    install_requires=[
        'polars>=1.0.0',
        'numpy>=1.26.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'psycopg2-binary>=2.9.9',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
