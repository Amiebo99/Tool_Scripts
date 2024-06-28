from setuptools import setup, find_packages

setup(
    name='FTIR_concatenate',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'FTIR_concatenate=FTIR_concatenate.processor:main',
        ],
    },
    install_requires=[
        'pandas',
    ],
    author='Mitchell Mibus',
    author_email='mitchmibus@gmail.com',
    description='A script to process and concatenate CSV files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/csv_processor',
)
