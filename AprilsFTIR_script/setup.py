from setuptools import setup, find_packages

setup(
    name='csv_processor',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'csv_processor=csv_processor.processor:main',
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
