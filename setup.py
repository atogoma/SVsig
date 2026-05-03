from setuptools import setup, find_packages

setup(
    name='SVSig',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'SVSig = svsig.main:main',
        ],
    },
    install_requires=[
        'pysam>=0.15.0',
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'torch>=2.0.0',
    ],
    author='zzz',
    description='SV特征分析工具包',
    python_requires='>=3.7',
)