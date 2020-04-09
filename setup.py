from setuptools import setup, find_packages

setup(
    name='compression',
    version='0.1',
    description='Data Compression Codecs',
    author='Josh Goldman',
    url='https://github.com/josheligoldman/TorchCompression',
    packages=find_packages(),
    install_requires=['torch', 'numpy']
)

