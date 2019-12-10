from setuptools import find_packages, setup


def read(path):
    with open(path, 'r') as f:
        return f.read()


long_description = read('README.md')

setup(
    name='beb103_9_1',
    version='0.0.1',
    url='https://github.com/jye309/bebi103-9-1',
    description='Pip-installable package with functions used in hw9.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=read('requirements.txt').strip().split('\n'),
)
