from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]
    

setup(
    name='ConfSelect',
    version='0.0.1',
    description='Conformalized Selection',
    url='https://github.com/ying531/conformal-selection',
    author='Ying Jin and Emmanuel Candes',
    author_email='ying531@stanford.edu',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT License",
)