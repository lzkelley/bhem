from setuptools import setup

readme = open('README.md').read()

requirements = ['numpy', 'scipy', 'matplotlib']

VERSION_FNAME = "VERSION.txt"
version = open(VERSION_FNAME, 'r').read().strip()

setup(
    name="bhem",
    version=version,
    author="Luke Zoltan Kelley",
    author_email="lkelley@cfa.harvard.edu",
    description=("Black-hole disk and electromagnetic-spectrum calculations."),
    license="MIT",
    url="https://github.com/lzkelley/bhem",
    packages=['bhem'],
    include_package_data=True,
    install_requires=requirements,
    long_description=readme,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
