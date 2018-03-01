# BH EM - Black-Hole Electromagnetic Emission

This is a package for calculating the electromagnetic (EM) emission from black holes (BH) using analytic/semi-analytic and simple numerical prescriptions for disk structure and its subsequent radiation.  The package uses class structures to first create a `Disk` object which calculates magnetohydrodynamic profiles, and second a emission object which calculates the observed spectra and electromagnetic characteristics.

A description of the basic file structure and the changes in each version can be found in `CHANGES.md`.  Reference material is included in the `docs/` directory.  Jupyter notebooks for exploring the basic structure, properties and usage of the code can be found in the `notebooks/` directory.

Bugs, suggestions and improvements are welcome through the [github page (github.com/lzkelley/bhem)](https://github.com/lzkelley/bhem).


## Installation from Source
- Open a terminal, move to the directory where you want to install the code.
- Clone the source repository: `$ git clone git@github.com:lzkelley/bhem.git`
- Move into the downloaded repository: `$ cd bhem`
- Setup the "upstream" remote branch to get updates: `$ git remote add upstream https://github.com/lzkelley/bhem.git`
- Install the package: `pip install -e`.

### Updating
With the git `upstream branch` setup, you can [update your code simply](https://help.github.com/articles/syncing-a-fork/):
- 'Fetch' the most recent changes: `git fetch upstream`
- Switch to the 'master' (or target) branch: `git checkout master`
- Merge in changes: `git merge upstream/master`
- Re-install the package: `pip install -e .`


## References
- [SS97]: **Shakura & Sunyaev 1977** - [1973A&A....24..337S](http://adsabs.harvard.edu/abs/1973A%26A....24..337S)
- [NY95a]: **Narayan & Yi 1995a** - [1995ApJ...444..231N](http://adsabs.harvard.edu/abs/1995ApJ...444..231N)
    - First detailed derivation of the "ADAF" solution, self-similar radial profiles.
- [NY95b]: **Narayan & Yi 1995b** - [1995ApJ...452..710N](http://adsabs.harvard.edu/abs/1995ApJ...452..710N)
    - Expansion on [NY95b] including heating/cooling sources and simple spectral models.
- [M96]: **Mahadevan 1997** - [1997ApJ...477..585M](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:astro-ph/9609107)
    - Simplified spectral models based on NY95b, but considering integrated quantities over the whole disk to get 'characteristic' quantities and spectra.
    - Also includes numerous numerical tricks for easily calculating relevant quantities.
- [QN98]: **Quataert & Narayan 1998** - [1999ApJ...520..298Q](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:astro-ph/9810136)
