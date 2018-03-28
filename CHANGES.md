# BH EM - Change-Log

## Future / To-Do
    - General
        - Implement a full, radial ADAF spectral calculation by solving the heating/cooling equations as a function of radius (instead of integrated quantities as in Mahadevan 1996).
        - Consider reddening / obscuration.  How much would this modify spectra for typical parameters?

    - `bhem/`


## Current
    - `bhem/`
        - `basics.py`
            - Contains methods for simple physical calculations and operations.
        - `constants.py`
            - Store numerical and physical constants.
        - `disks.py`
            - Classes for different disk models.
            - Only require `mass` and either `mdot` or `fedd` for disks.  Use `**kwargs`.
            - `Thin`
                - Shakura-Sunyaev disk-model, with associated black-body-like spectrum.
            - `ADAF`
                - Narayan & Yi 1995 ADAF disk.
        - `radiation.py`
            - Methods for general radiation/emission calculations.
        - `spectra.py`
            - Classes for spectral calculations.  Currently only `Mahadevan96` for ADAF spectral calculations.
            - `Mahadevan96`
                - Class for calculating simple analytic spectra from an ADAF disk.
        - `utils.py`
            - Contains methods for simple logistical operations of general use.
            
    - `notebooks/`
        - `adaf.ipynb`
            - Examine ADAF disk structure and spectra
        - `shakura-sunyaev.ipynb`
            - Examine thin (Shakura-Sunyaev) disk structure and spectra
        - `bhem.ipynb`
            - Comparisons between both disk models



## [0.1] - 2018/03/01
    - Basic, working models of both ADAF and Thin (Shakura-Sunyaev) accretion disks.
        - Each are implemented as separated classes in the `disks.py` file.
    - Spectra can be calculated for each model.
        - Thin-Disk is a simple black-body calculation at each annulus of the disk.  This is included in the `Thin` disk class.
        - ADAF uses the Mahadevan 1996 simple analytic approximation for disk-integrated quantities, including Synchrotron, Bremsstrahlung, and Compton-Scattering (of the synchrotron).  This spectral model is implemented in the `Mahadevan96` class in `spectra.py`.
