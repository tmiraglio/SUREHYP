# SUREHYP
## Surface Reflectance from Hyperion: a Python package to preprocess Hyperion imagery and retrieve surface reflectance

This package was designed to obtain desmiled, destriped and georeferenced reflectance images from Hyperion imagery. It requires both L1R and L1T radiance data, as well as their associated metadata, that can be downloaded from the [USGS](https://earthexplorer.usgs.gov/) website.

## Description

[example](./example.py) is an example script containing the whole processing chain, that can be employed to process batches of hyperion images from L1R/L1T to surface reflectance. Users should update the various paths and filenames to their desired configuration.

[preprocess](./func/preprocess.py) contains the various functions called in the preprocessing step, to obtain georeferenced, desmiled, and destriped hyperspectral images.

[atmoCorrection](./func/atmoCorrection.py) contains the various functions called during the atmospheric correction.  

[various](./func/various.py) contains a variety of useful functions that may be called by the other files.  

### Installation

This package has been tested on Python 3.7.5 to 3.9.9.

For ease of installation, it is recommended to install `pyhdf`, `rasterio`, `richdem` and `gdal` with `conda` before running the package's pip command:

```
conda install pyhdf rasterio richdem gdal
```

Then, SUREHYP and all other dependencies can be installed with `pip`: 

```
pip install surehyp
```

<!---
```
python -m pip install SUREHYP
```
-->

An extra cython library may be compiled to allow usage of a 3D interpolation function faster than Scipy's. To install it, download [surehyp_cython_extra](https://github.com/tmiraglio/surehyp_cython_extra), navigate to the downloaded folder and run:

```
python setup.py install
```

If this extra library is not installed, the program will revert to Scipy functions.

To obtain Earth Engine credentials and be able to download data from GEE, users can follow the steps described [here](https://developers.google.com/earth-engine/guides/python_install-conda#get_credentials).

To obtain SMARTS, refer to [this section](https://github.com/tmiraglio/SUREHYP#third-party-softwares).

### Use




Functions for preprocessing the radiance data can be called with

```
import surehyp.preprocess
```

Functions dedicated to the atmospheric correction can be called with

```
import surehyp.atmoCorrection
```

### Preprocessing

The steps undertaken for the preprocessing of the L1R images are those presented in Thenkabail et al. (2018):

- VNIR and SWIR are treated separately
- desmiling is done according to the method presented by San and Suzen (2011)
- two destriping methods are available: 
    - the local destriping method described by Datt et al. (2003)
    - quadratic regression using local spatial statistics by Pal et al. (2020)
- VNIR and SWIR are aligned

The corrected L1R image is then georeferenced using the L1T image, using matching features to apply a homography. The corrected radiance image is then saved as a .bip file.

### Atmospheric correction

A thin cirrus removal method, according to the works of Gao and Li (2017), is available, as well as a cloud and cloud shadow detection algorithm, adapted from Braaten et al. (2015).

The atmospheric correction is based on the SMARTS (Gueymard (2001), Gueymard (2019)) radiative transfer model. Two options are available:

- surface if considered flat, with an altitude corresponding to the site average
- topography is taken into account

The equation to retrieve surface reflectance <img src="https://render.githubusercontent.com/render/math?math=\rho"> from radiance is:

<img src="https://render.githubusercontent.com/render/math?math=\rho=\frac{\pi{}*(L-L_{haze})}{T_{gs}*(E_{sun}*cos\theta_{Z}*T_{sg} + E_{down})}">

with <img src="https://render.githubusercontent.com/render/math?math=T_{sg}"> the atmospheric transmittance along the ground-sensor path, <img src="https://render.githubusercontent.com/render/math?math=E_{sun}"> the solar irradiance, <img src="https://render.githubusercontent.com/render/math?math=\theta_{Z}"> angle of solar incidence on the surace (zenith angle if surface is considered flat), <img src="https://render.githubusercontent.com/render/math?math=T_{sg}"> the atmospheric transmittance along the sun-ground path, and <img src="https://render.githubusercontent.com/render/math?math=E_{down}"> the diffuse irradiance.

<img src="https://render.githubusercontent.com/render/math?math=\theta_{Z}"> is known from the image metadata, <img src="https://render.githubusercontent.com/render/math?math=E_{sun}">, <img src="https://render.githubusercontent.com/render/math?math=T_{gs}">, <img src="https://render.githubusercontent.com/render/math?math=T_{sg}"> and <img src="https://render.githubusercontent.com/render/math?math=E_{down}"> are outputs from SMARTS, and <img src="https://render.githubusercontent.com/render/math?math=L_{haze}"> is extracted from the image using the dark objet method presented by Chavez (1988).

Parameters such as ozone concentration, water vapor, or site altitude are extracted from the image using the water vapor absorption bands (for water vapor) or from Google Earth Engine (for water vapor, ozone and altitude). 

The DEM can be downloaded from GEE and slope and aspect are obtained locally to save download time as downloading the three images from GEE may be slow. A Modified-Minnaert method can be caller after the topographic correction to compensate over-corrected areas.

The reflectance image is then saved as a .bip file.

## Third-party softwares

This package uses SMARTS: Simple Model of the Atmospheric Radiative Transfer of Sunshine, and an updated function from the py-SMARTS package.

### SMARTS 
**Users can download SMARTS 2.9.5 from [NREL](https://www.nrel.gov/grid/solar-resource/smarts.html), or contact Dr. Christian A. Gueymard (Chris@SolarConsultingServices.com) to obtain the latest version available.**

Users will have to update the path and the file names depending on their SMARTS version with extra keywords when calling the function `runSMARTS` of [atmoCorrection](./func/atmoCorretion.py). Please note that depending on the SMARTS version, some output variables from SMARTS may have different names and therefore need to be updated. The names used in the present script should work with both SMARTS v.2.9.5 and v.2.9.8.1.

Users should add the path to their SMARTS installation at the start of their script:

```
os.environ['SMARTSPATH']='./path/to/smarts/folder/'
```

### py-SMARTS 
[py-SMARTS](https://github.com/NREL/pySMARTS) is shared under a BSD-3-Clause license:

Copyright (c) 2021 National Renewable Energy Laboratory, University of Arizona Board of Regents

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## References

P. S. Thenkabail, J. G. Lyon, and A. Huete, Advanced Applications in Remote Sensing of Agricultural Crops and Natural Vegetation. 2018. doi: 10.1201/9780429431166.

B. T. San and M. L. Suzen, "Evaluation of cross-track illumination in EO-1 hyperion imagery for lithological mapping", International Journal of Remote Sensing, vol. 32, no. 22, pp. 7873-7889, 2011, doi: 10.1080/01431161.2010.532175.

B. Datt, T. R. McVicar, T. G. van Niel, D. L. B. Jupp, and J. S. Pearlman, "Preprocessing EO-1 Hyperion hyperspectral data to support the application of agricultural indexes", IEEE Transactions on Geoscience and Remote Sensing, vol. 41, no. 6 PART I, pp. 1246-1259, Jun. 2003, doi: 10.1109/TGRS.2003.813206.

M. K. Pal, A. Porwal, T. M. Rasmussen,Noise reduction and destriping usinglocal spatial statistics and quadratic regression from Hyperion images,J. Appl. Remote Sens.14(1), 016515 (2020), doi: 10.1117/1.JRS.14.016515

B.C. Gao and R.R. Li, Removal of thin cirrus scattering effects in Landsat 8 OLI images using the cirrus detecting channel, Remote Sensing 9, 834, 2017

Braaten, J. D., Cohen, W. B., & Yang, Z. (2015). Automated cloud and cloud shadow identification in Landsat MSS imagery for temperate ecosystems. Remote Sensing of Environment, 169, 128�138. https://doi.org/10.1016/j.rse.2015.08.006

C. A. Gueymard, "Parameterized transmittance model for direct beam and circumsolar spectral irradiance", Solar Energy, vol. 71, no. 5, pp. 325-346, Nov. 2001, doi: 10.1016/S0038-092X(01)00054-8.

C. A. Gueymard, "The SMARTS spectral irradiance model after 25 years: New developments and validation of reference spectra", Solar Energy, vol. 187, pp. 233-253, Jul. 2019, doi: 10.1016/j.solener.2019.05.048.

P. S. Chavez, "An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data", Remote Sensing of Environment, vol. 24, no. 3, pp. 459-479, Apr. 1988, doi: 10.1016/0034-4257(88)90019-3.
