# SUREHYP
## Surface Reflectance from Hyperion: a Python package to preprocess Hyperion imagery and retrieve surface reflectance

This package was designed to obtain desmiled, destriped and georeferenced reflectance images from Hyperion imagery. It requires both L1R and L1T radiance data, as well as their associated metadata, that can be downloaded from https://earthexplorer.usgs.gov/.

## Description

`surehyp.py` is a script containing the whole processing chain. It allows multithreaded processing. Users should update the various paths and filenames to their desired configuration.

`preprocess.py` contains the various functions called in the preprocessing step, to obtain georeferenced, desmiled, and destriped hyperspectral images.

`atmoCorrection.py` contains the various functions called during the atmospheric correction.  

### Required packages

- `numpy`
- `ee`
- `matplotlib`
- `spectral`
- `scipy`
- `pandas`
- `opencv`
- `pyhdf`
- `scikit-learn`
- `rasterio`

### Preprocessing

The steps undertaken for the preprocessing of the L1R images are those presented in Thenkabail et al. (2018):

- VNIR and SWIR are treated separately;
- desmiling is done according to the method presented by San and Suzen (2011)
- destriping is done using the local destriping method described by Datt et al. (2003)

The corrected L1R image is then georeferenced using the L1T image, using matching features to apply a homography. The corrected radiance image is then saved as a .bip file.

### Atmospheric correction

The atmospheric correction is based on the SMARTS (Gueymard (2001), Gueymard (2019)) radiative transfer model. The equation to retrieve surface reflectance <img src="https://render.githubusercontent.com/render/math?math=\rho"> from radiance is:

<img src="https://render.githubusercontent.com/render/math?math=\rho=\frac{\pi{}(L-L_{haze})}{cos\theta_{V}(E_{sun}cos\theta_{Z}T+E_{down})}">


with <img src="https://render.githubusercontent.com/render/math?math=\theta_{V}"> the satellite zenith angle, <img src="https://render.githubusercontent.com/render/math?math=E_{sun}"> the solar irradiance, <img src="https://render.githubusercontent.com/render/math?math=\theta_{Z}"> the solar zenith angle, <img src="https://render.githubusercontent.com/render/math?math=T"> the atmospheric transmittance along the sun-ground-sensor path, and <img src="https://render.githubusercontent.com/render/math?math=E_{down}"> the diffuse irradiance.

<img src="https://render.githubusercontent.com/render/math?math=\theta_{V}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta_{Z}"> are known from the image metadata, <img src="https://render.githubusercontent.com/render/math?math=E_{sun}">, <img src="https://render.githubusercontent.com/render/math?math=T"> and <img src="https://render.githubusercontent.com/render/math?math=E_{down}"> are outputs from SMARTS, and <img src="https://render.githubusercontent.com/render/math?math=L_{haze}"> is extracted from the image using the dark objet method presented by Chavez (1988).

Parameters such as ozone concentration, water vapor, or site altitude are extracted from the image using the water vapor absorption bancs (for water vapor) or from Google Earth Engine (for water vapor, ozone and altitude). 

The reflectance image is then saved as a .bip file.

## NREL softwares

This package uses SMARTS: Simple Model of the Atmospheric Radiative Transfer of Sunshine, and an updated function from the py-SMARTS package.

### SMARTS 
**Users can download SMARTS 2.9.5 from https://www.nrel.gov/grid/solar-resource/smarts.html, or contact Dr. Christian A. Gueymard (Chris@SolarConsultingServices.com) to obtain the latest version available.**

Users will have to update the path and the file names depending on their SMARTS version and installation folder in the functions `runSMARTS` and `smartsALL_original` of `atmoCorrection.py`. Please note that depending on the SMARTS version, some output variables from SMARTS may have different names and therefore need to be updated. The names used in the present script are those of SMARTS v.2.9.8.1.

### py-SMARTS 
py-SMARTS (https://github.com/NREL/pySMARTS) is shared under a BSD-3-Clause license:

Copyright (c) 2021 National Renewable Energy Laboratory, University of Arizona Board of Regents

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## References

P. S. Thenkabail, J. G. Lyon, and A. Huete, Advanced Applications in Remote Sensing of Agricultural Crops and Natural Vegetation. 2018. doi: 10.1201/9780429431166.

B. T. San and M. L. S\"{u}zen, "Evaluation of cross-track illumination in EO-1 hyperion imagery for lithological mapping",ù International Journal of Remote Sensing, vol. 32, no. 22, pp. 7873-7889, 2011, doi: 10.1080/01431161.2010.532175.

B. Datt, T. R. McVicar, T. G. van Niel, D. L. B. Jupp, and J. S. Pearlman, "Preprocessing EO-1 Hyperion hyperspectral data to support the application of agricultural indexes",ù IEEE Transactions on Geoscience and Remote Sensing, vol. 41, no. 6 PART I, pp. 1246-1259, Jun. 2003, doi: 10.1109/TGRS.2003.813206.

C. A. Gueymard, "Parameterized transmittance model for direct beam and circumsolar spectral irradiance",ù Solar Energy, vol. 71, no. 5, pp. 325-346, Nov. 2001, doi: 10.1016/S0038-092X(01)00054-8.

C. A. Gueymard, "The SMARTS spectral irradiance model after 25†years: New developments and validation of reference spectra",ù Solar Energy, vol. 187, pp. 233-253, Jul. 2019, doi: 10.1016/j.solener.2019.05.048.

P. S. Chavez, "An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data",ù Remote Sensing of Environment, vol. 24, no. 3, pp. 459-479, Apr. 1988, doi: 10.1016/0034-4257(88)90019-3.
