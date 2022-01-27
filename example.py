import numpy as np
import sys
import ee
from functools import partial
from multiprocessing import Pool
import sys, os

import surehyp.preprocess
import surehyp.atmoCorrection

def processImage(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut):
    ee.Initialize()

    print('concatenate the L1T image')
    surehyp.preprocess.processImage(fname,pathToL1Timages,pathToL1TimagesFiltered)

    print('read the L1R image')
    arrayL1R=surehyp.preprocess.readL1R(pathToL1Rimages+fname+'/',fname)

    print('get the L1R image parameters')
    metadata,bands,fwhms=surehyp.preprocess.getImageMetadata(pathToL1Rimages+fname+'/',fname)

    print('separates VNIR and SWIR')
    VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm=surehyp.preprocess.separating(arrayL1R,bands,fwhms)

    print('converts DN to radiance')
    VNIR,SWIR=surehyp.preprocess.DN2Radiance(VNIR,SWIR)

    print('aligning VNIR and SWIR, part 1')
    VNIR,SWIR=surehyp.preprocess.alignSWIR2VNIRpart1(VNIR,SWIR)

    print('desmiling')
    VNIR=surehyp.preprocess.smileCorrectionAll(VNIR,2,check=False)
    SWIR=surehyp.preprocess.smileCorrectionAll(SWIR,2,check=False)

    print('destriping -  Pal et al. (2020)')
    VNIR,nc=surehyp.preprocess.destriping_quadratic(VNIR)
    VNIR=surehyp.preprocess.destriping_local(VNIR,nc)
    SWIR,nc=surehyp.preprocess.destriping_quadratic(SWIR)
    SWIR=surehyp.preprocess.destriping_local(SWIR,nc)

    #print('destriping - Datt (2003)')
    #VNIR=surehyp.preprocess.destriping(VNIR,'VNIR',0.11)
    #SWIR=surehyp.preprocess.destriping(SWIR,'SWIR',0.11)

    print('aligning VNIR and SWIR, part 2')
    VNIR,SWIR=surehyp.preprocess.alignSWIR2VNIRpart2(VNIR,VNIRb,SWIR,SWIRb)

    print('assemble VNIR and SWIR')
    arrayL1R,wavelengths,fwhms=surehyp.preprocess.concatenateImages(VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm)

    print('smooth the cirrus bands for later thin cirrus removal') #may be necessary for Hyperion as an incomplete destriping for this band would then affect every other band during the thin cirrus removal
    arrayL1R=surehyp.preprocess.smoothCirrusBand(arrayL1R,wavelengths)

    print('georeference the corrected L1R data using the L1T data')
    arrayL1Rgeoreferenced, metadataGeoreferenced=surehyp.preprocess.georeferencing(arrayL1R,pathToL1TimagesFiltered,fname)

    print('save the processed image as an ENVI file')
    surehyp.preprocess.savePreprocessedL1R(arrayL1Rgeoreferenced,wavelengths,fwhms,metadataGeoreferenced,pathToL1Rimages,pathToL1Rmetadata,metadata,fname,pathOut+fname+'_L1R_complete')

    for f in os.listdir(pathOut):
        if (fname in f) and ('_tmp' in f):
            os.remove(os.path.join(pathOut,f))

def atmosphericCorrection(fname,pathOut):
    print('open processed radiance image')
    L,bands,fwhms,processing_metadata,metadata=surehyp.atmoCorrection.getImageAndParameters(pathOut+fname+'_L1R_complete')

    ####
    #get info from the processing metadata for clearer visualization in the input of the subsequent functions
    longit=processing_metadata['longit']
    latit=processing_metadata['latit']
    datestamp1=processing_metadata['datestamp1']
    datestamp2=processing_metadata['datestamp2']
    zenith=processing_metadata['zenith']
    azimuth=processing_metadata['azimuth']
    satelliteZenith=processing_metadata['satelliteZenith']
    satelliteAzimuth=processing_metadata['satelliteAzimuth']
    scaleFactor=processing_metadata['scaleFactor']

    UL_lat=processing_metadata['UL_lat']
    UL_lon=processing_metadata['UL_lon']
    UR_lat=processing_metadata['UR_lat']
    UR_lon=processing_metadata['UR_lon']
    LL_lat=processing_metadata['LL_lat']
    LL_lon=processing_metadata['LL_lon']
    LR_lat=processing_metadata['LR_lat']
    LR_lon=processing_metadata['LR_lon']

    year=processing_metadata['year']
    month=processing_metadata['month']
    day=processing_metadata['day']
    hour=processing_metadata['hour']
    minute=processing_metadata['minute']
    doy=processing_metadata['doy']

    thetaZ=processing_metadata['thetaZ']
    thetaV=processing_metadata['thetaV']
    ####

    print('removal of thin cirrus')
    L=surehyp.atmoCorrection.cirrusRemoval(bands,L,latit,longit,year,month,day,hour,doy,thetaV)

    print('get haze spectrum')
    L,Lhaze=surehyp.atmoCorrection.darkObjectDehazing(L,bands)

    print('get average elevation of the scene from GEE')
    altit=surehyp.atmoCorrection.getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon)

    print('get atmosphere content')
    wv,o3=surehyp.atmoCorrection.getAtmosphericParameters(bands,L,datestamp1,year,month,day,hour,minute,doy,longit,latit,altit,thetaV)


    ########################################
    # Atmospheric correction -- flat surface
    print('obtain radiative transfer outputs')
    #get the atmosphere parameters for the sun-ground section using the image acquisition time to determine sun angle
    df=surehyp.atmoCorrection.runSMARTS(ALTIT=altit,LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=int(hour)+int(minute)/60,SUNCOR=atmoCorrection.get_SUNCOR(doy),IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    #get the atmosphere parameters for the ground-satellite section by setting the 'sun' (in SMARTS) at the satellite zenith position to get the transmittance over the correct optical path length
    df_gs=surehyp.atmoCorrection.runSMARTS(ALTIT=altit,LATIT=0,LONGIT=0,IMASS=0,SUNCOR=atmoCorrection.get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(thetaV)*180/np.pi,AZIM=0,IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)

    print('compute radiance to reflectance')
    R=surehyp.atmoCorrection.computeLtoE(L,bands,df,df_gs)

    print('save the reflectance image')
    surehyp.atmoCorrection.saveRimage(R,metadata,pathOut+fname+'_Reflectance_flat')

    #######################################
    #atmospheric correction -- rough terrain
    print('download DEM images from GEE')
    path_to_dem = surehyp.atmoCorrection.getDEMimages(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat)

    print('reproject DEM images')
    path_to_reprojected_dem = surehyp.atmoCorrection.reprojectDEM(pathOut+fname+'_L1R_complete.img',path_elev=path_to_dem)

    print("extract the data corresponding to the Hyperion image's pixels")
    elev, slope, wazim=surehyp.atmoCorrection.extractDEMdata(pathOut+fname+'_L1R_complete.img',path_elev=path_to_reprojected_dem)

    #define the steps of the LUT
    stepAltit=2
    stepTilt=45 #15#there is barely any difference in the results using 15 degrees or 7.5 degrees (less than 1% of relative reflectance difference)
    stepWazim=45 #15

    print('computing the LUT for the rough terrain correciton')
    R=surehyp.atmoCorrection.getDemReflectance(altitMap=elev,tiltMap=slope,wazimMap=wazim,stepAltit=stepAltit,stepTilt=stepTilt,stepWazim=stepWazim,latit=latit,longit=longit,WV=wv,AbO3=o3,year=year,month=month,day=day,hour=hour,doy=doy,satelliteZenith=satelliteZenith,satelliteAzimuth=satelliteAzimuth,L=L,bands=bands)

    print('save the reflectance image')
    surehyp.atmoCorrection.saveRimage(R,metadata,pathOut+fname+'_Reflectance_topo')

if __name__ == '__main__':

    ee.Initialize()

    pathToL1Rmetadata='./METADATA/METADATA.csv'

    pathToL1Timages="./L1T/"

    pathToL1Rimages="./L1R/"
    pathToL1TimagesFiltered="./L1T/filteredImages/"

    pathOut='./OUT/'

    fnames=['EO1H0430332014136110P3']

    ### IF MULTITHREADING -- example 
    #preprocessing
    #processes=10
    #with Pool(processes=processes) as pool:
    #    print('pooling the preprocessing')
    #    pool.map(partial(processImage,pathToL1Rmetadata=pathToL1Rmetadata,pathToL1Rimages=pathToL1Rimages,pathToL1Timages=pathToL1Timages,pathToL1TimagesFiltered=pathToL1TimagesFiltered,pathOut=pathOut),fnames)

    for fname in fnames:
        processImage(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut)

    for fname in fnames:
        atmosphericCorrection(fname,pathOut)

