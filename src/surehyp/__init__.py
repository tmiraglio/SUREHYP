import sys, os
import pandas as pd
from scipy import interpolate

import surehyp.preprocess
import surehyp.atmoCorrection

def main():
    """Entry point for the application script"""
    print("Call your main application code here")

def preprocess_radiance(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut,nameOut,destripingMethod='Pal',localDestriping=False,smileCorrectionOrder=2,checkSmile=False):    
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
    VNIR=surehyp.preprocess.smileCorrectionAll(VNIR,smileCorrectionOrder,check=checkSmile)
    SWIR=surehyp.preprocess.smileCorrectionAll(SWIR,smileCorrectionOrder,check=checkSmile)

    if destripingMethod=='Datt':
        print('destriping - Datt (2003)')
        VNIR=surehyp.preprocess.destriping(VNIR,'VNIR',0.11)
        SWIR=surehyp.preprocess.destriping(SWIR,'SWIR',0.11)
    elif destripingMethod=='Pal':
        print('destriping -  Pal et al. (2020)')
        VNIR,nc=surehyp.preprocess.destriping_quadratic(VNIR)
        if localDestriping==True:
            VNIR=surehyp.preprocess.destriping_local(VNIR,nc)
        SWIR,nc=surehyp.preprocess.destriping_quadratic(SWIR)
        if localDestriping==True:
            SWIR=surehyp.preprocess.destriping_local(SWIR,nc)
    else:
        print('no destriping method selected -> no destriping')

        print('aligning VNIR and SWIR, part 2')
    VNIR,SWIR=surehyp.preprocess.alignSWIR2VNIRpart2(VNIR,VNIRb,SWIR,SWIRb)

    print('assemble VNIR and SWIR')
    arrayL1R,wavelengths,fwhms=surehyp.preprocess.concatenateImages(VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm)

    print('smooth the cirrus bands for later thin cirrus removal') #may be necessary for Hyperion as an incomplete destriping for this band would then affect every other band during the thin cirrus removal
    arrayL1R=surehyp.preprocess.smoothCirrusBand(arrayL1R,wavelengths)

    print('georeference the corrected L1R data using the L1T data')
    arrayL1Rgeoreferenced, metadataGeoreferenced=surehyp.preprocess.georeferencing(arrayL1R,pathToL1TimagesFiltered,fname)

    print('save the processed image as an ENVI file')
    surehyp.preprocess.savePreprocessedL1R(arrayL1Rgeoreferenced,wavelengths,fwhms,metadataGeoreferenced,pathToL1Rimages,pathToL1Rmetadata,metadata,fname,pathOut+nameOut)

    for f in os.listdir(pathOut):
        if (fname in f) and ('_tmp' in f):
            os.remove(os.path.join(pathOut,f))

    return pathOut+nameOut

def atmosphericCorrection(pathToRadianceImage,pathToOutImage,stepAltit=1,stepTilt=15,stepWazim=15,demID='JAXA/ALOS/AW3D30/V3_2',elevationName='DSM',topo=False,smartsAlbedoFilePath='./SMARTS2981-PC_Package/Albedo/Albedo.txt'):
    print('open processed radiance image')
    L,bands,fwhms,processing_metadata,metadata=surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)

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

    print('removal of thin cirrus - Gao and Li (2017)')
    L=surehyp.atmoCorrection.cirrusRemoval(bands,L,latit,longit,year,month,day,hour,doy,thetaV)

    print('get haze spectrum - Chavez (1998)')
    L,Lhaze=surehyp.atmoCorrection.darkObjectDehazing(L,bands)

    print('get average elevation of the scene from GEE')
    altit=surehyp.atmoCorrection.getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon,demID=demID,elevationName=elevationName)

    print('get atmosphere content')
    wv,o3=surehyp.atmoCorrection.getAtmosphericParameters(bands,L,datestamp1,year,month,day,hour,minute,doy,longit,latit,altit,thetaV)

    ########################################
    # Atmospheric correction -- flat surface
    print('obtain radiative transfer outputs')
    #get the atmosphere parameters for the sun-ground section using the image acquisition time to determine sun angle
    df=surehyp.atmoCorrection.runSMARTS(ALTIT=altit,LATIT=latit,LONGIT=longit,IMASS=0,YEAR=year,MONTH=month,DAY=day,HOUR=int(hour)+int(minute)/60,ZENITH=zenith,AZIM=azimuth,SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    #get the atmosphere parameters for the ground-satellite section by setting the 'sun' (in SMARTS) at the satellite zenith position to get the transmittance over the correct optical path length
    df_gs=surehyp.atmoCorrection.runSMARTS(ALTIT=altit,LATIT=0,LONGIT=0,IMASS=0,SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(thetaV)*180/np.pi,AZIM=0,IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)

    print('compute radiance to reflectance')
    R=surehyp.atmoCorrection.computeLtoR(L,bands,df,df_gs)

    if topo==False:
        print('save the reflectance image')
        surehyp.atmoCorrection.saveRimage(R,metadata,pathToOutImage)
    else:
        #######################################
        #atmospheric correction -- rough terrain
        print('write Albedo.txt file for SMARTS')
        pathToAlbedoFile=surehyp.atmoCorrection.writeAlbedoFile(R,bands,pathOut=smartsAlbedoFilePath)

        print('get scene background reflectance')
        sp=pd.read_csv(pathToAlbedoFile,header=3,sep='\s+')
        w=sp.values[:,0]
        r=sp.values[:,1]
        f=interpolate.interp1d(w,r,bounds_error=False,fill_value='extrapolate')
        rho_background=f(df['Wvlgth']*1E-3)

        print('download DEM images from GEE')
        path_to_dem = surehyp.atmoCorrection.getDEMimages(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat,demID=demID,elevationName=elevationName)

        print('reproject DEM images')
        path_to_reprojected_dem = surehyp.atmoCorrection.reprojectDEM(pathToRadianceImage,path_elev=path_to_dem)

        print('resampling')
        path_elev=surehyp.atmoCorrection.matchResolution(pathToRadianceImage,path_elev=path_to_reprojected_dem)

        print("extract the data corresponding to the Hyperion image's pixels")
        elev, slope, wazim=surehyp.atmoCorrection.extractDEMdata(pathToRadianceImage,path_elev=path_to_reprojected_dem)

        print('computing the LUT for the rough terrain correction')
        R=surehyp.atmoCorrection.getDemReflectance(altitMap=elev,tiltMap=slope,wazimMap=wazim,stepAltit=stepAltit,stepTilt=stepTilt,stepWazim=stepWazim,latit=latit,longit=longit,WV=wv,AbO3=o3,year=year,month=month,day=day,hour=hour,doy=doy,zenith=zenith,azimuth=azimuth,satelliteZenith=satelliteZenith,satelliteAzimuth=satelliteAzimuth,L=L,bands=bands,IALBDX=1,rho_background=rho_background)

        print('MM topography correction - Richter 1998')
        R=surehyp.atmoCorrection.MM_topo_correction(R,bands,slope*np.pi/180,wazim*np.pi/180,zenith*np.pi/180,azimuth*np.pi/180)

        print('save the reflectance image')
        surehyp.atmoCorrection.saveRimage(R,metadata,pathToOutImage)

    return pathToOut
