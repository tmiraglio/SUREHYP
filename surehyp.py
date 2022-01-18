import numpy as np
import sys
import ee
from functools import partial
from multiprocessing import Pool
import sys

sys.path.append('./func/')
import preprocess
import atmoCorrection

def processImage(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut):
    ee.Initialize()
    
    #f=open(pathOut+fname+'_out.txt','w')
    #sys.stdout=f

    print('concatenate the L1T image')
    preprocess.processImage(fname,pathToL1Timages,pathToL1TimagesFiltered) 

    print('read the L1R image')
    arrayL1R=preprocess.readL1R(pathToL1Rimages+fname+'/',fname)
    
    print('get the L1R image parameters')
    metadata,bands,fwhms=preprocess.getImageMetadata(pathToL1Rimages+fname+'/',fname)

    print('separates VNIR and SWIR')
    VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm=preprocess.separating(arrayL1R,bands,fwhms)
       
    print('converts DN to radiance')
    VNIR,SWIR=preprocess.DN2Radiance(VNIR,SWIR)
    
    print('aligning VNIR and SWIR, part 1')
    VNIR,SWIR=preprocess.alignSWIR2VNIRpart1(VNIR,SWIR)

    print('desmiling')
    VNIR=preprocess.smileCorrectionAll(VNIR,2,check=True)
    SWIR=preprocess.smileCorrectionAll(SWIR,2,check=True)
    
    print('destriping')
    VNIR=preprocess.destriping(VNIR,'VNIR',0.11)
    SWIR=preprocess.destriping(SWIR,'SWIR',0.11)

    print('aligning VNIR and SWIR, part 2')
    VNIR,SWIR=preprocess.alignSWIR2VNIRpart2(VNIR,VNIRb,SWIR,SWIRb)

    print('assemble VNIR and SWIR')
    arrayL1R,wavelengths,fwhms=preprocess.concatenateImages(VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm)

    print('georeference the corrected L1R data using the L1T data')
    arrayL1Rgeoreferenced, metadataGeoreferenced=preprocess.georeferencing(arrayL1R,pathToL1TimagesFiltered,fname)

    print('save the processed image as an ENVI file')
    preprocess.savePreprocessedL1R(arrayL1Rgeoreferenced,wavelengths,fwhms,metadataGeoreferenced,pathToL1Rimages,pathToL1Rmetadata,metadata,fname,pathOut)


    #f.close()        

def atmosphericCorrection(fname,pathOut):
    print('open processed radiance image')
    L,bands,fwhms,longit,latit,datestamp1,datestamp2,zenith,azimuth,satelliteZenith,satelliteAzimuth,scaleFactor,year,month,day,hour,minute,doy,thetaZ,thetaV,UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon,metadata=atmoCorrection.getImageAndParameters(pathOut+fname+'_L1R_complete')
    
    print('get haze spectrum')
    L,Lhaze,DOBJ,bands_dobj=atmoCorrection.darkObjectDehazing(L,bands)

    print('get average elevation of the scene from GEE')
    altit=atmoCorrection.getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon)

    print('get atmosphere content')
    wv,o3=atmoCorrection.getAtmosphericParameters(bands,L,datestamp1,year,month,day,hour,minute,doy,longit,latit,altit,thetaV,thetaZ)
   
    ########################################
    # Atmospheric correction -- flat surface
    print('obtain radiative transfer outputs')
    #get the atmosphere parameters for the sun-ground section using the image acquisition time to determine sun angle
    df=atmoCorrection.runSMARTS(ALTIT=altit,LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=int(hour)+int(minute)/60,SUNCOR=atmoCorrection.get_SUNCOR(doy),IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    #get the atmosphere parameters for the ground-satellite section by setting the 'sun' (in SMARTS) at the satellite zenith position to get the transmittance over the correct optical path length
    df_gs=atmoCorrection.runSMARTS(ALTIT=altit,LATIT=0,LONGIT=0,IMASS=0,SUNCOR=atmoCorrection.get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(thetaV)*180/np.pi,AZIM=0,IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    
    print('compute radiance to reflectance')
    R=atmoCorrection.computeLtoE(L,bands,df,df_gs,thetaV,thetaZ)

    print('save the reflectance image')
    atmoCorrection.saveRimage(R,bands,metadata,pathOut+fname+'_Reflectance')

    ########################################
    #atmospheric correction -- rough terrain
    print('download DEM images from GEE')
    atmoCorrection.getDEMimages(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat)    
 
    print('reproject DEM images')
    im2=rasterio.open('./elev/SRTMGL1_003.elevation.tif')
    reprojectImage(im2,im1.profile['crs'],'./elev/tmp.tif')
    im2=rasterio.open('./slope/download.slope.tif')
    reprojectImage(im2,im1.profile['crs'],'./slope/tmp.tif')
    im2=rasterio.open('./aspect/download.aspect.tif')
    reprojectImage(im2,im1.profile['crs'],'./aspect/tmp.tif')
    
    print("extract the data corresponding to the Hyperion image's pixels")
    im1_transform=rasterio.open(pathOut+fname+'_L1R_complete.img').transform
    rows1,cols1,rows2, cols2=get_target_rows_cols(im1_transform,L,rasterio.open('./elev/tmp.tif')) #elev, slope and aspect all have the same projection
    elev=extractSecondaryData(im1,L,rasterio.open('./elev/tmp.tif'),rows1,cols1,rows2, cols2)
    elev[L[:,:,40]<=0]=np.nan
    elev=elev*1e-3
    slope=extractSecondaryData(im1,L,rasterio.open('./slope/tmp.tif'),rows1,cols1,rows2, cols2)
    slope[L[:,:,40]<=0]=np.nan
    wazim=extractSecondaryData(im1,L,rasterio.open('./aspect/tmp.tif'),rows1,cols1,rows2, cols2)
    wazim[L[:,:,40]<=0]=np.nan

    #define the steps of the LUT
    stepAltit=1
    stepTilt=15 #there is barely any difference in the results using 15 degrees or 7.5 degrees (less than 1% of relative reflectance difference)
    stepWazim=15

    print('computing the LUT for the rough terrain correciton')
    R=getSmartsFactorDem(altitMap=elev,tiltMap=slope,wazimMap=wazim,stepAltit=stepAltit,stepTilt=stepTilt,stepWazim=stepWazim,latit=latit,longit=longit,IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3,year=year,month=month,day=day,hour=hour,doy=doy,thetaZ=thetaZ,satelliteZenith=satelliteZenith,satelliteAzimuth=satelliteAzimuth,L=L,bands=bands)

    print('save the reflectance image')
    atmoCorrection.saveRimage(R,bands,metadata,pathOut+fname+'_Reflectance_topo')


if __name__ == '__main__':
    ee.Initialize()
    
    processes=10

    pathToL1Rmetadata='./METADATA/METADATA.csv'
    
    pathToL1Timages="./L1T/"    

    pathToL1Rimages="./L1R/"
    pathToL1TimagesFiltered="./L1T/filteredImages/"

    pathOut='./OUT/'

    fnames=['EO1H0190262011062110K3']


    #preprocessing
    #with Pool(processes=processes) as pool:
    #    print('pooling the preprocessing')
    #    pool.map(partial(processImage,pathToL1Rmetadata=pathToL1Rmetadata,pathToL1Rimages=pathToL1Rimages,pathToL1Timages=pathToL1Timages,pathToL1TimagesFiltered=pathToL1TimagesFiltered,pathOut=pathOut),fnames)

    for fname in fnames: 
        processImage(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut)

    for fname in fnames:
        atmosphericCorrection(fname,pathOut)
