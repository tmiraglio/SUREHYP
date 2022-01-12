import numpy as np
import sys
import ee
from functools import partial
from multiprocessing import Pool

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


    f.close()        

def atmosphericCorrection(fname,pathOut):
    print('open processed radiance image')
    L,bands,fwhms,longit,latit,datestamp1,datestamp2,zenith,azimuth,satelliteZenith,scaleFactor,year,month,day,hour,minute,doy,thetaZ,thetaV,UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon,metadata=atmoCorrection.getImageAndParameters(pathOut+fname+'_L1R_complete')

    print('get average elevation of the scene from GEE')
    altit=atmoCorrection.getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon)

    print('get atmosphere content')
    wv,o3=atmoCorrection.getAtmosphericParameters(bands,L,datestamp1,year,month,day,hour,minute,doy,longit,latit,altit,thetaV,thetaZ)
   
    print('obtain radiative transfer outputs')
    df=atmoCorrection.runSMARTS(ALTIT=altit,LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=int(hour)+int(minute)/60,SUNCOR=atmoCorrection.get_SUNCOR(doy),IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    df_gs=atmoCorrection.runSMARTS(ALTIT=altit,LATIT=0,LONGIT=0,IMASS=0,SUNCOR=atmoCorrection.get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(thetaV)*180/np.pi,AZIM=0.1,IH2O=0,WV=wv,IO3=0,IALT=0,AbO3=o3)
    
    print('get haze spectrum')
    L,Lhaze,DOBJ,bands_dobj=atmoCorrection.darkObjectDehazing(L,bands)
    
    print('compute radiance to reflectance')
    R=atmoCorrection.computeLtoE(L,bands,df,df_gs,thetaV,thetaZ)

    print('save the reflectance image')
    atmoCorrection.saveRimage(R,bands,metadata,pathOut,fname)


if __name__ == '__main__':
    ee.Initialize()
    
    processes=10

    pathToL1Rmetadata='./METADATA/METADATA.csv'
    
    pathToL1Timages="./L1T/"    

    pathToL1Rimages="./L1R/"
    pathToL1TimagesFiltered="./L1T/filteredImages/"

    pathOut='./OUT/'

    #fnames=['EO1H0430332016170110KF']
    #fnames=['EO1H0430332015288110KF']
    #fnames=['EO1H0430332014136110P3']
    #fnames=['EO1H0430332015166110KF']
    #fnames=['EO1H0430332013149110K4']
    fnames=['EO1H0190262011062110K3']


    #preprocessing
    #with Pool(processes=processes) as pool:
    #    print('pooling the preprocessing')
    #    pool.map(partial(processImage,pathToL1Rmetadata=pathToL1Rmetadata,pathToL1Rimages=pathToL1Rimages,pathToL1Timages=pathToL1Timages,pathToL1TimagesFiltered=pathToL1TimagesFiltered,pathOut=pathOut),fnames)

    for fname in fnames: 
        processImage(fname,pathToL1Rmetadata,pathToL1Rimages,pathToL1Timages,pathToL1TimagesFiltered,pathOut)

    for fname in fnames:
        atmosphericCorrection(fname,pathOut)
