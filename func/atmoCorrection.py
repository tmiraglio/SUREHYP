import matplotlib.pyplot as plt
import spectral.io.envi as envi
import numpy as np
import ee
import geetools
import os
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy import interpolate
from interp3d import interp_3d
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import richdem as rd
from tqdm.auto import tqdm
import pandas as pd
import subprocess

import various

def runSMARTS(ALTIT=0.3,LATIT=48.1,LONGIT=-79.3,YEAR=2013,MONTH=9,DAY=26,HOUR=10,ITILT=0,TILT=None,WAZIM=None,TAU550=None,IMASS=0,ZENITH=None,AZIM=None,SUNCOR=1,doy=269,ITURB=5,VISI=None,IH2O=1,WV=None,IO3=1,IALT=None,AbO3=None):
    os.environ['SMARTSPATH']='./SMARTS2981-PC_Package/'

    if doy<46:
        col=1
    elif doy<105:
        col=2
    elif doy<166:
        col=3
    elif doy<227:
        col=4
    elif doy<288:
        col=5
    else:
        col=6
    dico_aero={ #From FLAASH user manual
            (8,1):'SAW',(8,2):'SAW',(8,3):'SAW',(8,4):'MLW',(8,5):'MLW',(8,5):'SAW',
            (7,1):'SAW',(7,2):'SAW',(7,3):'MLW',(7,4):'MLW',(7,5):'MLW',(7,5):'SAW',
            (6,1):'MLW',(6,2):'MLW',(6,3):'MLW',(6,4):'SAS',(6,5):'SAS',(6,5):'MLW',
            (5,1):'MLW',(5,2):'MLW',(5,3):'SAS',(5,4):'SAS',(5,5):'SAS',(5,5):'SAS',
            (4,1):'SAS',(4,2):'SAS',(4,3):'SAS',(4,4):'MLS',(4,5):'MLS',(4,5):'SAS',
            (3,1):'MLS',(3,2):'MLS',(3,3):'MLS',(3,4):'TRL',(3,5):'TRL',(3,5):'MLS',
            (2,1):'TRL',(2,2):'TRL',(2,3):'TRL',(2,4):'TRL',(2,5):'TRL',(2,5):'TRL',
            (1,1):'TRL',(1,2):'TRL',(1,3):'TRL',(1,4):'TRL',(1,5):'TRL',(1,5):'TRL',
            (0,1):'TRL',(0,2):'TRL',(0,3):'TRL',(0,4):'TRL',(0,5):'TRL',(0,5):'TRL',
            (-1,1):'TRL',(-1,2):'TRL',(-1,3):'TRL',(-1,4):'TRL',(-1,5):'TRL',(-1,5):'TRL',
            (-2,1):'TRL',(-2,2):'TRL',(-2,3):'TRL',(-2,4):'MLS',(-2,5):'MLS',(-2,5):'TRL',
            (-3,1):'MLS',(-3,2):'MLS',(-3,3):'MLS',(-3,4):'MLS',(-3,5):'MLS',(-3,5):'MLS',
            (-4,1):'SAS',(-4,2):'SAS',(-4,3):'SAS',(-4,4):'SAS',(-4,5):'SAS',(-4,5):'SAS',
            (-5,1):'SAS',(-5,2):'SAS',(-5,3):'SAS',(-5,4):'MLW',(-5,5):'MLW',(-5,5):'SAS',
            (-6,1):'MLW',(-6,2):'MLW',(-6,3):'MLW',(-6,4):'MLW',(-6,5):'MLW',(-6,5):'MLW',
            (-7,1):'MLW',(-7,2):'MLW',(-7,3):'MLW',(-7,4):'MLW',(-7,5):'MLW',(-7,5):'MLW',
            (-8,1):'MLW',(-8,2):'MLW',(-8,3):'MLW',(-8,4):'SAW',(-8,5):'MLW',(-8,5):'MLW',
            }

    CMNT='test'
    ISPR='2'
    SPR=None
    ALTIT=str(ALTIT)
    HEIGHT='0'
    IATMOS='1'
    ATMOS=dico_aero[(np.round(LATIT/10,0),col)]
    LATIT=str(LATIT)
    RH=None
    TAIR=None
    SEASON=None
    TDAY=None
    IH2O=str(IH2O)
    W=str(WV)
    IO3=str(IO3)
    IALT=str(IALT)
    AbO3=str(AbO3)
    IGAS='1'
    ILOAD=None
    ApCH2O=None
    ApCH4=None
    ApCO=None
    ApHNO2=None
    ApHNO3=None
    ApNO=None
    ApNO2=None
    ApNO3=None
    ApO3=None
    ApSO2=None
    qCO2='370'
    ISPCTR='9'
    AEROS='S&F_RURAL'
    ALPHA1=None
    ALPHA2=None
    OMEGL=None
    GG=None
    ITURB=str(ITURB)
    TAU5=None
    BETA=None
    BCHUEP=None
    RANGE=None
    VISI=str(VISI)
    if TAU550==None:
        TAU550=str(np.exp(-3.2755-0.15078*float(ALTIT)))
    IALBDX='17'
    RHOX=None
    ITILT=str(ITILT)
    IALBDG=IALBDX
    TILT=str(TILT)
    WAZIM=str(WAZIM)
    RHOG=None
    WLMN='350'
    WLMX='2600'
    SUNCOR=str(SUNCOR)
    SOLARC='1367'
    IPRT='2'
    WPMN='350'
    WPMX='2600'
    INTVL='1'
    IOUT='1 2 3 4 5 6 7 8 15 16 17 18 19 20 21 27 28 32'
    ICIRC='0'
    SLOPE=None
    APERT=None
    LIMIT=None
    ISCAN='0'
    IFILT=None
    WV1=None
    WV2=None
    STEP=None
    FWHM=None
    ILLUM='0'
    IUV='0'
    IMASS=str(IMASS)
    ZENITH=str(ZENITH)
    AZIM=str(AZIM)
    ELEV=None
    AMASS=None
    YEAR=str(int(YEAR))
    MONTH=str(int(MONTH))
    DAY=str(int(DAY))
    ZONE=(np.abs(LONGIT)-7.5)//15
    if (np.abs(LONGIT)-7.5)<0:
        ZONE=ZONE
    else:
        ZONE+=1
    ZONE=int(np.sign(LONGIT)*ZONE)
    HOUR=float(HOUR)+ZONE
    HOUR=str(HOUR)
    ZONE=str(ZONE)
    LONGIT=str(LONGIT)
    DSTEP=None

    output=smartsAll_original(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, AZIM, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP)

    return output

def smartsAll_original(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, AZIM, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP):

    # Check if SMARTSPATH environment variable exists and change working
    # directory if it does.
    original_wd = None
    if 'SMARTSPATH' in os.environ:
        original_wd = os.getcwd()
        os.chdir(os.environ['SMARTSPATH'])

    try:
        os.remove('smarts298.inp.txt')
    except:
        pass
    try:
        os.remove('smarts298.out.txt')
    except:
        pass
    try:
        os.remove('smarts298.ext.txt')
    except:
        pass
    try:
        os.remove('smarts298.scn.txt')
    except:
        pass

    f = open('smarts298.inp.txt', 'w')

    IOTOT = len(IOUT.split())

    ## Card 1: Comment.
    if len(CMNT)>62:
        CMNT = CMNT[0:61]

    CMNT = CMNT.replace(" ", "_")
    CMNT = "'"+CMNT+"'"
    print('{}' . format(CMNT), file=f)

    ## Card 2: Site Pressure
    print('{}'.format(ISPR), file=f)

    ##Card 2a:
    if ISPR=='0':
       # case '0' #Just input pressure.
        print('{}'.format(SPR), file=f)
    elif ISPR=='1':
        # case '1' #Input pressure, altitude and height.
        print('{} {} {}'.format(SPR, ALTIT, HEIGHT), file=f)
    elif ISPR=='2':
        #case '2' #Input lat, alt and height
        print('{} {} {}'.format(LATIT, ALTIT, HEIGHT), file=f)
    else:
        print("ISPR Error. ISPR should be 0, 1 or 2. Currently ISPR = ", ISPR)

    ## Card 3: Atmosphere model
    print('{}'.format(IATMOS), file=f)

    ## Card 3a:
    if IATMOS=='0':
        #case '0' #Input TAIR, RH, SEASON, TDAY
        print('{} {} {} {}'.format(TAIR, RH, SEASON, TDAY), file=f)
    elif IATMOS=='1':
        #case '1' #Input reference atmosphere
        ATMOS = "'"+ATMOS+"'"
        print('{}'.format(ATMOS), file=f)

    ## Card 4: Water vapor data
    print('{}'.format(IH2O), file=f)

    ## Card 4a
    if IH2O=='0':
        #case '0'
        print('{}'.format(W), file=f)
    elif IH2O=='1':
        #case '1'
        #The subcard 4a is skipped
        pass  #      print("")

    ## Card 5: Ozone abundance
    print('{}'.format(IO3), file=f)

    ## Card 5a
    if IO3=='0':
        #case '0'
        print('{} {}'.format(IALT, AbO3), file=f)
    elif IO3=='1':
        #case '1'
        #The subcard 5a is skipped and default values are used from selected
        #reference atmosphere in Card 3.
        pass #      print("")

    ## Card 6: Gaseous absorption and atmospheric pollution
    print('{}'.format(IGAS), file=f)

    ## Card 6a:  Option for tropospheric pollution
    if IGAS=='0':
        # case '0'
        print('{}'.format(ILOAD), file=f)

        ## Card 6b: Concentration of Pollutants
        if ILOAD=='0':
            #case '0'
            print('{} {} {} {} {} {} {} {} {} {} '.format(ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, ApO3, ApSO2), file=f)
        elif ILOAD=='1':
            #case '1'
                #The subcard 6b is skipped and values of PRISTINE
                #ATMOSPHERIC conditions are assumed
            pass #     print("")
        elif ILOAD=='2' or ILOAD =='3' or ILOAD == '4':
            #case {'2', '3', '4'}
            #The subcard 6b is skipped and value of ILOAD will be used
            #as LIGHT POLLUTION (ILOAD = 2), MODERATE POLLUTION (ILOAD = 3),
            #and SEVERE POLLUTION (ILOAD = 4).
            pass #     print("")

    elif IGAS=='1':
        #case '1'
        #The subcard 6a is skipped, and values are for default average
        #profiles.
        print("")

    ## Card 7:  CO2 columnar volumetric concentration (ppmv)
    print('{}'.format(qCO2), file=f)

    ## Card 7a: Option of proper extraterrestrial spectrum
    print('{}'.format(ISPCTR), file=f)

    ## Card 8: Aerosol model selection out of twelve
    AEROS = "'"+AEROS+"'"

    print('{}'.format(AEROS), file=f)

    ## Card 8a: If the aerosol model is 'USER' for user supplied information
    if AEROS=="'USER'":
        print('{} {} {} {}'.format(ALPHA1, ALPHA2, OMEGL, GG), file=f)
    else:
        #The subcard 8a is skipped
        pass #     print("")

    ## Card 9: Option to select turbidity model
    print('{}'.format(ITURB), file=f)

    ## Card 9a
    if ITURB=='0':
        #case '0'
        print('{}'.format(TAU5), file=f)
    elif ITURB=='1':
        #case '1'
        print('{}'.format(BETA), file=f)
    elif ITURB=='2':
        #case '2'
        print('{}'.format(BCHUEP), file=f)
    elif ITURB=='3':
        #case '3'
        print('{}'.format(RANGE), file=f)
    elif ITURB=='4':
        #case '4'
        print('{}'.format(VISI), file=f)
    elif ITURB=='5':
        #case '5'
        print('{}'.format(TAU550), file=f)
    else:
        print("Error: Card 9 needs to be input. Assign a valid value to ITURB = ", ITURB)

    ## Card 10:  Select zonal albedo
    print('{}'.format(IALBDX), file=f)

    ## Card 10a: Input fix broadband lambertial albedo RHOX
    if IALBDX == '-1':
        print('{}'.format(RHOX), file=f)
    else:
        pass #     print("")
        #The subcard 10a is skipped.

    ## Card 10b: Tilted surface calculation flag
    print('{}'.format(ITILT), file=f)

    ## Card 10c: Tilt surface calculation parameters
    if ITILT == '1':
        print('{} {} {}'.format(IALBDG, TILT, WAZIM), file=f)

        ##Card 10d: If tilt calculations are performed and zonal albedo of
        ##foreground.
        if IALBDG == '-1':
            print('{}'.format(RHOG), file=f)
        else:
            pass #     print("")
            #The subcard is skipped


    ## Card 11: Spectral ranges for calculations
    print('{} {} {} {}'.format(WLMN, WLMX, SUNCOR, SOLARC), file=f)

    ## Card 12: Output selection.
    print('{}'.format(IPRT), file=f)

    ## Card 12a: For spectral results (IPRT >= 1)
    if float(IPRT) >= 1:
        print('{} {} {}'.format(WPMN, WPMX, INTVL), file=f)

        ## Card 12b & Card 12c:
        if float(IPRT) == 2 or float(IPRT) == 3:
            print('{}'.format(IOTOT), file=f)
            print('{}'.format(IOUT), file=f)
        else:
            pass #     print("")
            #The subcards 12b and 12c are skipped.
    else:
        pass #     print("")
        #The subcard 12a is skipped

    ## Card 13: Circumsolar calculations
    print('{}'.format(ICIRC), file=f)

    ## Card 13a:  Simulated radiometer parameters
    if ICIRC == '1':
        print('{} {} {}'.format(SLOPE, APERT, LIMIT), file=f)
    else:
        pass #     print("")
        #The subcard 13a is skipped since no circumsolar calculations or
        #simulated radiometers have been requested.


    ## Card 14:  Scanning/Smoothing virtual filter postprocessor
    print('{}'.format(ISCAN), file=f)

    ## Card 14a:  Simulated radiometer parameters
    if ISCAN == '1':
        print('{} {} {} {} {}'.format(IFILT, WV1, WV2, STEP, FWHM), file=f)
    else:
        pass #     print("")
        #The subcard 14a is skipped since no postprocessing is simulated.

    ## Card 15: Illuminace, luminous efficacy and photosythetically active radiarion calculations
    print('{}'.format(ILLUM), file=f)

    ## Card 16: Special broadband UV calculations
    print('{}'.format(IUV), file=f)

    ## Card 17:  Option for solar position and air mass calculations
    print('{}'.format(IMASS), file=f)

    ## Card 17a: Solar position parameters:
    if IMASS=='0':
        #case '0' #Enter Zenith and Azimuth of the sun
        print('{} {}'.format(ZENITH, AZIM), file=f)
    elif IMASS=='1':
        #case '1' #Enter Elevation and Azimuth of the sun
        print('{} {}'.format(ELEV, AZIM), file=f)
    elif IMASS=='2':
        #case '2' #Enter air mass directly
        print('{}'.format(AMASS), file=f)
    elif IMASS=='3':
        #case '3' #Enter date, time and latitude
        print('{} {} {} {} {} {} {}'.format(YEAR, MONTH, DAY, HOUR, LATIT, LONGIT, ZONE), file=f)
    elif IMASS=='4':
        #case '4' #Enter date and time and step in min for a daily calculation.
        print('{}, {}, {}'.format(MONTH, LATIT, DSTEP), file=f)

    ## Input Finalization
    print('', file=f)
    f.close()

    ## Run SMARTS 2.9.5
    #dump = os.system('smarts295bat.exe')
    commands = ['smarts2981_PC_64bit.exe']
    command = None
    for cmd in commands:
        if os.path.exists(cmd):
            command = cmd
            break

    if not command:
        print('Could not find SMARTS2 executable.')
        data = None
    else:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=open("output.txt", "w"), shell=True)
        p.wait()

        ## Read SMARTS 2.9.5 Output File
        data = pd.read_csv('smarts298.ext.txt', delim_whitespace=True)

    try:
        os.remove('smarts298.inp.txt')
    except:
        pass #     print("")
    try:
        os.remove('smarts298.out.txt')
    except:
        pass #     print("")
    try:
        os.remove('smarts298.ext.txt')
    except:
        pass #     print("")
    try:
        os.remove('smarts298.scn.txt')
    except:
        pass #     print("")

    # Return to original working directory.
    if original_wd:
        os.chdir(original_wd)



    return data

def getGEEdate(datestamp1,year,doy,longit,latit):
    numPixels=1

    WTR = ee.ImageCollection('NCEP_RE/surface_wv') #kg/m2
    WTR=WTR.select('pr_wtr').filterDate(datestamp1+'T17:30',datestamp1+'T18:30')
    coord=ee.Geometry.Point(longit,latit)
    wv=WTR.first().sample(region=coord,numPixels=numPixels).getInfo()['features'][0]['properties']['pr_wtr']#.get('air').getInfo()
    wv=wv*1E-1 #g.cm-2

    year=int(year)
    doy=int(doy)
    i=1
    while True:
        try:
            O3=ee.ImageCollection('TOMS/MERGED') #dobson
            O3=O3.select('ozone').filter(ee.Filter.calendarRange(year,year,'year'))
            O3=O3.filter(ee.Filter.dayOfYear(doy-i,doy+i))
            coord=ee.Geometry.Point(longit,latit)
            o3=O3.first().sample(region=coord,numPixels=numPixels).getInfo()['features'][0]['properties']['ozone']#.get('air').getInfo()
            break
        except:
            i+=1
    o3=o3*1E-3 # atm-cm

    return wv, o3

def getImageAndParameters(path):
    img=envi.open(path+'.hdr',path+'.img') #img in uW.cm-2.nm-1.sr-1, with a scaleFactor

    bands=np.asarray(img.metadata['wavelength']).astype(float)
    fwhms=np.asarray(img.metadata['fwhm']).astype(float)
    longit=float(img.metadata['center lon'])
    latit=float(img.metadata['center lat'])
    datestamp1=img.metadata['acquisition date']
    datestamp2=img.metadata['acquisition time start']
    zenith=float(img.metadata['sun zenith'])
    azimuth=float(img.metadata['sun azimuth'])
    satelliteZenith=float(img.metadata['satellite zenith'])
    satelliteAzimuth=float(img.metadata['satellite azimuth'])-90
    scaleFactor=float(img.metadata['scale factor'])

    UL_lat=float(img.metadata['ul_lat'])
    UL_lon=float(img.metadata['ul_lon'])
    UR_lat=float(img.metadata['ur_lat'])
    UR_lon=float(img.metadata['ur_lon'])
    LL_lat=float(img.metadata['ll_lat'])
    LL_lon=float(img.metadata['ll_lon'])
    LR_lat=float(img.metadata['lr_lat'])
    LR_lon=float(img.metadata['lr_lon'])

    year=datestamp1[:4]
    month=datestamp1[5:7]
    day=datestamp1[8:]
    hour=datestamp2[9:11]
    minute=datestamp2[12:14]
    doy=datestamp2[5:8]

    thetaZ=zenith*np.pi/180
    thetaV=satelliteZenith*np.pi/180

    metadata=img.metadata.copy()

    L=img[:,:,:]/scaleFactor #image in uW.cm-2.nm-1.sr-1 with scalefactor
    return L,bands,fwhms,longit,latit,datestamp1,datestamp2,zenith,azimuth,satelliteZenith,satelliteAzimuth,scaleFactor,year,month,day,hour,minute,doy,thetaZ,thetaV,UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon,metadata

def getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon):
    dem = ee.Image('USGS/SRTMGL1_003');
    dem=dem.select('elevation')
    region=ee.Geometry.Polygon([[[UL_lon,UL_lat],[UR_lon,UR_lat],[LR_lon,LR_lat],[LL_lon,LL_lat]]],None,False)
    elev=dem.reduceRegion(reducer=ee.Reducer.mean(),geometry=region,scale=30)
    return elev.getInfo()['elevation']*1e-3

def getWaterVapor(bands,L,altit,latit,longit,imass,year,month,day,hour,doy,thetaV,thetaZ,io3,ialt,o3):
    #remove water pixels to keep only land surfaces
    L=L[L[:,:,0]!=0]
    L=L[L[:,np.argmin(np.abs(bands-940))]>0.25]
    L=np.mean(L,axis=0)

    #bands at 820,940 and 1120
    l940=[860,880]
    r940=[990,1120]
    l1120=[1060,1090]
    r1120=[1170,1200]

    ratios940=[]
    ratios1120=[]
    wvOut=[]

    for wv in np.concatenate((np.arange(0,1.5,0.2),np.arange(1.5,4,0.5),np.arange(4,12,1))):
        df=runSMARTS(ALTIT=altit,LATIT=latit,LONGIT=longit,IMASS=imass,YEAR=year,MONTH=month,DAY=day,HOUR=hour,SUNCOR=get_SUNCOR(doy),IH2O=0,WV=wv,IO3=io3,IALT=ialt,AbO3=o3)
        df_gs=runSMARTS(ALTIT=altit,LATIT=0,LONGIT=0,IMASS=0,SUNCOR=get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(thetaV)*180/np.pi,AZIM=0.1,IH2O=0,WV=wv,IO3=io3,IALT=ialt,AbO3=o3)

        W=df['Wvlgth']
        E=df['Extraterrestrial_spectrm']
        Dft=df['Total_Diffuse_tilt_irrad']
        T=df['Direct_rad_transmittance']
        T_gs=df_gs['Direct_rad_transmittance']

        Lout=T_gs*(T*E+Dft)*5
        Lout[W<=1700]=gaussian_filter(Lout[W<=1700],various.fwhm2sigma(10))
        Lout[W>1700]=gaussian_filter(Lout[W>1700],various.fwhm2sigma(2))

        lShoulder940=np.mean(Lout[np.logical_and(W>=l940[0],W<=l940[1])])
        rShoulder940=np.mean(Lout[np.logical_and(W>=r940[0],W<=r940[1])])
        btm940=Lout[np.argmin(np.abs(W-940))]
        ratios940.append(btm940/np.mean([lShoulder940,rShoulder940]))

        lShoulder1120=np.mean(Lout[np.logical_and(W>=l1120[0],W<=l1120[1])])
        rShoulder1120=np.mean(Lout[np.logical_and(W>=r1120[0],W<=r1120[1])])
        btm1120=Lout[np.argmin(np.abs(W-1120))]
        ratios1120.append(btm1120/np.mean([lShoulder1120,rShoulder1120]))

        wvOut.append(wv)
    ratios940=np.asarray(ratios940)
    ratios1120=np.asarray(ratios1120)
    wvOut=np.asarray(wvOut)

    #for image
    lShoulder940=np.mean(L[np.logical_and(bands>=l940[0],bands<=l940[1])])
    rShoulder940=np.mean(L[np.logical_and(bands>=r940[0],bands<=r940[1])])
    btm940=L[np.argmin(np.abs(bands-940))]
    ratio940=btm940/np.mean([lShoulder940,rShoulder940])

    lShoulder1120=np.mean(L[np.logical_and(bands>=l1120[0],bands<=l1120[1])])
    rShoulder1120=np.mean(L[np.logical_and(bands>=r1120[0],bands<=r1120[1])])
    btm1120=L[np.argmin(np.abs(bands-1120))]
    ratio1120=btm1120/np.mean([lShoulder1120,rShoulder1120])

    if btm1120>=0.15:
        f=interpolate.interp1d(ratios1120,wvOut)
        wvImg=f(ratio1120)
    else:
        f=interpolate.interp1d(ratios940,wvOut)
        wvImg=f(ratio940)

    return wvImg

def darkObjectDehazing(L,bands):
    Ltmp=L[L[:,:,0]>0,:]
    DOBJ=np.amin(Ltmp,axis=0)
    DOBJ=smoothing(DOBJ,5,1)
    bands_dobj=bands
    c=-0.5
    bref=bands[np.logical_and(bands>410,bands<480)][np.argmax(DOBJ[np.logical_and(bands>410,bands<480)])]
    deepB=np.amax(DOBJ[np.logical_and(bands>410,bands<480)])
    Lhaze=(bands*1E-3)**c
    offset=deepB-Lhaze[np.argmin(np.abs(bands-bref))]
    Lhaze+=offset

    while (np.amin(DOBJ[np.logical_and(bands_dobj>bref,bands<1500)]-Lhaze[np.logical_and(bands_dobj>bref,bands<1500)])<0) or (Lhaze[np.argmin(np.abs(bands-2150))]>0.05):
        c-=0.001
        Lhaze=(bands*1E-3)**c
        offset=deepB-Lhaze[np.argmin(np.abs(bands-bref))]
        Lhaze+=offset
    Lhaze[Lhaze<0]=0
    L=L-Lhaze
    L[L<=0]=0
    return L,Lhaze,DOBJ,bands_dobj

def get_SUNCOR(doy):
    #compute the sun-earth distance correction factor depending on the day of year
    return 1-0.0335*np.sin(360*(int(doy)-94)/365*np.pi/180)

def smoothing(R,width=3,order=1,processO2=False,bands=None):
    if processO2==True: #remove band values from ]750,780[ and replaces them by values interpolated used the two previous and next R values
        argb=np.zeros(bands.shape).astype(int)
        argb[np.logical_and(bands>=730,bands<=750)]=1
        argb[np.logical_and(bands>=780,bands<=800)]=1
        Rtmp=R[R[:,:,0]>0,:]
        f=interp1d(bands[argb==1],Rtmp[...,argb==1], kind='cubic')
        Rtmp[:,:,argb]=f(Rtmp[...,argb==1])
        R[R[:,:,0]>0,:]=Rtmp

    return savgol_filter(R,width,order)

def computeLtoEfactor(df,df_gs,thetaV,thetaZ):
    #compute the factor to convert TOA radiance to surface reflectance
    W=df['Wvlgth'].values
    E=df['Extraterrestrial_spectrm']
    T=df['RayleighScat_trnsmittnce']*df['Ozone_totl_transmittance']*df['Trace_gas__transmittance']*df['WaterVapor_transmittance']*df['Mixed_gas__transmittance']*df['Aerosol_tot_transmittnce']
    T_gs=df_gs['RayleighScat_trnsmittnce']*df_gs['Ozone_totl_transmittance']*df_gs['Trace_gas__transmittance']*df_gs['WaterVapor_transmittance']*df_gs['Mixed_gas__transmittance']*df_gs['Aerosol_tot_transmittnce']
    Dnt=df['Direct_normal_irradiance']
    Dtt=df['Direct_tilted_irradiance']
    Dft=df['Total_Diffuse_tilt_irrad']
    Dgt=df['Global_tilted_irradiance']

    factor=np.pi/T_gs/Dgt
    factor[W<=1700]=gaussian_filter(factor[W<=1700],various.fwhm2sigma(10))
    factor[W>1700]=gaussian_filter(factor[W>1700],various.fwhm2sigma(2))

    return factor,W,E,Dtt,Dft,T,T_gs

def getAtmosphericParameters(bands,L,datestamp1,year,month,day,hour,minute,doy,longit,latit,altit,thetaV,thetaZ,imass=3,io3=0,ialt=0,o3=3):
    #obtain some atmospheric parameters using GEE
    print('get water vapor and ozone from GEE')
    wvGEE,o3=getGEEdate(datestamp1,year,doy,longit,latit)

    print('get water vapor from the radiance image')
    wv=getWaterVapor(bands,L,altit,latit,longit,imass,year,month,day,int(hour)+int(minute)/60,doy,thetaV,thetaZ,io3,ialt,o3)

    if (wv>12) or (np.isnan(wv)):
        wv=wvGEE
    return wv,o3

def computeLtoE(L,bands,df,df_gs,thetaV,thetaZ):
    #get the factor to convert TOA radiance to surface reflectance and return the reflectance
    factor,W,E,Dtt,Dft,T,T_gs=computeLtoEfactor(df,df_gs,thetaV,thetaZ)
    fun=interpolate.interp1d(W,factor)
    factor=fun(bands)
    R=factor*L
    return R

def saveRimage(R,bands,metadata,pathOut,scaleFactor=1e5):
    scale=1E4*np.ones(bands.shape)
    metadata['scale factor']=scale.tolist()
    R=np.round(R*scaleFactor,0).astype(int)
    R[R>1e7]=1e7
    R[R<0]=0
    envi.save_image(pathOut+fname+'_Reflectance.hdr',R,metadata=metadata,force=True)

def getTOAreflectanceFactor(bands,latit,longit,year,month,day,hour,doy,thetaV):
    #compute TOA reflectance
    df=runSMARTS(ALTIT=0,LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=hour,SUNCOR=get_SUNCOR(doy),IH2O=0,WV=0,IO3=0,IALT=0,AbO3=0)
    W=df['Wvlgth'].values
    E=df['Extraterrestrial_spectrm'].values
    factor=np.pi/(E*np.cos(thetaV))
    factor[W<=1700]=gaussian_filter(factor[W<=1700],various.fwhm2sigma(10))
    factor[W>1700]=gaussian_filter(factor[W>1700],various.fwhm2sigma(2))
    f=interpolate.interp1d(W,factor)
    factor=f(bands)
    return factor

def cirrusRemoval(bands,L,latit,longit,year,month,day,hour,doy,thetaV):
    #uses the method presented by Gao et al 1997, 2017 to remove cirrus effects
    #compute TOA reflectance, performs the cirrus removal, and retransforms to TOA radiance
    #returns the cirrus-removed TOA radiance
    factor=getTOAreflectanceFactor(bands,latit,longit,year,month,day,hour,doy,thetaV)
    R=factor*L
    R[L<=0]=0

    Rcirrus=R[:,:,np.argmin(np.abs(bands-1380))]
    r1380=Rcirrus[Rcirrus>0]
    rlambda=R[Rcirrus>0,:]

    xs=[]
    ys=[]
    for i in np.arange(np.floor(np.amin(r1380)),np.ceil(np.amax(r1380)),0.5):
        cond=np.logical_and(r1380>=i,r1380<i+1)
        tmp1380=r1380[cond]
        tmpLambda=rlambda[cond,:]
        if not len(tmpLambda)==0:
            xs.append(np.amin(tmpLambda,axis=0))
            ys.append(tmp1380[np.argmin(tmpLambda,axis=0)])
    xs=np.asarray(xs)
    ys=np.asarray(ys)

    Ka=[]
    for b in np.arange(xs.shape[1]):
        xs_trim=xs[:,b].copy()
        ys_trim=ys[:,b].copy()
        while (np.amin(np.abs(np.diff(xs_trim)))<0.01): #remove 'vertical' points
            ys_trim=np.delete(ys_trim,np.argmin(np.abs(np.diff(xs_trim)))+1)
            xs_trim=np.delete(xs_trim,np.argmin(np.abs(np.diff(xs_trim)))+1)
            if xs_trim.size==1:
                break
        try:
            ys_trim=ys_trim[1:]
            xs_trim=xs_trim[1:]
            p=np.polyfit(xs_trim,ys_trim,1)
            Ka.append(p[0])
        except:
            Ka.append(1e10)
    Ka=np.asarray(Ka)
    Rcirrus=np.moveaxis(np.tile(Rcirrus,(xs.shape[1],1,1)),0,2)
    Rcorr=R-Rcirrus/Ka

    Lcorr=Rcorr/factor
    Lcorr[L<=0]=0
    return Lcorr
    envi.save_image(pathOut+'.hdr',R,metadata=metadata)

def getDEMimages(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat):
    elev = ee.Image('USGS/SRTMGL1_003');
    elev=elev.select('elevation')

    region=ee.Geometry.Polygon([[[UL_lon-0.05,UL_lat+0.05],[UR_lon+0.05,UR_lat+0.05],[LR_lon+0.05,LR_lat-0.05],[LL_lon-0.05,LL_lat-0.05]]],None,False)

    elev=elev.clip(region)
    geetools.batch.image.toLocal(elev,'elev',scale=30,region=region)

    for f in os.listdir('.'):
        if '.zip' in f:
            os.remove(os.path.join('.',f))


def reprojectImage(im,dst_crs,pathOut):
    with im as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1
        })

        with rasterio.open(pathOut, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                num_threads=10,
                warp_mem_limit=1024)
        dst.close()

def get_target_rows_cols(T0,array1,imSecondary):
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array1.shape[1]), np.arange(array1.shape[0]))

    cols=cols[array1[:,:,40]>=0]
    rows=rows[array1[:,:,40]>=0]
    xs,ys=rasterio.transform.xy(T0,rows,cols)

    #get coordinates row/cols in the landcover map
    rowsSecondary, colsSecondary = rasterio.transform.rowcol(imSecondary.transform, xs,ys)

    rowsSecondary=np.asarray(rowsSecondary)
    colsSecondary=np.asarray(colsSecondary)
    return rows,cols,rowsSecondary, colsSecondary

def extractSecondaryData(array1,array2,rows,cols,rowsSecondary,colsSecondary):
    #remove OOB values
    rows=rows[rowsSecondary>0]
    cols=cols[rowsSecondary>0]
    colsSecondary=colsSecondary[rowsSecondary>0]
    rowsSecondary=rowsSecondary[rowsSecondary>0]
    rows=rows[colsSecondary>0]
    cols=cols[colsSecondary>0]
    rowsSecondary=rowsSecondary[colsSecondary>0]
    colsSecondary=colsSecondary[colsSecondary>0]
    rows=rows[rowsSecondary<array2.shape[0]]
    cols=cols[rowsSecondary<array2.shape[0]]
    colsSecondary=colsSecondary[rowsSecondary<array2.shape[0]]
    rowsSecondary=rowsSecondary[rowsSecondary<array2.shape[0]]
    rows=rows[colsSecondary<array2.shape[1]]
    rowsSecondary=rowsSecondary[colsSecondary<array2.shape[1]]
    cols=cols[colsSecondary<array2.shape[1]]
    colsSecondary=colsSecondary[colsSecondary<array2.shape[1]]

    arraySecondaryNew=np.zeros((array1.shape[0],array1.shape[1]))
    arraySecondaryNew[rows,cols]=array2[rowsSecondary,colsSecondary].copy()
    arraySecondaryNew=np.reshape(arraySecondaryNew,(array1.shape[0],array1.shape[1]))
    return arraySecondaryNew

def getSmartsFactorDem(altitMap,tiltMap,wazimMap,stepAltit,stepTilt,stepWazim,latit,longit,IH2O,WV,IO3,IALT,AbO3,year,month,day,hour,doy,thetaZ,satelliteZenith,satelliteAzimuth,L,bands):

    #prepare the iteration vectors for the LUT building
    ALTITS=np.arange(np.floor(np.nanmin(altitMap)),np.maximum(np.ceil(np.nanmax(altitMap))+stepAltit,np.floor(np.nanmin(altitMap))+2*stepAltit),stepAltit)
    TILTS=np.arange(np.floor(np.nanmin(tiltMap)),np.maximum(np.ceil(np.nanmax(tiltMap))+stepTilt,np.floor(np.nanmin(tiltMap))+2*stepTilt),stepTilt) #decimal degree
    WAZIMS=np.arange(np.floor(np.nanmin(wazimMap)),np.maximum(np.ceil(np.nanmax(wazimMap))+stepWazim,np.floor(np.nanmin(wazimMap))+stepWazim),stepWazim)

    #first dry run to get W
    df=runSMARTS(ALTIT=0,ITILT='1',TILT=0,WAZIM=0,LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=hour,SUNCOR=get_SUNCOR(doy))
    W=df['Wvlgth'].values

    points = (ALTITS, TILTS, WAZIMS)
    xv,yv,zv=np.meshgrid(*points,indexing='ij')
    data=np.zeros((xv.shape[0],xv.shape[1],xv.shape[2],len(W)))

    for i in tqdm(np.arange(xv.shape[0]),desc='ALTITS'):
        for j in tqdm(np.arange(xv.shape[1]),desc='TILTS '):
            for k in tqdm(np.arange(xv.shape[2]),desc='WAZIMS'):
                df=runSMARTS(ALTIT=xv[i,j,k],ITILT='1',TILT=yv[i,j,k],WAZIM=zv[i,j,k],LATIT=latit,LONGIT=longit,IMASS=3,YEAR=year,MONTH=month,DAY=day,HOUR=hour,SUNCOR=get_SUNCOR(doy),IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3)
                df_gs=runSMARTS(ALTIT=xv[i,j,k],ITILT='1',TILT=yv[i,j,k],WAZIM=zv[i,j,k],LATIT=0,LONGIT=0,IMASS=0,SUNCOR=get_SUNCOR(doy),ITURB=5,ZENITH=satelliteZenith,AZIM=satelliteAzimuth,IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3)

                Dgt=df['Global_tilted_irradiance']
                T_gs=df_gs['Direct_rad_transmittance']
                tmp=np.pi/T_gs/(Dgt)
                tmp[W<=1700]=gaussian_filter1d(tmp[W<=1700],various.fwhm2sigma(10))
                tmp[W>1700]=gaussian_filter1d(tmp[W>1700],various.fwhm2sigma(2))
                data[i,j,k,:]=tmp
    W=df['Wvlgth'].values

    #interpFunction = interpolate.RegularGridInterpolator((ALTITS, TILTS, WAZIMS), data)
    interpFunction = interp_3d.Interp3D(data,ALTITS,TILTS,WAZIMS)

    data=None

    L[L<0]=0
    L[np.isnan(L)]=0
    idx=np.argwhere(L[:,:,40].flatten()>0).squeeze()

    R=np.zeros((L.shape[0]*L.shape[1],L.shape[2])).squeeze()
    Lflat=L.reshape((L.shape[0]*L.shape[1],L.shape[2]))
    for idxx in np.array_split(idx,100):
        values=np.stack((altitMap.flatten()[idxx],tiltMap.flatten()[idxx],wazimMap.flatten()[idxx]),axis=-1)
        M=interpFunction(values)
        f=interpolate.interp1d(W,M)
        R[idxx,:]=f(bands)*Lflat[idxx,:]
    R=R.reshape(L.shape)
    return R

def reprojectDEM(path_im1,path_elev='./elev/SRTMGL1_003.elevation.tif',path_elev_out='./elev/tmp.tif'):
    im1=rasterio.open(path_im1)

    im2=rasterio.open(path_elev)
    reprojectImage(im2,im1.profile['crs'],path_elev_out)

def extractDEMdata(pathToIm1,path_elev='./elev/tmp.tif'):
    im1=rasterio.open(pathToIm1)
    im1_transform=im1.transform
    ar1=im1.read()
    ar1=np.moveaxis(ar1,0,2)
    rows1,cols1,rows2, cols2=get_target_rows_cols(im1_transform,ar1,rasterio.open('./elev/tmp.tif')) #elev, slope and aspect all have the same projection
    elev=extractSecondaryData(ar1,np.squeeze(rasterio.open(path_elev).read()),rows1,cols1,rows2, cols2)
    elev[ar1[:,:,40]<=0]=np.nan
    elev=elev*1e-3

    slope=rd.TerrainAttribute(rd.LoadGDAL(path_elev),attrib='slope_degrees')
    slope=extractSecondaryData(ar1,slope,rows1,cols1,rows2, cols2)
    slope[ar1[:,:,40]<=0]=np.nan

    wazim=rd.TerrainAttribute(rd.LoadGDAL(path_elev),attrib='aspect')
    wazim=extractSecondaryData(ar1,wazim,rows1,cols1,rows2, cols2)
    wazim[ar1[:,:,40]<=0]=np.nan

    fig,ax=plt.subplots(1,3)
    ax[0].imshow(elev)
    ax[1].imshow(slope)
    ax[2].imshow(wazim)

    return elev, slope, wazim
