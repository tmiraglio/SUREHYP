import numpy as np
from scipy.ndimage.filters import uniform_filter
np.seterr(invalid='ignore')


def getRed(R,W):
    b=[640,680]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getMIR(R,W):
    b=[2100,2200]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getNIR(R,W):
    b=[780,860]
    iMin=np.argmin(np.abs(W-b[0]))
    iMax=np.argmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getGreen(R,W):
    b=[530,590]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getBlue(R,W):
    b=[450,490]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getDeepBlue(R,W):
    b=[400,460]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getDeeperBlue(R,W):
    b=[360,400]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getTeal(R,W):
    b=[480,530]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getSnow(R,W):
    b=[400,600]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getNDVI(R,W):
    NIRb=[780,850]
    redb=[620,660]
    iNirMin=np.nanargmin(np.abs(W-NIRb[0]))
    iNirMax=np.nanargmin(np.abs(W-NIRb[1]))
    iRedMin=np.nanargmin(np.abs(W-redb[0]))
    iRedMax=np.nanargmin(np.abs(W-redb[1]))
    NIR=np.nanmean(R[...,iNirMin:iNirMax+1],axis=-1)
    red=np.nanmean(R[...,iRedMin:iRedMax+1],axis=-1)
    return (NIR-red)/(NIR+red), red, NIR

def getCloud(R,W):
    b=[400,1000]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getBLandsat(R,W,numB):
    dico_b={'1':[500,600],'2':[600,700],'3':[700,800],'4':[800,1100]}
    b=dico_b[numB]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def NDWI_water(R,W):
    NIRb=[790,860]
    greenb=[540,577]
    iNirMin=np.nanargmin(np.abs(W-NIRb[0]))
    iNirMax=np.nanargmin(np.abs(W-NIRb[1]))
    iGreenMin=np.nanargmin(np.abs(W-greenb[0]))
    iGreenMax=np.nanargmin(np.abs(W-greenb[1]))
    NIR=np.nanmean(R[...,iNirMin:iNirMax+1],axis=-1)
    green=np.nanmean(R[...,iGreenMin:iGreenMax+1],axis=-1)
    return (green-NIR)/(NIR+green), green, NIR

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def is_odd(num):
    return num & 0x1

def window_stdev(X, window_size):
    r,c=X.shape
    X+=np.random.rand(r,c)*1e-6 #avoids errors when c2-c1*c1 is too small and sqrt returns a nan by eliminating these small numbers
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)
