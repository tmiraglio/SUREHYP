import numpy as np
from scipy.ndimage.filters import uniform_filter
np.seterr(invalid='ignore')


def getRed(R,W):
    redb=[640,680]
    iRedMin=np.nanargmin(np.abs(W-redb[0]))
    iRedMax=np.nanargmin(np.abs(W-redb[1]))
    red=np.nanmean(R[...,iRedMin:iRedMax+1],axis=-1)
    return red, np.mean(redb)

def getMIR(R,W):
    mirb=[2100,2200]
    iMirMin=np.nanargmin(np.abs(W-mirb[0]))
    iMirMax=np.nanargmin(np.abs(W-mirb[1]))
    mir=np.nanmean(R[...,iMirMin:iMirMax+1],axis=-1)
    return mir, np.mean(mirb)

def getNIR(R,W):
    mirb=[780,860]
    iMirMin=np.argmin(np.abs(W-mirb[0]))
    iMirMax=np.argmin(np.abs(W-mirb[1]))
    mir=np.nanmean(R[...,iMirMin:iMirMax+1],axis=-1)
    return mir, np.mean(mirb)

def getGreen(R,W):
    greenb=[530,590]
    iGreenMin=np.nanargmin(np.abs(W-greenb[0]))
    iGreenMax=np.nanargmin(np.abs(W-greenb[1]))
    green=np.nanmean(R[...,iGreenMin:iGreenMax+1],axis=-1)
    return green, np.mean(greenb)

def getBlue(R,W):
    blueb=[450,490]
    iBlueMin=np.nanargmin(np.abs(W-blueb[0]))
    iBlueMax=np.nanargmin(np.abs(W-blueb[1]))
    blue=np.nanmean(R[...,iBlueMin:iBlueMax+1],axis=-1)
    return blue, np.mean(blueb)

def getDeepBlue(R,W):
    blueb=[400,460]
    iBlueMin=np.nanargmin(np.abs(W-blueb[0]))
    iBlueMax=np.nanargmin(np.abs(W-blueb[1]))
    blue=np.nanmean(R[...,iBlueMin:iBlueMax+1],axis=-1)
    return blue, np.mean(blueb)

def getDeeperBlue(R,W):
    blueb=[360,400]
    iBlueMin=np.nanargmin(np.abs(W-blueb[0]))
    iBlueMax=np.nanargmin(np.abs(W-blueb[1]))
    blue=np.nanmean(R[...,iBlueMin:iBlueMax+1],axis=-1)
    return blue, np.mean(blueb)

def getTeal(R,W):
    tealb=[480,530]
    iTealMin=np.nanargmin(np.abs(W-tealb[0]))
    iTealMax=np.nanargmin(np.abs(W-tealb[1]))
    teal=np.nanmean(R[...,iTealMin:iTealMax+1],axis=-1)
    return teal, np.mean(tealb)

def getSnow(R,W):
    tealb=[400,600]
    iTealMin=np.nanargmin(np.abs(W-tealb[0]))
    iTealMax=np.nanargmin(np.abs(W-tealb[1]))
    teal=np.nanmean(R[...,iTealMin:iTealMax+1],axis=-1)
    return teal, np.mean(tealb)

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

def getVisible(R,W):
    tealb=[400,1000]
    iTealMin=np.nanargmin(np.abs(W-tealb[0]))
    iTealMax=np.nanargmin(np.abs(W-tealb[1]))
    teal=np.nanmean(R[...,iTealMin:iTealMax+1],axis=-1)
    return teal, np.mean(tealb)

def NDWI_water(R,W):
    NIRb=[780,860]
    greenb=[492,577]
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
