import numpy as np
from scipy.ndimage.filters import uniform_filter
np.seterr(invalid='ignore')


def getRed(R,W):
    '''
    Computes average reflectance of the red spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[640,680]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getMIR(R,W):
    '''
    Computes average reflectance of the MIR spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[2100,2200]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getNIR(R,W):
    '''
    Computes average reflectance of the NIR spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[780,860]
    iMin=np.argmin(np.abs(W-b[0]))
    iMax=np.argmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getGreen(R,W):
    '''
    Computes average reflectance of the green spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[530,590]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getBlue(R,W):
    '''
    Computes average reflectance of the blue spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[450,490]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getDeepBlue(R,W):
    '''
    Computes average reflectance of the deeperr blue spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[400,460]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getDeeperBlue(R,W):
    '''
    Computes average reflectance of an even deeper blue spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[360,400]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getTeal(R,W):
    '''
    Computes average reflectance of the teal spectral range
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[480,530]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getSnow(R,W):
    '''
    Computes average reflectance over a spectral range where reflectance is supposed to be high for snow
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[400,600]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getNDVI(R,W):
    '''
    Computes the NDVI
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            (NIR-red)/(NIR+red): NDVI (m,n)
            red: reflectance in the red region (m,n)
            NIR: reflectance in the NIR region (m,n)
    '''
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
    '''
    Computes average reflectance over a spectral range where reflectance is supposed to be high for clouds
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    b=[400,1000]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def getBLandsat(R,W,numB):
    '''
    Computes the reflectance over a spectral ranges corresponding o Landsat bands
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
            numB: Landsat band
        Returns:
            r: reflectance (m,n)
            np.mean(b): central wavelength corresponding to the reflectance
    '''
    dico_b={'1':[500,600],'2':[600,700],'3':[700,800],'4':[800,1100]}
    b=dico_b[numB]
    iMin=np.nanargmin(np.abs(W-b[0]))
    iMax=np.nanargmin(np.abs(W-b[1]))
    r=np.nanmean(R[...,iMin:iMax+1],axis=-1)
    return r, np.mean(b)

def NDWI_water(R,W):
    '''
    Computes the NDWI
        Parameters:
            R: reflectance array (m,n,b)
            W: wavelengths (b)
        Returns:
            (green-NIR)/(NIR+green): NDWI (m,n)
            green: reflectance in the green region (m,n)
            NIR: reflectance in the NIR region (m,n)
    '''
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
    '''
    Converts the standard deviation of a gaussian curve to the equivalent full-width at half-maximum
        Parameters:
            sigma: standard deviation
        Returns:
            sigma * np.sqrt(8 * np.log(2)): FWHM
    '''
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    '''
    Converts the full-width at half-maximum of a gaussian curve to the equivalent standard deviation
        Parameters:
            fwhm: full-width at half-maximum
        Returns:
            fwhm / np.sqrt(8 * np.log(2)): standard deviation
    '''
    return fwhm / np.sqrt(8 * np.log(2))

def is_odd(num):
    '''
    Assesses if a number is odd 
        Parameters:
            num: number
        Returns:
            num & 0x1: boolean, True if odd, False if even
    '''
    return num & 0x1

def window_stdev(X, window_size):
    '''
    Computes the standard deviation over a sliding window going through an array, and assigning it to the center pixel
        Parameters:
            X: array 
            window_size: size of the sliding window
        Returns:
            np.sqrt(c2 - c1*c1): array containing the standard deviations
    '''
    r,c=X.shape
    X+=np.random.rand(r,c)*1e-6 #avoids errors when c2-c1*c1 is too small and sqrt returns a nan by eliminating these small numbers
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)
