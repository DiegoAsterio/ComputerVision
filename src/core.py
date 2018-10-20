"""This is the code developed for computer vision classes at Granada university for 
the year 2018/19.

"""
import numpy as np
import cv2 as cv
import math
#from functools import reduce    
import pdb
import random

def leeImagen(filename, flagColor):
    """ Reads an image from a file and shows it in grey or color.
    
    Parameters
    ----------
    filename : string
        A string that specifies the route to the image.

    flagColor : flag
        Flags used in opencv to read images. e.g. cv.IMREAD_GRAYSCALE OR cv.IMREAD_COLOR

    Returns
    -------
    type
        Opencv matrix comes from numpy
    
    """
    return cv.imread(filename, flagColor) 

def pintaI(im):
    """ Shows an image from a matrix
    
    Parameters
    ----------
    im : matrix
        Matrix containing values for an image

    """
    cv.namedWindow('imagen', cv.WINDOW_AUTOSIZE)
    cv.imshow('imagen',im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def transformarColor(vim):
    """ A vector of images that doesn't have to be in color is transformed into a vector
    of coloured images.
    
    Parameters
    ----------
    vim : array_like
        An array of images. Each image is represented as a matrix.

    Returns
    -------
    type
        Vector of opencv matrixes
    
    """
    ret =[]
    for im in vim:
        if len(im.shape) != 3:
            mat = cv.cvtColor(im,cv.COLOR_GRAY2RGB)
            ret.append(mat)
        else:
            ret.append(np.copy(im))
    return ret

def getAltoMaximo(vim):
    """ A function to determine the maximun height of a vector of matrixes.
    
    Parameters
    ----------
    vim : array_like
        An array of images. Each image is represented as a matrix.

    Returns
    -------
    type
        int
    
    """
    altoMaximo = 0
    for im in vim:
        alto, ancho, pro = im.shape
        if  alto > altoMaximo:
            altoMaximo = alto
    return altoMaximo

def rellenaPorDebajo(vimOrig, altoMaximo):
    """ Every image in an array is replaced by an image which is similar to the 
    previous. It is essentially the same image as it used to but enlarged with
    black pixels so they all have the same height.
    
    Parameters
    ----------
    vimOrig : array_like
        An array of images. Each image is represented as a matrix.

    altoMaximo : int
        Height every image in the vector is going to have

    Returns
    -------
    type
        array of opencv matrixes
    
    """
    vim = np.copy(vimOrig)
    for i in range(len(vim)):             
        alto , ancho, profundo = vim[i].shape
        mat = cv.vconcat([vim[i],np.zeros((altoMaximo-alto,ancho,3), dtype=np.uint8)])
        vim[i] = mat
    return vim
            
def pintaVarias(vim):
    """ Various images are shown in the same window.
    
    Parameters
    ----------
    vim : array_like
        An array of images. Each image is represented as a matrix.
    
    """
    cv.namedWindow('varias', cv.WINDOW_AUTOSIZE) 
    vimColor = vim
    vimColor = transformarColor(vim)
    altoMaximo = getAltoMaximo(vimColor)
    vimColor = rellenaPorDebajo(vimColor,altoMaximo)
    
    imAImprimir = vimColor[0]
    for i in range(1,len(vim)):
        imAImprimir = cv.hconcat([imAImprimir,vimColor[i]]) 
    cv.imshow('varias', imAImprimir)
    cv.waitKey(0)
    cv.destroyAllWindows()

def modI(im, vpix):             
    """ Modifies pixels in an image
    
    Parameters
    ----------
    im : matrix
        Matrix containing an image

    vpix : array_like
        An array of coordinates of each of the pixels that will
        be modified.

    Returns
    -------
    type
        opencv matrixes
    
    """
    for y, x in vpix:
        im[y][x] = 0
def pintaVentana(vfilename):
    """ Various images are shown in the same window with their titles
    
    Parameters
    ----------
    vfilename : array_like
        An array of names. Each image is represented as a matrix.
n    
    """
    imagenes = []
    for name in vfilename:
        imagenSinTitulo = leeImagen(name, cv.IMREAD_COLOR)
        alto, ancho, profundo = imagenSinTitulo.shape
        nuevaImagen = cv.vconcat( (np.zeros((50,ancho,3),dtype=np.uint8),imagenSinTitulo))
        imagenConTitulo = cv.putText(nuevaImagen, name, (int(0.25*ancho), 30), cv.FONT_HERSHEY_COMPLEX, 1, 300) # inspiration came from https://stackoverflow.com/questions/42420470/opencv-subplots-images-with-titles-and-space-around-borders#42421245
        imagenes.append(imagenConTitulo)
    pintaVarias(imagenes)
        
def calculateGaussian(im, sigma, size=(0,0)):
    """ Calculate an image after applying a gaussian mask to it.
    
    Parameters
    ----------
    im : matrix
        Matrix containing an image
    sigma : double
        Standard devation of the gaussian distribution

    Returns
    -------
    type
        Opencv matrix comes from numpy
    

    """
    
    return cv.GaussianBlur(im,ksize=size,sigmaX=sigma)

def obtainMasks(dx,dy,size):
    """ A function that returns 1D masks to calculate derivative masks 2D convolution
    
    Parameters
    ----------
    size : int
        An int representing the size of the mask

    Returns
    -------
    type
        Numpy array
    
    """
    return cv.getDerivKernels(dx,dy,size)

def calculateConvolutionLDG(im,sigma,size,border=cv.BORDER_DEFAULT):
    """ Calculates 2D convolution with a Laplacian-of-Gaussian operator
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    size : int
        Size of the mask that will be applied to the image

    sigma : float
        Variance of the gaussian distribution

    border : flag
        Flag indicating the type of border used when applying the filter by default cv.BORDER_DEFAULT is used

    Returns
    -------
    type
        Opencv matrix comes from numpy
    
    """

    imagen = cv.GaussianBlur(im,ksize=(size,size),sigmaX=sigma)
    return cv.Laplacian(imagen, -1, ksize=size, scale=sigma, borderType=border)

def calculateConvSeparableMask(im,kerX,kerY,border=cv.BORDER_DEFAULT):
    """ Calculates 2D convolution with a separable mask
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    size : int
        Size of the mask that will be applied to the image

    sigma : float
        Variance of the gaussian distribution

    border : flag
        OpenCV flag for setting the border

    Returns
    -------
    type
        Opencv matrix comes from numpy
    
    """
    return cv.sepFilter2D(im,-1,kerX,kerY,borderType=border)

def calculateConvFirstDerivative(im, x, y, size, border=cv.BORDER_DEFAULT):
    """ Calculates 2D convolution with a mask of first derivatives
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    size : int
        Size of the mask that will be applied to the image

    Returns
    -------
    type
        Opencv matrix comes from numpy
    
    """
    dx = None
    dy = None

    if x and y:
        dx = 1
        dy = 1
    elif x:
        dx = 1
        dy = 0
    else:
        dx = 0
        dy = 1

    kerX, kerY = cv.getDerivKernels(dx, dy, size)
    matrix = kerY*np.transpose(kerX)
    return cv.filter2D(im, -1, matrix, borderType=border) # Multiplicar dos vectores y usar filter2D

def calculateConvSecondDerivative(im, x, y, size,border=cv.BORDER_DEFAULT):
    """ Calculates 2D convolution with a mask of second derivatives
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    size : int
        Size of the mask that will be applied to the image

    Returns
    -------
    type
        Opencv matrix comes from numpy
    
    """
    dx = None
    dy = None
    if x and y:
        dx = 2
        dy = 2
    elif x:
        dx = 2
        dy = 0
    else:
        dx = 0
        dy = 2
    kerX, kerY = cv.getDerivKernels(dx, dy, size)
    matrix = kerY*np.transpose(kerX)
    return cv.filter2D(im,-1,matrix, borderType=border)

def nLevelPyr(im,n,pyrFunct,border):
    """ Shows a pyramid of the same levels as one wants
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    n : int
        Number of levels for the pyramid

    pyrFunct : function_like
        A function that given an image returns the next image in
    the piramid.

    border : flag
        OpenCV flag for setting the border

    """
    vim = [im]
    newim = im
    for i in range(n):
        newim = pyrFunct(newim,borderType=border)
        vim.append(newim)
    return vim

def correctSignal(sig):
    imM = np.copy(sig)
    itsShape = imM.shape
    imM = imM.reshape(-1)
    m = imM.min()
    M = imM.max()
    f = lambda x : (x-m)/(M-m)
    imM = np.fromiter((f(i) for i in imM), imM.dtype)
    imM.reshape(itsShape)
    return imM

def showGaussianPyr(im,border=cv.BORDER_DEFAULT):
    """ Shows a Gaussian pyramid of four levels
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    border : flag
        OpenCV flag for setting the border

    """
    vim = nLevelPyr(im,4,cv.pyrDown,border)
    pintaVarias(vim[1:])
    
def showLaplacianPyr(im,border=cv.BORDER_DEFAULT):
    """ Shows a Laplacian pyramid of four levels
    
    Parameters
    ----------
    im : matrix_like
        An image in OpenCV format

    border : flag
        OpenCV flag for setting the border

    """
    vim = nLevelPyr(im,5,cv.pyrDown,border)
    for i in range(len(vim))[1:-1]:
        nextLevel = cv.pyrUp(vim[i+1],dstsize=vim[i].shape[::-1])
        vim[i] = cv.subtract(vim[i],nextLevel)

    pintaVarias(vim[1:-1])

def getHybridIm(size1,sigma1,im1,size2,sigma2,im2):
    im1blurr = calculateGaussian(im1, sigma1, (size1,size1))
    im2blurr = calculateGaussian(im2, sigma2, (size2,size2))
    im2detail = cv.subtract(im2, im2blurr)

    hybridIm = cv.add(im1blurr,im2detail)

    return im1blurr, im2detail, hybridIm

    
def showHybridIm(size1,sigma1,im1,size2,sigma2,im2):
    """ Shows a hybrid image using two images
    
    Parameters
    ----------
    im1 : matrix_like
        An image in OpenCV format

    im2 : matrix_like
        An image in OpenCV format

    """

    im1blurr, im2detail, hybridIm = getHybridIm(size1,sigma1,im1,size2,sigma2,im2)
    
    vim = [im1blurr, im2detail, hybridIm]

    pintaVarias(vim)


def calculate1DGaussian(sigma):
    """ Calculates a Gaussian mask vector 

    Parameters
    ----------

    sigma : int
        Number of pixels for the standard deviation

    """
    f = lambda x : math.exp(-0.5*x*x/(sigma*sigma))
    mask = []
    for i in np.arange(2*math.floor(3*sigma)+1) - math.floor(3*sigma): # 99.7% de la masa de la funcion probabilidad
        mask.append(f(i))

    #pdb.set_trace()
    suma = 0
    for elem in mask:
        suma = elem + suma

    mask = [x/suma for x in mask]

    return np.array(mask)

def convoluteOverSignal(mask,signal):

    K = len(mask)

    N = 1
    for elem in signal.shape:
        N = N*elem
    
    signal=signal.reshape(N)

    bordeIzq = [signal[i] for i in range(math.floor(K/2))]
    bordeIzq.reverse()
    bordeIzq = np.array(bordeIzq)
    
    bordeDch = [signal[-i] for i in range(math.floor(K/2))]
    bordeDch = np.array(bordeDch)
    
    band = np.hstack([bordeIzq,signal,bordeDch])

    maskP = np.copy(mask[::-1])
    
    resultado = []

    for i in range(N):
        aux = 0
        for j in range(K):
            aux = aux + maskP[j]*band[j+i]
        resultado.append(aux)

    return np.array(resultado)

def calculateConvolution1D(mask, signal):
    """ Calculates convolution for a vector-mask (1D) over a signal
    we use reflected borders.

    Parameters
    ----------

    mask : array_like
        Mask in an array.

    signal : array_like
        Array that represents a signal

    """
    vectorDeColor = False    
    if len(signal.shape) == 3:
        vectorDeColor = True

    if vectorDeColor :

        layers = cv.split(signal)

        layersChanged = []

        for color in layers:
            layersChanged.append(convoluteOverSignal(mask,color))
#        pdb.set_trace()

        ret = cv.merge(layersChanged)
    else:
        v=convoluteOverSignal(mask,signal)
        ret = v.reshape((len(v),1))

    return ret
        
def convoluteWithSeparableMask(kerX,kerY,im):
    """ Calculates convolution for a 2D separable mask
    we use reflected borders.

    Parameters
    ----------

    kerX : array_like
        Mask for x in an array.

    kerY : array_like
        Mask for x in an array.

    im : matrix_like
        OpenCV matrix

    """
    alto = None
    ancho = None
    profundo = None
    if len(im.shape) == 3:
        alto, ancho, profundo = im.shape
    else:
        alto, ancho = im.shape

    imFilas = np.vsplit(im,alto)
    FilasConv = np.array([ calculateConvolution1D(kerX,x) for x in imFilas])
    
    imConvFilas = np.hstack(FilasConv)

    imColumnas = np.vsplit(imConvFilas,ancho)
    ColumnasConv = np.array([ calculateConvolution1D(kerY,x) for x in imColumnas])
        
    imFinal = np.hstack(ColumnasConv)

    ret = imFinal.astype('uint8')

    return ret

def subSample(im):
    """ Subsamples an image in half the previous size

    Parameters
    ----------

    im : matrix_like
        OpenCV matrix

    """
    mask = calculate1DGaussian(1)
    imC = convoluteWithSeparableMask(mask,mask,im)

    alto = None
    ancho = None
    b = None
    ret = None
    
    if len(imC.shape)==3:
        alto, ancho, profundo = imC.shape
        booleanArray = [(j%2==0)and(k%2==0) for k in range(alto) for j in range(ancho) for i in range(profundo)] #Cojo los pts.(de profundidad 3) pares en cada linea y las lineas pares
        b = np.array(booleanArray).reshape((alto,ancho,profundo)) # Hago una matriz con la forma adeacuada para el bradcasting
        ret = imC[b].reshape((int(alto/2)+alto%2,int(ancho/2)+ancho%2,profundo)) # El casting devuelve un vector y se transforma en una matriz
    else:
        alto, ancho = imC.shape
        booleanArray = [(j%2==0)and(k%2==0) for k in range(alto) for j in range(ancho)]#Cojo los pts.(de profundidad 1) pares en cada linea y las lineas pares
        b = np.array(booleanArray).reshape((alto,ancho)) # Hago una matriz con la forma adeacuada para el bradcasting
        ret = imC[b].reshape((int(alto/2)+alto%2,int(ancho/2)+ancho%2)) # El casting devuelve un vector y se transforma en una matriz
        
    return np.copy(ret)

def subSampleForFunction(im,borderType=None):
    """ Two parameter function to reuse code

    Parameters
    ----------

    im : matrix_like
        OpenCV matrix

    """
    return subSample(im)

def showMyOwnGPyr(im):
    """ Shows a gaussian pyramid

    Parameters
    ----------

    im : matrix_like
        OpenCV matrix

    """
    fiveLevels = nLevelPyr(im,5,subSampleForFunction,None)
    pintaVarias(fiveLevels)

def correctOverFlows(im):
    """ Corrects overflow by truncating

    Parameters
    ----------

    im : matrix_like
        OpenCV matrix

    """
    aShape = im.shape
    ret = im.reshape(-1)
    f = lambda x : 0 if x < 0 else 255 if x>255 else x
    ret = np.fromiter((f(i) for i in ret), ret.dtype)
    ret = ret.reshape(aShape)
    return ret
    
def myOwnHybridIm(sigma1,im1,sigma2,im2):
    """ Generates an hybrid image

    Parameters
    ----------

    sigma1 : float
        Sigma for blurring

    im1 : matrix_like
        OpenCV matrix

    sigma2 : float
        Sigma for sharpening

    im2 : matrix_like
        OpenCV matrix

    """
    im1I = im1.astype('float')
    im2I = im2.astype('float')
    ker1 = calculate1DGaussian(sigma1)
    blurred1 = convoluteWithSeparableMask(ker1,ker1,im1I)
    ker2 = calculate1DGaussian(sigma2)
    blurred2 = convoluteWithSeparableMask(ker2,ker2,im2I)
    hifreq = im2I - blurred2
    ret = hifreq + blurred1
    blurred1 = correctOverFlows(blurred1)
    hifreq = correctOverFlows(hifreq)
    ret = correctOverFlows(ret)
    return blurred1.astype('uint8'), hifreq.astype('uint8'), ret.astype('uint8')

def showMyOwnHybridIm(sigma1,im1,sigma2,im2):
    """ Shows an hybrid image

    Parameters
    ----------

    sigma1 : float
        Sigma for blurring

    im1 : matrix_like
        OpenCV matrix

    sigma2 : float
        Sigma for sharpening

    im2 : matrix_like
        OpenCV matrix

    """
    imgs = myOwnHybridIm(sigma1,im1,sigma2,im2)
    pintaVarias(imgs)
