"""This is the code developed for computer vision classes at Granada university for 
the year 2018/19.

"""
import numpy as np
import cv2 as cv
#import pdb

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
        # pdb.set_trace()
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
        
def calculateGaussian(im, ksize, shape):
    """ Calculate an image after applying a gaussian mask to it.
    
    Parameters
    ----------
    im : matrix
        Matrix containing an image
    ksize : int
        Size of the mask
    shape : double
        Variance of the gaussian distribution

    Returns
    -------
    type
        Opencv matrix comes from numpy
    

    """
    return cv.GaussianBlur(im, (ksize,ksize), shape)

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

def calculateConvolutionLDG(im,size,sigma,border=cv.BORDER_DEFAULT):
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
    return cv.Laplacian(im, -1, ksize=size, scale=sigma, borderType=border)

def calculateConvSeparableMask(im,size,sigma,border=cv.BORDER_DEFAULT):
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
    ker = cv.getGaussianKernel(size, sigma)
    return cv.sepFilter2D(im,-1,ker,ker,borderType=border)

def calculateConvFirstDerivative(im,size, border=cv.BORDER_DEFAULT):
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
    kerX, kerY = cv.getDerivKernels(1,0,size)
    matrix = kerY*np.transpose(kerX)
    return cv.filter2D(im, -1, matrix, borderType=border) # Multiplicar dos vectores y usar filter2D

def calculateConvSecondDerivative(im, size,border=cv.BORDER_DEFAULT):
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
    kerX, kerY = cv.getDerivKernels(2,0,size)
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
    vim = nLevelPyr(im,4,cv.pyrDown,border)
    for i in range(len(vim))[1:-1]:
        nextLevel = cv.pyrUp(vim[i+1],dstsize=vim[i].shape[::-1])
        vim[i] = cv.subtract(vim[i],nextLevel)

    pintaVarias(vim[1:])
    #showestamalNLevelPyr(im,4,cv.pyrUp,border) #ESTA MAL anadir a vim en cada imagen + redimensionar(orig - blurred)

def showHybridIm(size1,sigma1,im1,size2,sigma2,im2):
    """ Shows a hybrid image using two images
    
    Parameters
    ----------
    im1 : matrix_like
        An image in OpenCV format

    im2 : matrix_like
        An image in OpenCV format

    """
    im1blurr = calculateGaussian(im1, size1, sigma1)
    im2blurr = calculateGaussian(im2, size2, sigma2)
    im2detail = cv.subtract(im2, im2blurr)

    hybridIm = cv.add(im1blurr,im2detail)

    vim = [im1blurr, im2detail, hybridIm]

    pintaVarias(vim)
    
