"""This is the code developed for computer vision classes at Granada university for 
the year 2018/19.

"""
import numpy as np
import cv2 as cv
# import pdb

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
        alto, ancho, profundo = im.shape
        if profundo != 3:
            mat = cv.cvtColor(im,COLOR_RGB2Luv)
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
    altoMaximo = getAltoMaximo(vim)
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
        im[y,x,0] = 0           

def pintaVentana(vfilename):
    """ Various images are shown in the same window with their titles
    
    Parameters
    ----------
    vfilename : array_like
        An array of names. Each image is represented as a matrix.
    
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

    """
    return cv.GaussianBlur(im, (ksize,ksize), shape)

def obtainMasks(size):
    return cv.getDerivKernels(1,1,size)

def calculateConvolutionLDG(im,size,sigma,border=cv.BORDER_DEFAULT):
    im2 = cv.GaussianBlur(im,(size,size),sigma)
    return cv.Laplacian(im2, -1, borderType=cv.BORDER_DEFAULT)

def calculateConvSeparableMask(im,size):
    ker = cv.getGaussianKernel(size, sigma)
    return cv.sepFilter2D(im,-1,ker,ker)

# def calculateConvFirstDerivativeIm(im,size):
#     kerX, kerY = cv.getDerivKernels(1,0,size)
#     return (im, -1, kerX, kerY, borderType=cv.BORDER_REFLECT)

def calculateConvSecondDerivativeIm(im, size):
    kerX, kerY = cv.getDerivKernels(2,0,size)
    return cv.sepFilter2D(im,-1,kerX,kerY)

def showFourLevelPyr(im,pyrFunct,border):
    vim = [im]
    newim = im
    for i in range(4):
        newim = pyrFunct(newim,borderType=border)
        vim.append(newim)
    pintaVarias(vim)

def showGaussianPyr(im,border=cv.BORDER_DEFAULT):
    showFourLevelPyr(im,cv.pyrDown,border)
    
def showLaplacianPyr(im,border=cv.BORDER_DEFAULT):
    showFourLevelPyr(im,cv.pyrUp,border)
