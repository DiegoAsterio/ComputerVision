import core                     # <- Functions are implemented here
import numpy as np
import cv2 as cv

if __name__=="__main__":
    # 1st exercise
    im1 = core.leeImagen("../images/marilyn.bmp",cv.IMREAD_COLOR)

    # a) Showing convolution using a Gaussian 2D mask
    im1Gauss = core.calculateGaussian(im1, 3, 1)
    im2Gauss = core.calculateGaussian(im1, 7, 5)

    core.pintaI(im1)            # Shows original image
    core.pintaI(im1Gauss)       # Shows image after Gaussian conv. size=3, sigma=1
    core.pintaI(im2Gauss)       # Shows image with size=5, sigma=3

    # b) Obtaining masks and representing them as arrays
    parameters = [(1,0,3), (1,0,5), (2,0,3), (2,0,3), (1,1,3), (1,1,5), (2,1,3), (2,1,5), (2,2,3), (2,2,5) ]

    for dx, dy, size in parameters:
        kx, ky = core.obtainMasks(dx,dy,size)
        kx.resize(size)
        ky.resize(size)
        print("dx={:d}, dy={:d}, kerX=[{:s}], kerY=[{:s}]".format(dx,dy,", ".join(map(str,kx)),", ".join(map(str,ky))))
    
     # c) Using laplacian of a Gaussian

    # L-G with sigma = 1 border aaaaaa|abcdefgh|hhhhhhh
    imLG1B1 = core.calculateConvolutionLDG(im1, 3, 1, cv.BORDER_REPLICATE)
    core.pintaI(imLG1B1)
    
    # L-G with sigma = 1 border fedcba|abcdefgh|hgfedcb
    imLG1B2 = core.calculateConvolutionLDG(im1, 3, 1, cv.BORDER_REFLECT)
    core.pintaI(imLG1B2)

    # L-G with sigma = 3 border aaaaaa|abcdefgh|hhhhhhh
    imLG3B1 = core.calculateConvolutionLDG(im1, 5, 3, cv.BORDER_REPLICATE)
    core.pintaI(imLG3B1)

    # L-G with sigma = 3 border fedcba|abcdefgh|hgfedcb
    imLG3B2 = core.calculateConvolutionLDG(im1, 5, 3, cv.BORDER_REFLECT)
    core.pintaI(imLG3B2)

    # 2nd exercise
    im2 = core.leeImagen("../images/bicycle.bmp",cv.IMREAD_GRAYSCALE)
    core.pintaI(im2)

    # a) Calculate convolution with a separable mask
    # Gaussian with size = 3, sigma = 1, border fedcba|abcdefgh|hgfedcb
    imSep = core.calculateConvSeparableMask(im2, 3, 1, cv.BORDER_REFLECT)
    core.pintaI(imSep)

    # b) Calculate convolution with a 2D derivative mask
    # First der. size = 3, border iiiiiii|abcdefgh|iiiiiii i=0
    im1Der = core.calculateConvFirstDerivative(im2, 3, cv.BORDER_CONSTANT)
    core.pintaI(im1Der)
    
    # c) Calculating 2D convolution with 2D 2nd derivative mask
    # Second deriv size = 3 , border DEFAULT
    im2Der = core.calculateConvSecondDerivative(im2, 3)
    core.pintaI(im2Der)

    # d) Showing a Gaussian pyramid
    core.showGaussianPyr(im2)

    # Using borders
    core.showGaussianPyr(im2,cv.BORDER_DEFAULT)
        
    # e) Showing a Laplacian pyramid
    core.showLaplacianPyr(im2)

    # Using borders
    #core.showLaplacianPyr(im2,cv.BORDER_DEFAULT)

    # 3rd exercice
    im3 = core.leeImagen("../images/plane.bmp",cv.IMREAD_GRAYSCALE)
    im4 = core.leeImagen("../images/bird.bmp",cv.IMREAD_GRAYSCALE)

    # a) Shows three images high, low and hybrid
    core.showHybridIm(11,5,im3,7,5,im4) 

    im5 = core.leeImagen("../images/dog.bmp",cv.IMREAD_GRAYSCALE)
    im6 = core.leeImagen("../images/cat.bmp",cv.IMREAD_GRAYSCALE)

    core.showHybridIm(15,10,im5,35,33,im6)

    im7 = core.leeImagen("../images/submarine.bmp",cv.IMREAD_GRAYSCALE)
    im8 = core.leeImagen("../images/fish.bmp", cv.IMREAD_GRAYSCALE)

    core.showHybridIm(11,9,im7,7,5,im8)
                        
    im9 = core.leeImagen("../images/motorcycle.bmp",cv.IMREAD_GRAYSCALE)
    im10 = core.leeImagen("../images/bicycle.bmp", cv.IMREAD_GRAYSCALE)

    core.showHybridIm(5,3,im9,5,3,im10)

    im11 = core.leeImagen("../images/marilyn.bmp", cv.IMREAD_GRAYSCALE)
    im12 = core.leeImagen("../images/einstein.bmp", cv.IMREAD_GRAYSCALE)

    core.showHybridIm(9,6,im12,13,8,im11)
