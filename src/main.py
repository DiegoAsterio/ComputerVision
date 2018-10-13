import core                     # <- Functions are implemented here
import numpy as np
import cv2 as cv

if __name__=="__main__":
    # 1st exercise
    im1 = core.leeImagen("../images/im1.jpg",cv.IMREAD_COLOR)

    # a) Showing convolution using a Gaussian 2D mask
    im1Gauss = core.calculateGaussian(im1, 11, 5)

    core.pintaI(im1)            # Shows original image
    core.pintaI(im1Gauss)       # Shows image after Gaussian conv.

    # b) Obtaining masks and representing them as arrays
    mask1 = core.obtainMasks(3) # Mask with length 3
    mask2 = core.obtainMasks(5) # Mask with length 5
    mask3 = core.obtainMasks(7) # Mask with length 7

    masks = [mask1, mask2, mask3]

    for i in np.arange(len(masks)) :
        print("Mascara numero " + str(i+1) + ": ")
        print(masks[i])         # Showing masks in terminal
        
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
    im2 = core.leeImagen("../images/im2.jpg",cv.IMREAD_GRAYSCALE)
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
    core.showLaplacianPyr(im2,cv.BORDER_DEFAULT)

    # 3rd exercice
    im3 = core.leeImagen("../images/im3.jpg",cv.IMREAD_GRAYSCALE)
    im4 = core.leeImagen("../images/im4.jpg",cv.IMREAD_GRAYSCALE)

    # a) Shows three images high, low and hybrid
    core.showHybridIm(im3,im4) 

