import core
import numpy as np
import cv2 as cv

if __name__=="__main__":
    # im = core.leeImagen("../test/hm.jpg", cv.IMREAD_COLOR)
    # im2 = core.calculateGaussian(im , 7, 7)
    # im3 = core.calculateGaussian(im, 9, 20)
    # vim = np.array([im, im2, im3])
    # core.pintaVarias(vim)

    imL = core.leeImagen("../../dmdlarge.jpg", cv.IMREAD_GRAYSCALE)
    imS = core.leeImagen("../../dmsmall.png", cv.IMREAD_GRAYSCALE)

    functions = [core.calculateConvSeparableMask, core.calculateConvFirstDerivativeIm, core.calculateConvSecondDerivativeIm ]

    imSmod = core.calculateConvolutionLDG(imL, 7, 5)
    core.pintaI(imSmod)
    
    for f in functions:
        imSmod = f(imL,7)
        core.pintaI(imSmod)

    
    
