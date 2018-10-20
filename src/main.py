import core                     # <- Functions are implemented here
import numpy as np
import cv2 as cv

if __name__=="__main__":
    print("1st exercise")
    im1 = core.leeImagen("./images/einstein.bmp",cv.IMREAD_COLOR)

    print("This is the original image\n")
    core.pintaI(im1)
    
    print("a) Examples of convolution using a Gaussian 2D mask\n")
    im1Gauss = core.calculateGaussian(im1, 1) #sigma=1
    im2Gauss = core.calculateGaussian(im1, 3) #sigma=3

    print("Gaussian convolution sigma=1")
    core.pintaI(im1Gauss)       

    print("Gaussian convolution sigma=3")
    core.pintaI(im2Gauss)       

    print("b) Obtaining 1D masks and representing them as arrays")
    parameters = [(1,0,3),  (0,1,3), (1,1,3), (2,0,3), (0,2,3), (2,2,3)]

    for dx, dy, size in parameters:
        kx, ky = core.obtainMasks(dx,dy,size)
        kx.resize(size)
        ky.resize(size)
        print("dx={:d}, dy={:d}, kerX=[{:s}], kerY=[{:s}]".format(dx,dy,", ".join(map(str,kx)),", ".join(map(str,ky))))
    
    print(" c) Using laplacian of a Gaussian")

    print("L-G with sigma = 1 border aaaaaa|abcdefgh|hhhhhhh")
    imLG1B1 = core.calculateConvolutionLDG(im1, 1, 3, cv.BORDER_REPLICATE)
    core.pintaI(imLG1B1)
    
    print("L-G with sigma = 1 border fedcba|abcdefgh|hgfedcb")
    imLG1B2 = core.calculateConvolutionLDG(im1, 1, 3, cv.BORDER_REFLECT)
    core.pintaI(imLG1B2)

    print("L-G with sigma = 3 border aaaaaa|abcdefgh|hhhhhhh")
    imLG3B1 = core.calculateConvolutionLDG(im1, 3, 3, cv.BORDER_REPLICATE)
    core.pintaI(imLG3B1)

    print("L-G with sigma = 3 border fedcba|abcdefgh|hgfedcb")
    imLG3B2 = core.calculateConvolutionLDG(im1, 3, 3, cv.BORDER_REFLECT)
    core.pintaI(imLG3B2)

    print("2nd exercise")
    im2 = core.leeImagen("./images/bicycle.bmp",cv.IMREAD_GRAYSCALE)
    core.pintaI(im2)

    print("a) Calculate convolution with a separable Gaussian mask with size = 3, sigma = 1, border fedcba|abcdefgh|hgfedcb")
    ker = cv.getGaussianKernel(3,1)
    imSep = core.calculateConvSeparableMask(im2, ker, ker, cv.BORDER_REFLECT)
    core.pintaI(imSep)

    print(" b) Calculate convolution with a 2D derivative mask first der. size = 3, border iiiiiii|abcdefgh|iiiiiii i=0")
    im1dx = core.calculateConvFirstDerivative(im2, True, False, 3, cv.BORDER_CONSTANT) #dx
    core.pintaI(im1dx)

    im1dy = core.calculateConvFirstDerivative(im2, False, True, 3, cv.BORDER_CONSTANT) #dy
    core.pintaI(im1dy)

    im1dxdy = core.calculateConvFirstDerivative(im2, True, True, 3, cv.BORDER_CONSTANT) #dxdy

    core.pintaI(im1dxdy)
    
    print("c) Calculating 2D convolution with 2D 2nd derivative mask")
    print("Second deriv size = 3 , border DEFAULT")
    im1dxdx = core.calculateConvSecondDerivative(im2,True, False, 3) #dxdx
    core.pintaI(im1dxdx)

    im1dydy = core.calculateConvSecondDerivative(im2, False, True, 3) #dydy
    core.pintaI(im1dydy)

    im1dxdxdydy = core.calculateConvSecondDerivative(im2,True, True, 3) #dxdxdydy
    core.pintaI(im1dxdxdydy)
    
    print("d) Showing a Gaussian pyramid")
    core.showGaussianPyr(im2)

    print("dd) Shows image with sigma=3 using borders")
    core.showGaussianPyr(im2,cv.BORDER_REFLECT)
        
    print("e) Showing a Laplacian pyramid")
    core.showLaplacianPyr(im2)

    print("ee) Showing a Laplacian pyramid using borderse")
    core.showLaplacianPyr(im2,cv.BORDER_REPLICATE)

    print("3rd exercice")
    
    print("a) Shows three images low, high and hybrid")

    im3 = core.leeImagen("./images/plane.bmp",cv.IMREAD_GRAYSCALE)
    im4 = core.leeImagen("./images/bird.bmp",cv.IMREAD_GRAYSCALE)

    core.showHybridIm(11,5,im3,7,5,im4) 

    im5 = core.leeImagen("./images/dog.bmp",cv.IMREAD_GRAYSCALE)
    im6 = core.leeImagen("./images/cat.bmp",cv.IMREAD_GRAYSCALE)

    core.showHybridIm(15,10,im5,35,33,im6)

    im7 = core.leeImagen("./images/submarine.bmp",cv.IMREAD_GRAYSCALE)
    im8 = core.leeImagen("./images/fish.bmp", cv.IMREAD_GRAYSCALE)

    core.showHybridIm(11,9,im7,7,5,im8)
                        
    im9 = core.leeImagen("./images/motorcycle.bmp",cv.IMREAD_GRAYSCALE)
    im10 = core.leeImagen("./images/bicycle.bmp", cv.IMREAD_GRAYSCALE)

    core.showHybridIm(9,7,im9,15,13,im10)

    im11 = core.leeImagen("./images/marilyn.bmp", cv.IMREAD_GRAYSCALE)
    im12 = core.leeImagen("./images/einstein.bmp", cv.IMREAD_GRAYSCALE)
    
    core.showHybridIm(9,6,im12,13,8,im11)

    print("BONUS")

    print("Exercice 1")
    vectoresMascaraGaussianos = [ (i,core.calculate1DGaussian(i)) for i in 2*np.arange(4) + 1 ]
    
    for sigma,vector in vectoresMascaraGaussianos:
        print("sigma = {:d}, tam = {:d}, mascara1D = [{:s}, ...]\n".format(sigma,len(vector),", ".join(map(str,vector[0:2]))))

    print("Exercice 2")
        
    mask =vectoresMascaraGaussianos[0][1]

    im13 = core.leeImagen("./images/plane.bmp",cv.IMREAD_COLOR)

    alto, ancho, profundo = im13.shape
    
    colorRow = np.vsplit(im13,alto)[0]

    convColorRow = core.calculateConvolution1D(mask, colorRow)

    print("Fila de una imagen a color convolucionada [{:s}...,{:s}]".format(", ".join(map(str,convColorRow[0:3])),", ".join(map(str,convColorRow[-3:]))))

    alto,ancho = im2.shape
          
    grayscaleRow = np.vsplit(im2,alto)[0]
          
    convGSRow = core.calculateConvolution1D(mask, grayscaleRow)

    print("Fila de una imagen en escala de grises convolucionada [{:s}...,{:s}]".format(", ".join(map(str,convGSRow[0:3])),", ".join(map(str,convGSRow[-3:]))))

    print("Exercice 3")
          
    imC = core.convoluteWithSeparableMask(mask,mask,im13)

    core.pintaI(imC)
    
    imGS = core.convoluteWithSeparableMask(mask,mask,im2)

    core.pintaI(imGS)

    print("Exercice 4")

    a,b,hybridIm1 = core.myOwnHybridIm(4,im3,2,im4) 
    
    core.showMyOwnGPyr(hybridIm1)

    a,b,hybridIm2 = core.myOwnHybridIm(3,im5,5,im6)

    core.showMyOwnGPyr(hybridIm2)

    a,b,hybridIm3 = core.myOwnHybridIm(2,im7,2,im8)

    core.showMyOwnGPyr(hybridIm3)

    a,b,hybridIm4 = core.myOwnHybridIm(3,im9,2,im10)

    core.showMyOwnGPyr(hybridIm4)

    a,b,hybridIm5 = core.myOwnHybridIm(2,im12,4,im11)

    core.showMyOwnGPyr(hybridIm5)
    print("Exercice 5")

    print("a) Shows three images low, high and hybrid")
    im3C = core.leeImagen("./images/plane.bmp",cv.IMREAD_COLOR)
    im4C = core.leeImagen("./images/bird.bmp",cv.IMREAD_COLOR)

    core.showMyOwnHybridIm(5,im3C,2,im4C) 

    im5C = core.leeImagen("./images/dog.bmp",cv.IMREAD_COLOR)
    im6C = core.leeImagen("./images/cat.bmp",cv.IMREAD_COLOR)

    core.showMyOwnHybridIm(3,im5C,6,im6C)

    im7C = core.leeImagen("./images/submarine.bmp",cv.IMREAD_COLOR)
    im8C = core.leeImagen("./images/fish.bmp", cv.IMREAD_COLOR)

    core.showMyOwnHybridIm(3,im7C,2,im8C)

    im9C = core.leeImagen("./images/motorcycle.bmp",cv.IMREAD_COLOR)
    im10C = core.leeImagen("./images/bicycle.bmp", cv.IMREAD_COLOR)

    core.showMyOwnHybridIm(3,im9C,3,im10C)

    im11C = core.leeImagen("./images/einstein.bmp", cv.IMREAD_COLOR)
    im12C = core.leeImagen("./images/marilyn.bmp", cv.IMREAD_COLOR)
    core.showMyOwnHybridIm(4,im11C,3,im12C)
