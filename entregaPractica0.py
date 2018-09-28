import numpy as np
import cv2 as cv

def leeImagen(filename, flagColor):
    return cv.imread(filename, flagColor) # cv.IMREAD_GRAYSCALE OR cv.IMREAD_COLOR

def pintaI(im):
    cv.namedWindow('imagen', cv.WINDOW_AUTOSIZE)
    cv.imshow('imagen',im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def transformarColor(vim):
    ret =[]
    for im in vim:
        ancho, largo, profundo = im.shape()
        if profundo != 3:
            mat = cv.cvtColor(im,COLOR_RGB2Luv)
            ret.append(mat)
        ret.append(np.copy(im))
    return ret

def getLargoMaximo(vim):
    largoMaximo = 0
    for im in vim:
        an, lar, pro = im.shape()
        if lar > largoMaximo:
            lar = largoMaximo
    return largoMaximo

def rellenaLargo(vimOrig, largoMaximo):
    vim = np.copy(vimOrig)
    for i in range(len(vim)):             
        ancho , largo, profundo = vim[i].shape()
        mat = block([
            [im],
            [np.zeros(ancho*(largoMaximo-largo)*3).reshape(ancho,largoMaximo-largo,3)]
             ])
        vim[i] = mat
    return vim
            
def pintaVarias(vim):
    cv.namedWindow('varias', cv.WINDOW_AUTOSIZE)
    vimColor = transformarColor(vim)
    largoMaximo = getLargoMaximo(vim)
    vimColor = rellenaLargo(vimColor)
    imAImprimir = vimColor[0]                
    for i in range(1,len(vimColor)):
        imAImprimir = cv.hconcat((vimColor,imAImprimir)) # (1) Puede quede un fallo porque opencv tiene largo y ancho permutados VEASE (2)
    cv.imshow('varias', imAImprimir)
    cv.waitKey(0)
    cv.destroyAllWindows()

def modI(im, vpix):             # Esta modificacion es arbitraria ?
    for x, y in vpix:
        im[y,x,0] = 0           # (2) Aqui tuve en cuenta la permutacion sera vd?

def pintaVentana(vfilename):
    imagenes = []
    for name in vfilename:
        imagenSinTitulo = pintaI(name, cv.IMREAD_COLOR)
        ancho, largo, profundo = imagenSinTitulo.shape()
        nuevaImagen = np.stack( (np.zeros(50*largo*3).reshape(50,largo,3),imagenSinTitulo))
        imagenConTitulo = cv.putText(nuevaImagen, name, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, np.array([255, 0, 0])) # inspiration came from https://stackoverflow.com/questions/42420470/opencv-subplots-images-with-titles-and-space-around-borders#42421245
        imagenes.append(imagenConTitulo)
    pintaVarias(imagenes)
        
