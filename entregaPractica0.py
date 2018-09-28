import numpy as np
import cv2 as cv
# import pdb

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
        alto, ancho, profundo = im.shape
        if profundo != 3:
            mat = cv.cvtColor(im,COLOR_RGB2Luv)
            ret.append(mat)
        else:
            ret.append(np.copy(im))
    return ret

def getAltoMaximo(vim):
    altoMaximo = 0
    for im in vim:
        alto, ancho, pro = im.shape
        if  alto > altoMaximo:
            altoMaximo = alto
    return altoMaximo

def rellenaPorDebajo(vimOrig, altoMaximo):
    vim = np.copy(vimOrig)
    for i in range(len(vim)):             
        alto , ancho, profundo = vim[i].shape
        mat = cv.vconcat([vim[i],np.zeros((altoMaximo-alto,ancho,3), dtype=np.uint8)])
        vim[i] = mat
    return vim
            
def pintaVarias(vim):
    cv.namedWindow('varias', cv.WINDOW_AUTOSIZE)
    vimColor = vim
    vimColor = transformarColor(vim)
    altoMaximo = getAltoMaximo(vim)
#    pdb.set_trace()
    vimColor = rellenaPorDebajo(vimColor,altoMaximo)
    
    imAImprimir = vimColor[0]
    for i in range(1,len(vim)):
        imAImprimir = cv.hconcat([imAImprimir,vimColor[i]]) 
        pdb.set_trace()
    cv.imshow('varias', imAImprimir)
    cv.waitKey(0)
    cv.destroyAllWindows()

def modI(im, vpix):             
    for y, x in vpix:
        im[y,x,0] = 0           

def pintaVentana(vfilename):
    imagenes = []
    for name in vfilename:
        imagenSinTitulo = leeImagen(name, cv.IMREAD_COLOR)
        alto, ancho, profundo = imagenSinTitulo.shape
        nuevaImagen = cv.vconcat( (np.zeros((50,ancho,3),dtype=np.uint8),imagenSinTitulo))
        imagenConTitulo = cv.putText(nuevaImagen, name, (int(0.25*ancho), 30), cv.FONT_HERSHEY_COMPLEX, 1, 300) # inspiration came from https://stackoverflow.com/questions/42420470/opencv-subplots-images-with-titles-and-space-around-borders#42421245
        imagenes.append(imagenConTitulo)
    pintaVarias(imagenes)
        
