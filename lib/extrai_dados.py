from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import neighbors
from sklearn import metrics, cross_validation
from sklearn import preprocessing

import ia636 as ia
import iatexture
import numpy as np
import pickle

import os.path

representacoes = []

representacoes_file = '/awmedia/www/media/Attachments/courseIA9012S2015/projeto_jt_dados/representacoes.dat'

def get_dados(use_hist = False, use_co = False, use_cc = False, preprocessamento = 0):

 path = '/awmedia/www/media/ia901_2S2015/retina/'
 features = []
 classes = []
 dados_textura_co = []
 dados_textura_cc = []

 # preprocessing classe 0
 for i in range(300):
    img_filename = path+'CLASSE 0/%s.jpg' % str(i+1).zfill(3)
    if not os.path.isfile(img_filename):
       continue
        img = adreadgray(img_filename)
        #img = iatexture.normhiststret(img_ori)

        features.append([])

        if use_hist:
           h = ia.iahistogram(img)
           stats = ia.iah2stats(h)
           features[-1] = np.hstack((features[-1],stats))

        if use_co:
           glcm, v = iatexture.glcmdesc(img, offset=np.array([3,0],dtype=np.int32))
           features[-1] = np.hstack((features[-1],v))


        if use_cc:
           rl, g = iatexture.rldesc(img, theta= np.pi)
           features[-1] = np.hstack((features[-1],g))

        classes.append(0)

     # preprocessing classe 1
     for i in range(300):
        img_filename = path+'CLASSE 1/%s.jpg' % str(i+1).zfill(3)
        if not os.path.isfile(img_filename):
           continue
        img = adreadgray(img_filename)
        #img = iatexture.normhiststret(img_ori)

        features.append([])

        if use_hist:
           h = ia.iahistogram(img)
           stats = ia.iah2stats(h)
           features[-1] = np.hstack((features[-1],stats))

        if use_co:
           glcm, v = iatexture.glcmdesc(img, offset=np.array([3,0],dtype=np.int32))
           features[-1] = np.hstack((features[-1],v))

        if use_cc:
           rl, g = iatexture.rldesc(img, theta= np.pi)
           features[-1] = np.hstack((features[-1],g))

        classes.append(1)

     return features, classes

 # extração de atributo a partir das imagens originais PARTE 1

 features, classes = get_dados(use_hist = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 features, classes = get_dados(use_cc = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 features, classes = get_dados(use_co = True, use_cc = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 # extração de atributo a partir das imagens originais PARTE 2

 features, classes = get_dados(use_co = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 features, classes = get_dados(use_hist = True, use_cc = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 features, classes = get_dados(use_hist = True, use_co = True, use_cc = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 features, classes = get_dados(use_hist = True, use_co = True, preprocessamento = 0)
 representacoes.append(np.asarray(features))

 # salva representacoes para utilizar nas próximas etapas
 f = open( representacoes_file, "wb" )

 pickle.dump( representacoes, f )
 pickle.dump( classes, f )

 f.close()
