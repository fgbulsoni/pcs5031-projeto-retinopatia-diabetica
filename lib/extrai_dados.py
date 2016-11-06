from sklearn.feature_selection import SelectPercentile, f_classif
 2 from sklearn import neighbors
 3 from sklearn import metrics, cross_validation
 4 from sklearn import preprocessing
 5 
 6 import ia636 as ia
 7 import iatexture
 8 import numpy as np
 9 import pickle
10 
11 import os.path
12 
13 representacoes = []
14 
15 representacoes_file = '/awmedia/www/media/Attachments/courseIA9012S2015/projeto_jt_dados/representacoes.dat'
16 
17 def get_dados(use_hist = False, use_co = False, use_cc = False, preprocessamento = 0):
18 
19     path = '/awmedia/www/media/ia901_2S2015/retina/'
20     features = []
21     classes = []
22     dados_textura_co = []
23     dados_textura_cc = []
24 
25     # preprocessing classe 0
26     for i in range(300):
27        img_filename = path+'CLASSE 0/%s.jpg' % str(i+1).zfill(3)
28        if not os.path.isfile(img_filename):
29           continue
30        img = adreadgray(img_filename)
31        #img = iatexture.normhiststret(img_ori)
32 
33        features.append([])
34 
35        if use_hist:
36           h = ia.iahistogram(img)
37           stats = ia.iah2stats(h)
38           features[-1] = np.hstack((features[-1],stats))
39 
40        if use_co:
41           glcm, v = iatexture.glcmdesc(img, offset=np.array([3,0],dtype=np.int32))
42           features[-1] = np.hstack((features[-1],v))
43 
44 
45        if use_cc:
46           rl, g = iatexture.rldesc(img, theta= np.pi)
47           features[-1] = np.hstack((features[-1],g))
48 
49        classes.append(0)
50 
51     # preprocessing classe 1
52     for i in range(300):
53        img_filename = path+'CLASSE 1/%s.jpg' % str(i+1).zfill(3)
54        if not os.path.isfile(img_filename):
55           continue
56        img = adreadgray(img_filename)
57        #img = iatexture.normhiststret(img_ori)
58 
59        features.append([])
60 
61        if use_hist:
62           h = ia.iahistogram(img)
63           stats = ia.iah2stats(h)
64           features[-1] = np.hstack((features[-1],stats))
65 
66        if use_co:
67           glcm, v = iatexture.glcmdesc(img, offset=np.array([3,0],dtype=np.int32))
68           features[-1] = np.hstack((features[-1],v))
69 
70        if use_cc:
71           rl, g = iatexture.rldesc(img, theta= np.pi)
72           features[-1] = np.hstack((features[-1],g))
73 
74        classes.append(1)
75 
76     return features, classes
–
 1 # extração de atributo a partir das imagens originais PARTE 1
 2 
 3 features, classes = get_dados(use_hist = True, preprocessamento = 0)
 4 representacoes.append(np.asarray(features))
 5 
 6 features, classes = get_dados(use_cc = True, preprocessamento = 0)
 7 representacoes.append(np.asarray(features))
 8 
 9 features, classes = get_dados(use_co = True, use_cc = True, preprocessamento = 0)
10 representacoes.append(np.asarray(features))
–
 1 # extração de atributo a partir das imagens originais PARTE 2
 2 
 3 features, classes = get_dados(use_co = True, preprocessamento = 0)
 4 representacoes.append(np.asarray(features))
 5 
 6 features, classes = get_dados(use_hist = True, use_cc = True, preprocessamento = 0)
 7 representacoes.append(np.asarray(features))
 8 
 9 features, classes = get_dados(use_hist = True, use_co = True, use_cc = True, preprocessamento = 0)
10 representacoes.append(np.asarray(features))
11 
12 features, classes = get_dados(use_hist = True, use_co = True, preprocessamento = 0)
13 representacoes.append(np.asarray(features))
–
1 # salva representacoes para utilizar nas próximas etapas
2 f = open( representacoes_file, "wb" )
3 
4 pickle.dump( representacoes, f )
5 pickle.dump( classes, f )
6 
7 f.close()