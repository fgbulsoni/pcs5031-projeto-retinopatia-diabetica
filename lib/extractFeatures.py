import numpy as np
import pickle
import csv

representacoes = []

representacoes_file = '../data/representacoes.dat'
f = open(representacoes_file, 'r')

representacoes = pickle.load(f)
classes = pickle.load(f)

classes = np.asarray(classes)

names = ('hist', 'cc', 'co')
idx = (0,1,3)

for i in range(3):
	ff = open('../data/retinopatia_' + names[i] + '.csv', 'w');
	writer = csv.writer(ff, delimiter=',')
	aux = np.c_[ representacoes[ idx[i] ], classes ] 
	writer.writerows( aux )
	ff.close()
