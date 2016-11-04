from sklearn import preprocessing
from sklearn import metrics, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import naive_bayes, svm, linear_model, neighbors, tree, ensemble
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
 
import numpy as np
import pickle
import copy
import os.path
 
# inicio constantes

CLF_KNN = "KNN"
CLF_NBMULTINOMIAL = "NB Multinomial"
CLF_SVM_L = "SVM-L"
CLF_SVM_R = "SVM-R"
CLF_DTREE = "Decision Tree"
CLF_B_DTREE = "B. Decision Tree"
PARAM_VERBOSE = 0
PARAM_JOBS = -1
SVM_PARAM_JOBS = 1

BEST_REPRESENTATION = 2

KNN_RANGE = list(range(1,7))
DEGREE_RANGE = list(range(2,5))
BIN_RANGE = list((10.0**i) for i in range(-3,3))
RANGE7 = list((10.0**i) for i in range(-7,7))
RANGE3 = list((10.0**i) for i in range(-3,3))
EST_RANGE = [50,100,150,200,250]
K_FOLD = 10
CV_K_FOLD = 5
SVC_MAX_ITER = 100000
 
 ## fim constantes
 
def run_featureSelection(X,Y, use_norm = False):

   i = 5
   melhorQtde = 0
   maxFmedida = 0.00

   clf = neighbors.KNeighborsClassifier()

   X_norm = copy.copy(X)

   if use_norm:
       # representative normalization
       indices = np.random.random_integers(0,np.shape(X_norm)[0]-1,np.shape(X_norm)[0]/5)
       normalization = preprocessing.StandardScaler().fit(X_norm[indices])
       X_norm = normalization.transform(X_norm)

   while (i <= 100):
      aux_X = SelectPercentile(f_classif, percentile=i).fit_transform(X_norm, Y)
      scores = cross_validation.cross_val_score(clf, aux_X, Y, cv=5, scoring='f1',n_jobs=-1)
      if (maxFmedida < scores.mean()):
         maxFmedida = scores.mean()
         melhorQtde = i
      i = i + 5

   return SelectPercentile(f_classif, percentile=melhorQtde).fit_transform(X, Y)

def runGridSearch(clf_nome, X, Y):

   parametros = {}

   if(clf_nome==CLF_KNN):
      tuned_params = [{'n_neighbors': KNN_RANGE, 'weights': ['uniform', 'distance']}]
      clf = GridSearchCV(neighbors.KNeighborsClassifier(), tuned_params, cv=CV_K_FOLD, scoring="f1", verbose=PARAM_VERBOSE)
      clf.fit(X,Y)
      parametros["k"] = clf.best_estimator_.n_neighbors
      parametros["weights"] = clf.best_estimator_.weights
      parametros["resultado"] = "K: %f - Weight: %s" % (clf.best_estimator_.n_neighbors, clf.best_estimator_.weights)
      clf_configurado = neighbors.KNeighborsClassifier(n_neighbors=parametros["k"], weights=parametros["weights"])

   elif(clf_nome==CLF_NBMULTINOMIAL):
      tuned_params = [{'alpha': RANGE7}]
      clf = GridSearchCV(naive_bayes.MultinomialNB(), tuned_params, cv=CV_K_FOLD, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=PARAM_JOBS)
      clf.fit(X, Y)
      parametros["alpha"] = clf.best_estimator_.alpha
      parametros["resultado"] = "Alpha: %f" % (clf.best_estimator_.alpha)
      clf_configurado = naive_bayes.MultinomialNB(alpha=parametros["alpha"])

   elif(clf_nome==CLF_SVM_L):
      tuned_params = [{'kernel': ['linear'], 'C': RANGE3}]
      clf = GridSearchCV(svm.SVC(max_iter=SVC_MAX_ITER), tuned_params, cv=CV_K_FOLD, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=PARAM_JOBS)
      clf.fit(X, Y)
      parametros["c"] = clf.best_estimator_.C
      parametros["kernel"] = clf.best_estimator_.kernel
      parametros["resultado"] = "Kernel: %s - C: %f" % (clf.best_estimator_.kernel,clf.best_estimator_.C)
      clf_configurado = svm.SVC(max_iter=SVC_MAX_ITER,kernel=parametros["kernel"],C=parametros["c"])

   elif(clf_nome==CLF_SVM_R):
      tuned_params = [{'kernel': ['rbf'], 'C': RANGE3, 'gamma': RANGE3}]
      clf = GridSearchCV(svm.SVC(max_iter=SVC_MAX_ITER), tuned_params, cv=CV_K_FOLD, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=SVM_PARAM_JOBS)
      clf.fit(X, Y)
      parametros["c"] = clf.best_estimator_.C
      parametros["gamma"] = clf.best_estimator_.gamma
      parametros["kernel"] = clf.best_estimator_.kernel
      parametros["resultado"] = "Kernel: %s - C: %f - Gamma: %f " % (clf.best_estimator_.kernel, clf.best_estimator_.C, clf.best_estimator_.gamma)
      clf_configurado = svm.SVC(max_iter=SVC_MAX_ITER,kernel=parametros["kernel"],C=parametros["c"], gamma=parametros["gamma"])

   elif(clf_nome==CLF_DTREE):
      tuned_params = [{'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2']}]
      clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), tuned_params, cv=10, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=PARAM_JOBS)
      clf.fit(X, Y)
      parametros["criterion"] = clf.best_estimator_.criterion
      parametros["max_features"] = clf.best_estimator_.max_features
      parametros["resultado"] = "Criterion: %s - Max_Features: %s " % (parametros["criterion"], parametros["max_features"])
      clf_configurado = tree.DecisionTreeClassifier(random_state=0,criterion=parametros["criterion"],max_features=parametros["max_features"])

   elif(clf_nome==CLF_B_DTREE):

      tuned_params = [{'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2']}]
      clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), tuned_params, cv=10, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=PARAM_JOBS)
      clf.fit(X, Y)
      parametros["criterion"] = clf.best_estimator_.criterion
      parametros["max_features"] = clf.best_estimator_.max_features
      parametros["resultado"] = "Criterion: %s - Max_Features: %s " % (parametros["criterion"], parametros["max_features"])
      clf_configurado = tree.DecisionTreeClassifier(random_state=0,criterion=parametros["criterion"],max_features=parametros["max_features"])

      tuned_params = [{'n_estimators': [50,100,250,500]}]
      clf = GridSearchCV(ensemble.AdaBoostClassifier(clf_configurado), tuned_params, cv=10, scoring="f1", verbose=PARAM_VERBOSE, n_jobs=PARAM_JOBS)
      clf.fit(X, Y)
      parametros["n_estimator"] = clf.best_estimator_.n_estimators
      clf_configurado = ensemble.AdaBoostClassifier(clf_configurado,n_estimators=parametros["n_estimator"])

   return clf_configurado, clf.best_score_

def runExperiment(clf, train_cv, test_cv):

   partitionSamples = []
   trainSamples = []
   testSamples = []
   modelSelectionSamples = []

   # define houldout partition, but it changes for each "fold"
   holdoutTrainIdx = numpy.sort(train_cv)
   holdoutTestIdx = numpy.sort(test_cv)

   for j in range(len(representacoes)):

      normalization = preprocessing.StandardScaler().fit(representacoes[j][holdoutTrainIdx])
      auxPartitionSamples = normalization.transform(representacoes[j][holdoutTrainIdx])
      auxTestSamples = normalization.transform(representacoes[j][holdoutTestIdx])

      partitionSamples.append(auxPartitionSamples)
      testSamples.append(auxTestSamples)

   partitionSamples = copy.copy(partitionSamples)
   testSamples = copy.copy(testSamples)
   partitionClasses = copy.copy(classes[holdoutTrainIdx])
   testClasses = copy.copy(classes[holdoutTestIdx])

   modelSelectionCV = cross_validation.ShuffleSplit(len(partitionClasses), n_iter=1, test_size=0.25, random_state=0)
   auxTrainIndexes = []
   auxModelSelectionIndexes = []
   for auxTrainIndexes, auxModelSelectionIndexes in modelSelectionCV:
      break

   trainClasses = partitionClasses[auxTrainIndexes]
   modelSelectionClasses = partitionClasses[auxModelSelectionIndexes]

   for k in range(len(representacoes)):
      trainSamples.append(partitionSamples[k][auxTrainIndexes])
      modelSelectionSamples.append(partitionSamples[k][auxModelSelectionIndexes])

   classifier, _ = runGridSearch(clf,representacoes[0],classes)

   # fit data
   classifier = classifier.fit(trainSamples[0],trainClasses)

   predicted_labels = []

   # evalute final result
   predicted_labels = classifier.predict(testSamples[0])
   acc = metrics.accuracy_score(testClasses,predicted_labels)
   precision, recall, f1, support = metrics.precision_recall_fscore_support(testClasses,predicted_labels)

   return acc, f1

representacoes = []

representacoes_file = 'representacoes.dat'
f = open(representacoes_file, 'r')

representacoes = pickle.load(f)
classes = pickle.load(f)

classes = np.asarray(classes)

clfs = [ CLF_KNN, CLF_SVM_L, CLF_SVM_R, CLF_DTREE, CLF_B_DTREE ]

# add feature selection without norm
qtde = len(representacoes)
for idx in range(qtde):
   representacoes.append(run_featureSelection(representacoes[idx],classes))

# add feature selection with norm
for idx in range(qtde):
   representacoes.append(run_featureSelection(representacoes[idx],classes,use_norm=True))

representacoes = [ representacoes[BEST_REPRESENTATION] ]

acc = []
f1 = []

cv = cross_validation.StratifiedKFold(classes,n_folds=10,random_state=0)
for clf in clfs:
   for train_index, test_index in cv:
      auxAcc, auxF1 = runExperiment(clf,train_index,test_index)
      acc.append(auxAcc)
      f1.append(auxF1)

   print clf
   print 'ACC:', numpy.mean(acc), '+/-', numpy.std(acc)
   print 'F1:', numpy.mean(f1), '+/-', numpy.std(f1)
   print ''
KNN
ACC: 0.668333333333 +/- 0.116297033496
F1: 0.667208294723 +/- 0.120146780393

SVM-L
ACC: 0.6725 +/- 0.117411314049
F1: 0.668259151147 +/- 0.131407580149

SVM-R
ACC: 0.615 +/- 0.125709099821
F1: 0.556617211876 +/- 0.271066136443

Decision Tree
ACC: 0.623333333333 +/- 0.115217186218
F1: 0.579253471945 +/- 0.240727558105

B. Decision Tree
ACC: 0.63 +/- 0.11328430312
F1: 0.594362231868 +/- 0.222456945174
