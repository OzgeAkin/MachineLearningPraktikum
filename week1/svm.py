from sklearn import svm, metrics
from datasetpreparation import * 
        
trainSet = imagefeatures_and_labels('train')
#take first 1000 element of image_features and labels
subset_image_features = trainSet[1][:10000]
subset_labels = trainSet[2][:10000]
#svc training
C = 1.0  # SVM regularization parameter
default_svc=svm.SVC()
svc = svm.SVC(kernel='linear', C=C)
rbf_svc1 = svm.SVC(kernel='rbf', gamma=0.05, C=C)
rbf_svc2 = svm.SVC(kernel='rbf', gamma=0.7, C=C)
poly_svc1 = svm.SVC(kernel='poly', degree=3, C=C)
poly_svc2 = svm.SVC(kernel='poly', degree=5, C=C)
lin_svc = svm.LinearSVC(C=C)


testSet = imagefeatures_and_labels('test')
image_features_testSet = testSet[1]
labels_testSet = testSet[2]

results = zeros((7, labels_testSet.shape[0]), dtype = uint8)
for i, clf in enumerate((default_svc, svc, rbf_svc1, rbf_svc2, poly_svc1, poly_svc2, lin_svc)):
    clf = clf.fit(subset_image_features, subset_labels)
    results[i] = clf.predict(image_features_testSet)
    print '----'
    print (i+1)
    #scores
    print metrics.classification_report(labels_testSet, results[i])
    

