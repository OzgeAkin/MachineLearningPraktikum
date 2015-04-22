from sklearn import svm
from datasetpreparation import * 
  
trainSet = imagefeatures_and_labels('train')
#take first 1000 element of image_features and labels
subset_image_features = trainSet[0][:5]
subset_labels = trainSet[1][:5]
#svc training
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(subset_image_features, subset_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(subset_image_features, subset_labels)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(subset_image_features, subset_labels)
lin_svc = svm.LinearSVC(C=C).fit(subset_image_features, subset_labels)

testSet = imagefeatures_and_labels('test')
image_features_testSet = testSet[0][:5]
labels_testSet = testSet[1][:5]

results = zeros((4, labels_testSet.shape[0]), dtype = uint8)
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    results[i] = clf.predict(image_features_testSet)
    
    #todo: scores

