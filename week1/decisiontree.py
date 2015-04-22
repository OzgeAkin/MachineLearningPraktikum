from sklearn import tree
from datasetpreparation import * 
from sklearn.externals.six import StringIO  
import pydot

#load train set
trainSet = imagefeatures_and_labels('train')

#take first 1000 element of image_features and labels
subset_image_features = trainSet[0]
subset_labels = trainSet[1]

# decision tree with default parameters
default_tree = tree.DecisionTreeClassifier().fit(subset_image_features, subset_labels)
entropy_tree1 = tree.DecisionTreeClassifier(criterion = 'entropy').fit(subset_image_features, subset_labels)
entropy_tree2 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 2, max_features = 'log2').fit(subset_image_features, subset_labels)
entropy_tree3 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 2, max_features = 50).fit(subset_image_features, subset_labels)



#load test set
testSet = imagefeatures_and_labels('test')
image_features_testSet = testSet[0]
labels_testSet = testSet[1]

results = zeros((4, labels_testSet.shape[0]), dtype = uint8)
scores = zeros((4, 1), dtype = float)
importance = zeros((4, 784), dtype = uint8)
for i, clf in enumerate((default_tree, entropy_tree1, entropy_tree2, entropy_tree3)):
    results[i] = clf.predict(image_features_testSet)
    scores[i] = clf.score(image_features_testSet, labels_testSet)
    #feature importance
    importance[i] = clf.feature_importances_
    #visualize with pydot
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    name = 'C:\Users\Samsung\Desktop\TUM\SS15\MachineLearning\praktikum\week1\tree' + str(i)
    graph.write_pdf("tree.pdf") 
    
    
