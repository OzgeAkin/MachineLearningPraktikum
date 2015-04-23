from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from datasetpreparation import * 
from sklearn.externals.six import StringIO  
import pydot
from pylab import imshow, show, cm, figure
from sklearn import cross_validation
        
#load train set
trainSet = imagefeatures_and_labels('train')
image_features = trainSet[1]
labels = trainSet[2]

# decision tree 
default_tree = tree.DecisionTreeClassifier()
tree1 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 15, max_features = None)
tree2 = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 15, max_features = None)
tree3 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 15, max_features = (28*28/2))
tree4 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 25, max_features = (28*28/2))


#random forest
default_forest = RandomForestClassifier().fit(image_features, labels)
forest1 = RandomForestClassifier(criterion = 'entropy', max_depth=15, n_estimators=10, max_features='auto')
forest2 = RandomForestClassifier(criterion = 'entropy', max_depth=25, n_estimators=10, max_features='auto')
forest3 = RandomForestClassifier(criterion = 'entropy', max_depth=15, n_estimators=20, max_features='auto')
forest4 = RandomForestClassifier(criterion = 'gini', max_depth=15, n_estimators=20, max_features=None)
forest5 = RandomForestClassifier(criterion = 'entropy', max_depth=15, n_estimators=20, max_features=None)


classifier_count = 11
#load test set
testSet = imagefeatures_and_labels('test')
image_features_testSet = testSet[1]
labels_testSet = testSet[2]

results = zeros((classifier_count, labels_testSet.shape[0]), dtype = uint8)
scores = zeros((classifier_count, 1), dtype = float)
importance = zeros((classifier_count, 784), dtype = float)
for i, clf in enumerate((default_tree, tree1, tree2, tree3, tree4, default_forest, forest1, forest2, forest3, forest4, forest5)):
    clf = clf.fit(image_features, labels)
    results[i] = clf.predict(image_features_testSet)
    scores[i] = clf.score(image_features_testSet, labels_testSet)
    #feature importance
    importance[i] = clf.feature_importances_
    #show pixel importance on an image
    importance_shaped = importance[i].reshape((28,28))
    figure()
    imshow(importance_shaped, cmap = cm.hot)
    show()
    if(i < 5):
        #visualize with pydot for trees
        dot_data = StringIO() 
        tree.export_graphviz(clf, out_file=dot_data) #tree 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        name = 'tree' + str(i+1)
        graph.write_png(name+'.png') 

cross_validation_mean = zeros((2, 1), dtype = float)
cross_validation_mean[0] = cross_validation.cross_val_score(tree1, image_features, labels).mean()
cross_validation_mean[1] = cross_validation.cross_val_score(forest3, image_features, labels).mean()
