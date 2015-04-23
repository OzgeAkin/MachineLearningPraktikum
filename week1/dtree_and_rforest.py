from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from datasetpreparation import * 
from sklearn.externals.six import StringIO  
import pydot

#load train set
trainSet = imagefeatures_and_labels('train')
image_features = trainSet[1]
labels = trainSet[2]

# decision tree 
default_tree = tree.DecisionTreeClassifier().fit(image_features, labels)
entropy_tree1 = tree.DecisionTreeClassifier(criterion = 'entropy').fit(image_features, labels)
entropy_tree2 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, max_features = 'log2').fit(image_features, labels)
entropy_tree3 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, max_features = 50).fit(image_features, labels)

#random forest
default_forest = RandomForestClassifier().fit(image_features, labels)
forest1 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(image_features, labels)

classifier_count = 6
#load test set
testSet = imagefeatures_and_labels('test')
image_features_testSet = testSet[1]
labels_testSet = testSet[2]

results = zeros((classifier_count, labels_testSet.shape[0]), dtype = uint8)
scores = zeros((classifier_count, 1), dtype = float)
importance = zeros((classifier_count, 784), dtype = float)
for i, clf in enumerate((default_tree, entropy_tree1, entropy_tree2, entropy_tree3, default_forest, forest1)):
    results[i] = clf.predict(image_features_testSet)
    scores[i] = clf.score(image_features_testSet, labels_testSet)
    #feature importance
    importance[i] = clf.feature_importances_
    if(i < 4):
        #visualize with pydot for trees
        dot_data = StringIO() 
        tree.export_graphviz(clf, out_file=dot_data) #tree 
        graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
        name = 'tree' + str(i+1)
        graph.write_pdf(name+'.pdf') 
    
#todo: show pixel importance on an image


#cross-validation with best hyperparameters
train_image_features = trainSet[1][:45000]
train_labels = trainSet[2][:45000]
cross_validation_image_features = trainSet[1][-15000:]
cross_validation_labels = trainSet[2][-15000:]
entropy_tree1 = entropy_tree1.fit(train_image_features, train_labels)
forest1 = forest1.fit(train_image_features, train_labels)
cross_validation_score = zeros((2, 1), dtype = float)
cross_validation_score[0] = entropy_tree1.score(cross_validation_image_features, cross_validation_labels)
cross_validation_score[1] = forest1.score(cross_validation_image_features, cross_validation_labels)
