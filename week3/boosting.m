train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
%images(:,1:100); % the first 100 images
train_images(:, 1:1);
Xtrain = train_images(:, 1:1)';
Ytrain = train_labels(1:1);
ada = fitensenmble(Xtrain, Ytrain,'AdaBoostM1',50,'Tree');