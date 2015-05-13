train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%images(:,1:100); % the first 100 images
Xtrain = train_images(:, 1:1000)';
Ytrain = train_labels(1:1000);
adaBoost = fitensemble(Xtrain, Ytrain,'AdaBoostM2',150,'Tree');
%lpBoost = fitensemble(Xtrain, Ytrain,'LPBoost',150,'Tree');
%totalBoost = fitensemble(Xtrain, Ytrain,'TotalBoost',150,'Tree');
[labels,score] = predict(adaBoost,test_images');
plot(loss(adaBoost,test_images',test_labels,'Mode','Cumulative'));
xlabel('Number of trees');
ylabel('Test classification error');
hold on
%plot(loss(lpBoost,test_images',test_labels,'Mode','Cumulative'),'r--');
%plot(loss(totalBoost,test_images',test_labels,'Mode','Cumulative'),'g.');
hold off;
