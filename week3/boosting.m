clear all, close all
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%images(:,1:100); % the first 100 images
Xtrain = train_images(:, 1:10000)';
Ytrain = train_labels(1:10000);
%1000 learning cycles
adaBoost = fitensemble(Xtrain, Ytrain,'AdaBoostM2',1000,'Tree');
lpBoost = fitensemble(Xtrain, Ytrain,'LPBoost',1000,'Tree');
totalBoost = fitensemble(Xtrain, Ytrain,'TotalBoost',1000,'Tree');
[labels,score] = predict(adaBoost,test_images'); %calculate entropy with score, for loop ?
%accuracy
correct_labels = labels == test_labels;
accuracy = sum(correct_labels)/length(test_labels);
%entropy
score_ = score(:,1) .* log2(score(:,1));
entropy = sum(score_);
figure(1)
plot(loss(adaBoost,test_images',test_labels,'Mode','Cumulative'));
xlabel('Number of trees');
ylabel('Test classification error');
figure(2)
plot(resubLoss(adaBoost, 'Mode', 'Cumulative'));
hold on
plot(resubLoss(lpBoost,'Mode','Cumulative'),'r');
plot(resubLoss(totalBoost,'Mode','Cumulative'),'g');
hold off;
legend('AdaBoostM2','LPBoost','TotalBoost','Location','NE');
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');

