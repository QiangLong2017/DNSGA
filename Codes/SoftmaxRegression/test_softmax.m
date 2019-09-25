%% 
% This code test softmax regression using mnist data
clear

%% Step 1: Load data
addpath 'E:\BaiduCloud\MachineLearning\TestDataSets\mnist'
images = loadMNISTImages('mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
rmpath 'E:\BaiduCloud\MachineLearning\TestDataSets\mnist'

%% Step 2: Prepare data and train softmax model
inputData = images;
inputSize = size(inputData,1);
numClasses = 10;
lambda = 1e-4; % Weight decay parameter
theta = 0.005*randn(numClasses*inputSize,1);
labels(labels==0) = 10; % Remap 0 to 10
options.maxIter = 100;

[softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options);

%% Step 3: Test the model using mnist data 
addpath 'E:\BaiduCloud\MachineLearning\TestDataSets\mnist'
images = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
rmpath 'E:\BaiduCloud\MachineLearning\TestDataSets\mnist'

inputData = images;
[prob,pred] = softmaxPredict(softmaxModel,inputData);

acc = mean(labels(:)==pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);