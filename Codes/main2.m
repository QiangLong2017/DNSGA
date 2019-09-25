%% This code compares CLR-based predictor with other predictors

%% load dataset
addpath 'E:\BaiduCloud\ClusterwiseLinearRegression\paper_ReviewCLR\Data\LargeScaleDateSets';
load parkinson2.txt
XY = parkinson2;
rmpath 'E:\BaiduCloud\ClusterwiseLinearRegression\paper_ReviewCLR\Data\LargeScaleDateSets'
numofcluster = 5; % number of clusters for CLR training

%% data preprocess: normalize
[X,PSX] = mapminmax((XY(:,1:end-1))');
[Y,PSY] = mapminmax((XY(:,end))');
XY = [X' Y'];

%% separate dataset into training and testing set, build X and Y
% randomly separate XY into 80% and %20, i.e., training and testing sets.
numOfPoints = size(XY,1); % total number of points in dataset
index = randperm(numOfPoints);
trainIndex = index(1:ceil(numOfPoints*0.8));
testIndex = index(ceil(numOfPoints*0.8)+1:end);
trainXY = XY(trainIndex,:);
testXY = XY(testIndex,:);
% build X and Y
trainX = trainXY(:,1:end-1);  % #points * #dimension
trainY = trainXY(:,end);      % #points * 1
testX = testXY(:,1:end-1);    % #points * #dimension
testY = testXY(:,end);        % #points * 1  

%% Training the CLR models using the Multistart Spath Algorithm
addpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\my_SpathAlgorithm'
[sumOfError,Coeff,labels] = MultstartSpathAlgorithm(trainXY,numofcluster);
rmpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\my_SpathAlgorithm'

%% prediction using different CLR_based predictors
trainX = trainX';  % #dimension * #points
testX = testX';    % #dimension * #points

% using the largest cluster
PredY_Lagest = usingTheLargestCluster(trainX,trainY,Coeff,testX);

% using weights
PredY_Weight = usingWeights(trainX,trainY,Coeff,testX);

% using neighbours
PredY_neighbours = usingNeighbours(trainX,trainY,Coeff,testX);

% using distance
PredY_Distance = usingDistance(trainX,trainY,Coeff,testX);

% using classification
PredY_Classification = usingClassification(trainX,trainY,Coeff,testX);

% using softmax regression
PredY_softmaxRegreesion = usingSoftmaxRegression(trainX,trainY,Coeff,testX);

%% Prediction using other predictors
% linear regression
PredY_linearRegression = linearRegression(trainX,trainY,testX);

% artificial neural network
PredY_ann = artificialNeuralNetwork(trainX,trainY,testX);

% Ridge Regression
PredY_rr = RidgeorLasso(trainX,trainY,testX);

% Stepwise linear regression
PredY_slr = stepwiseLinearRegression(trainX,trainY,testX);

%% anti-normalize
testY = mapminmax('reverse',testY',PSY);
PredY_Lagest = mapminmax('reverse',PredY_Lagest,PSY);
PredY_Weight = mapminmax('reverse',PredY_Weight,PSY);
PredY_neighbours = mapminmax('reverse',PredY_neighbours,PSY);
PredY_Distance = mapminmax('reverse',PredY_Distance,PSY);
PredY_Classification = mapminmax('reverse',PredY_Classification,PSY);
PredY_softmaxRegreesion = mapminmax('reverse',PredY_softmaxRegreesion,PSY);
PredY_linearRegression = mapminmax('reverse',PredY_linearRegression,PSY);
PredY_ann = mapminmax('reverse',PredY_ann,PSY);
PredY_rr = mapminmax('reverse',PredY_rr,PSY);
PredY_slr = mapminmax('reverse',PredY_slr,PSY);

%% Performance evaluation
testY = testY';
diff = [testY-PredY_Classification',testY-PredY_softmaxRegreesion',...
    testY-PredY_linearRegression',testY-PredY_ann',testY-PredY_rr',...
    testY-PredY_slr'];

% mean absolute error
MAE = mean(abs(diff));
% Mean relative error
MARE = mean(abs(diff./repmat(testY+1,1,size(diff,2))));
% Mean absolute scaled error
MASE = mean(diff./(ones(length(testY),size(diff,2))*mean(testY(2:end)-testY(1:end-1))));
% Nash-Sutcliffe coefficient of efficiency
CE = 1-sum(diff.^2)./repmat(sum((testY-mean(testY)).^2),1,size(diff,2));

res = [MAE;MARE;MASE;CE]