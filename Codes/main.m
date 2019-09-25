%% This code compares different CLR-based predictors

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
trainX = trainXY(:,1:end-1);
trainY = trainXY(:,end);
testX = testXY(:,1:end-1);
testY = testXY(:,end);

%% Training the CLR models using the Multistart Spath Algorithm
addpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\my_SpathAlgorithm'
[sumOfError,Coeff,labels] = MultstartSpathAlgorithm(trainXY,numofcluster);
rmpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\my_SpathAlgorithm'

%% prediction using different methods
trainX = trainX';
testX = testX';

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

%% anti-normalize
testY = mapminmax('reverse',testY',PSY);
testY = testY';
PredY_Lagest = mapminmax('reverse',PredY_Lagest,PSY);
PredY_Weight = mapminmax('reverse',PredY_Weight,PSY);
PredY_neighbours = mapminmax('reverse',PredY_neighbours,PSY);
PredY_Distance = mapminmax('reverse',PredY_Distance,PSY);
PredY_Classification = mapminmax('reverse',PredY_Classification,PSY);
PredY_softmaxRegreesion = mapminmax('reverse',PredY_softmaxRegreesion,PSY);
% PredY = [testY PredY_Lagest' PredY_Weight' PredY_neighbours' PredY_Distance' PredY_Classification' PredY_softmaxRegreesion'];
% disp(PredY);

%% Performance evaluation
% mean absolute error
MAE = mean(abs([testY-PredY_Lagest',testY-PredY_Weight',testY-PredY_neighbours',...
    testY-PredY_Distance',testY-PredY_Classification', ...
    testY-PredY_softmaxRegreesion']));
% Mean relative error
MARE = mean(abs([testY-PredY_Lagest',testY-PredY_Weight',testY-PredY_neighbours',...
    testY-PredY_Distance',testY-PredY_Classification', ...
    testY-PredY_softmaxRegreesion']./repmat(testY+1,1,6)));
% Mean absolute scaled error
MASE = mean([testY-PredY_Lagest',testY-PredY_Weight',testY-PredY_neighbours',...
    testY-PredY_Distance',testY-PredY_Classification', ...
    testY-PredY_softmaxRegreesion']./(ones(length(testY),6)*mean(testY(2:end)-testY(1:end-1))));
% Nash-Sutcliffe coefficient of efficiency
CE = 1-sum([testY-PredY_Lagest',testY-PredY_Weight',testY-PredY_neighbours',...
    testY-PredY_Distance',testY-PredY_Classification', ...
    testY-PredY_softmaxRegreesion'].^2)./repmat(sum((testY-mean(testY)).^2),1,6);

res = [MAE;MARE;MASE;CE]