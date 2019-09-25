function PredY = linearRegression(trainX,trainY,testX)

% trainX: input for training, size = #dimension * #points
% trainY: output for training, size = #points * 1
% testX: input for testing, size = #dimension * #points

%% 
[~,numofpoint]=size(trainX);

% training
C = [trainX' ones(numofpoint,1)];
d = trainY;
coeffi=lsqlin(C,d);

% predicting
PredY = [testX' ones(size(testX,2),1)]*coeffi;
PredY = PredY'; 
