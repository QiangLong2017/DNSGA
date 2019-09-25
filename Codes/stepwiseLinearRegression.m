function PredY = stepwiseLinearRegression(trainX,trainY,testX)

% trainX: input for training, size = #dimension * #points
% trainY: output for training, size = #points * 1
% testX: input for testing, size = #dimension * #points

%%
mdl = stepwiselm(trainX',trainY','Verbose',0);
PredY = predict(mdl,testX');
PredY = PredY';