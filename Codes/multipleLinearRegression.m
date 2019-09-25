function PredY = multipleLinearRegression(trainX,trainY,testX)

% trainX: input for training, size = #dimension * #points
% trainY: output for training, size = #points * 1
% testX: input for testing, size = #dimension * #points

%%
coeffi = ridge(trainY,trainX,5);

mdl = fitlm(trainX',trainY','linear','RobustOpts','on');
PredY = predict(mdl,testX');
PredY = PredY';