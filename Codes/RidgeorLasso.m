function PredY = RidgeorLasso(trainX,trainY,testX)

% trainX: input for training, size = #dimension * #points
% trainY: output for training, size = #points * 1
% testX: input for testing, size = #dimension * #points

%% ridge
coeffi = ridge(trainY,trainX',5);
PredY = (coeffi(:))'*testX;

% %% Lasso
% [coeffi,FitInfo] = lasso(trainX',trainY);
% PredY = (coeffi(:,70))'*testX;