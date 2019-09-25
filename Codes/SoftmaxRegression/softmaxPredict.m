function [prob,pred] = softmaxPredict(softmaxModel, data)

% softmaxModel: model trained using softmaxTrain
% data          the N x M input matrix, where each column data(:, i) corresponds to
%               a single test set
%
% Your code should produce the prediction matrix 
% prob:     prob(:,i) is the probility of data(:,i) belonging to each label
% pred:     pred(i) is the index of the largest one in prob(:,i)

theta = softmaxModel.optTheta;  % this provides a numClasses * inputSize matrix
temp1 = exp(theta*data);
prob = temp1./repmat(sum(temp1,1),softmaxModel.numClasses,1);
[~,pred] = max(prob,[],1);