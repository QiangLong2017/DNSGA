function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses:   the number of classes 
% inputSize:    the size N of the input vector
% lambda:       weight decay parameter
% data:         the N x M input matrix, where each column data(:, i) corresponds to
%               a single test set
% labels:       an M x 1 matrix containing the labels corresponding for the input data

%% main code

theta = reshape(theta, numClasses, inputSize); % Unroll the parameters from theta
numCases = size(data, 2); % number of cases
groundTruth = full(sparse(labels, 1:numCases, 1)); % tansfer labels in to one hot variables

eta = bsxfun(@minus,theta*data,max(theta*data,[],1)); % I do not konw why we need this
eta = exp(eta);
pij = bsxfun(@rdivide,eta,sum(eta));

% value of objective function value
cost = -1./numCases*sum(sum(groundTruth.*log(pij)))+lambda/2*sum(sum(theta.^2));

% grad of objective function value
thetagrad = -1/numCases.*(groundTruth-pij)*data'+lambda.*theta; % size: numClasses*inputSize
grad = thetagrad(:); % Unroll the gradient matrices into a vector for minFunc