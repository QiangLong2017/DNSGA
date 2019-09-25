% Using classification
% TrainX: m input abservations, each has n variables, size = n*m;
% TrainY: m output variables, each has 1 dimension, size = m*1;
% Coeff: k cluster coefficients, each has n+1 variables. The last one is
%           bais. size = k*(n+1)
% TestX: l input abservation, each has n variables, size = n*l
% PredY: k output variables, each has 1 dimension, size = 1*k


function PredY = usingSoftmaxRegression(TrainX,TrainY,Coeff,TestX)

addpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\softmaxRegression';

[inputSize,numoftrains] = size(TrainX);
numoftests = size(TestX,2);
numofclusters = size(Coeff,1);

% Attribute training data to clusters
temp1 = abs(Coeff * [TrainX;ones(1,numoftrains)]-repmat(TrainY',numofclusters,1));
[~,belong] = min(temp1);

% Train a multiclassifier
lambda = 1e-4; % weight decay parameter
options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numofclusters, lambda, TrainX, belong, options);

% Attribute testing data to clusters
[prob,pred] = softmaxPredict(softmaxModel,TestX);

% prediction
for i =1:numoftests
    PredY(i) = prob(:,i)'*(Coeff*[TestX(:,i);1]);
end

rmpath 'E:\BaiduCloud\Lectures\硕士研究生指导\吴雪\Codes\softmaxRegression';