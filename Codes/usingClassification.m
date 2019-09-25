% Using classification
% TrainX: m input abservations, each has n variables, size = n*m;
% TrainY: m output variables, each has 1 dimension, size = 1*m;
% Coeff: k cluster coefficients, each has n+1 variables. The last one is
%           bais. size = k*(n+1)
% TestX: l input abservation, each has n variables, size = n*l
% TestY: k output variables, each has 1 dimension, size = 1*k

function PredY = usingClassification(TrainX,TrainY,Coeff,TestX)

numoftrains = size(TrainX,2);
numoftests = size(TestX,2);
numofclusters = size(Coeff,1);

% Attribute training data to clusters
temp1 = abs(Coeff * [TrainX;ones(1,numoftrains)]-repmat(TrainY',numofclusters,1));
[~,belong] = min(temp1);

% Train a classifier 
classifier = fitcecoc(TrainX',belong');

% Attribute testing data to clusters
test_belong = predict(classifier,TestX');

% prediction
PredY = zeros(1,numoftests);
for i = 1:numoftests
    test_x = TestX(:,i);
    PredY(i) = Coeff(test_belong(i),:)*[test_x;1];
end