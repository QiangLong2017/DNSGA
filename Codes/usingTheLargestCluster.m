% Using the largest cluster
% TrainX: m input abservations, each has n variables, size = n*m;
% TrainY: m output variables, each has 1 dimension, size = m*1;
% Coeff: k cluster coefficients, each has n+1 variables. The last one is
%           bais. size = k*(n+1)
% TestX: l input abservation, each has n variables, size = n*l
% PredY: k output variables, each has 1 dimension, size = 1*k

function PredY = usingTheLargestCluster(TrainX,TrainY,Coeff,TestX)

numoftrains = size(TrainX,2);
numoftests = size(TestX,2);
numofclusters = size(Coeff,1);

% Attribute training data to clusters
temp1 = abs(Coeff * [TrainX;ones(1,numoftrains)]-repmat(TrainY',numofclusters,1));
[~,belong] = min(temp1);

% Find the largest cluster
message = tabulate(belong);
[~,largest] = max(message(:,2));

% prediction 
PredY = Coeff(largest,:)*[TestX;ones(1,numoftests)];


