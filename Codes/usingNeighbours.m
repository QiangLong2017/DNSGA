% Using neighbouts
% TrainX: m input abservations, each has n variables, size = n*m;
% TrainY: m output variables, each has 1 dimension, size = m*1;
% Coeff: k cluster coefficients, each has n+1 variables. The last one is
%           bais. size = k*(n+1)
% TestX: l input abservation, each has n variables, size = n*l
% PredY: k output variables, each has 1 dimension, size = 1*k

function PredY = usingNeighbours(TrainX,TrainY,Coeff,TestX)

numoftrains = size(TrainX,2);
numoftests = size(TestX,2);
numofclusters = size(Coeff,1);
numofneighbours = round(0.3*numoftrains);

PredY = zeros(1,numoftests);
for i = 1:numoftests
    % compute the closest points
    test_x = TestX(:,i);
    temp1 = sum((repmat(test_x,1,numoftrains)-TrainX).^2);
    [~,temp2] = sort(temp1);
    cloest = temp2(1:numofneighbours);
    TrainX_C = TrainX(:,cloest);
    TrainY_C = TrainY(cloest);

    % Attribute training data to clusters
    temp1 = abs(Coeff * [TrainX_C;ones(1,numofneighbours)]-repmat(TrainY_C',numofclusters,1));
    [~,belong] = min(temp1);

    % Find the largest cluster
    weights=zeros(numofclusters,1);
    for j=1:numofclusters
        weights(j)=(sum(belong==j))/length(belong);
    end

    % prediction 
    PredY(i) = weights'*(Coeff*[test_x;1]);
end