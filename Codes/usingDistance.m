% Using distance
% TrainX: m input abservations, each has n variables, size = n*m;
% TrainY: m output variables, each has 1 dimension, size = m*1;
% Coeff: k cluster coefficients, each has n+1 variables. The last one is
%           bais. size = k*(n+1)
% TestX: l input abservation, each has n variables, size = n*l
% PredY: k output variables, each has 1 dimension, size = 1*k

function PredY = usingDistance(TrainX,TrainY,Coeff,TestX)

numoftrains = size(TrainX,2);
numoftests = size(TestX,2);
numofclusters = size(Coeff,1);
numofneighbours = numofclusters;  %round(0.3*numofclusters);

% Attribute training data to clusters
temp1 = abs(Coeff * [TrainX;ones(1,numoftrains)]-repmat(TrainY',numofclusters,1));
[~,belong] = min(temp1);

% calculate center point of each cluster
centerpoints = zeros(size(TrainX,1),numofclusters);
for i = 1:numofclusters
    centerpoints(:,i) = mean(TrainX(:,belong==i),2);
end

PredY = zeros(1,numoftests);
for i = 1:numoftests
    % compute the distance to each cluster
    test_x = TestX(:,i);
    distance = sum((repmat(test_x,1,numofclusters)-centerpoints).^2);
    [~,temp2] = sort(distance);
    clostest = temp2(1:numofneighbours);
    
    % cloest clusters
    clostCoeff = Coeff(clostest,:);
    
    % calculate weights
    clostestdis = distance(clostest);
    weights = clostestdis.^(-2)./sum(clostestdis.^(-2));
    
    % prediction 
    PredY(i) = weights*(clostCoeff*[test_x;1]);
end