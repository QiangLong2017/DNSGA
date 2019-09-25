function [sumoferror,coefficient,indi_point_k]=MultstartSpathAlgorithm(D,numofcluster)

% D: input + output, size = numofpoints*(numofinputdim+1)
% numofcluster: a scale number
% sumoferror: a scale number, the sum of linear error of each point
% coefficient: linear coefficients for each clusters, size =numofclusters*(numofinputdim+1)
% indi_point_k: index of cluster for each point belongs

% Algorithm parameters
numoftraining=1;
numofpoint=size(D,1); % D numofpoint*numofdim

%==========================================================================

sumoferror=inf;
for iter=1:numoftraining
    % intial randomized partition
    for k=1:numofpoint
        indi_point_k1(k)=mod(randi(100*numofcluster),numofcluster)+1;
    end
    
    % call Spath algorithm
    [sumoferror1,coefficient1,indi_point_k1]=SpathAlgorithm(D,indi_point_k1,numofcluster);
    
    % update the best sum of error
    if sumoferror1>=sumoferror
        fprintf('%d\n',iter);
    end
    if sumoferror1<sumoferror
        sumoferror=sumoferror1;
        coefficient=coefficient1;
        indi_point_k=indi_point_k1;
        fprintf('%d\t%f\n',iter,sumoferror);
    end
end