% data point is exchanged cluster by cluster

function [sumoferror,coefficient,indi_point_k]=SpathAlgorithm(D,indi_point_k,numofcluster)

[numofpoint,pointdim]=size(D);
coefficient=zeros(numofcluster,pointdim);

%% Calculate the initial partition
% Calculate coefficients
for j=1:numofcluster
   coefficient(j,:)=(EvalCoefficient(D(indi_point_k==j,:)))';
end

% Restore the old partition
oldPartition=indi_point_k;

% Clculate old sumoferror
error=(repmat(D(:,pointdim),1,numofcluster)-[D(:,1:pointdim-1) ones(numofpoint,1)]*coefficient').^2;
sumoferror=0;
for i=1:numofpoint
    sumoferror=sumoferror+error(i,oldPartition(i));
end

%% Repeatedly allocate clusters until stable, that is no point change cluster

while 1
%     part=zeros(1,numofcluster);
%     for j=1:numofcluster
%         part(j)=sum(indi_point_k==j);
%     end
%     fprintf('%f  ',[part sumoferror]);
%     fprintf('\n');
%     
    % Allocate points according to minimum error
    [~,indi_point_k]=min(error,[],2);
    
    % Calculate coefficients
    for j=1:numofcluster
        coefficient(j,:)=(EvalCoefficient(D(indi_point_k==j,:)))';
    end

    % Restore the new partition
    newPartition=indi_point_k;

    % Clculate new sumoferror
    error=(repmat(D(:,pointdim),1,numofcluster)-[D(:,1:pointdim-1) ones(numofpoint,1)]*coefficient').^2;
    sumoferror=0;
    for i=1:numofpoint
        sumoferror=sumoferror+error(i,newPartition(i));
    end
      
    % Check if the oldPartition equals the newPartition
    indicator=isOldEqualsNew(oldPartition,newPartition,numofcluster);
    
    % Stopping checking
    if indicator==1 % the oldPartion equals the newPartition
        return;
    else
        oldPartition=newPartition;
    end
end