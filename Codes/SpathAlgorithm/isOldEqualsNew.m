function indicator=isOldEqualsNew(oldPartition,newPartition,numofcluster)

for i=1:numofcluster
    temp1=find(oldPartition==i);
    indi=0;
    for j=1:numofcluster
        temp2=find(newPartition==j);
        if isequal(temp1,temp2)==1
            indi=1; % one cluster did not change
            break;
        end
    end
    if indi==0
        indicator=0; % at least one cluster changed
        return;
    end
end
indicator=1;

        


