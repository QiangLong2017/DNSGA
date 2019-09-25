function f=overallfitnessfun(D,coefficient,index)

f=0;
[numofpoint,pointdim]=size(D);

for k=1:numofpoint
    f=f+(D(k,pointdim)-[D(k,1:pointdim-1) 1]*coefficient(index(k),:)')^2;
end