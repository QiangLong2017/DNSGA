function coeffi=EvalCoefficient(subsetofD)

[numofpoint,pointdim]=size(subsetofD);

% least square regression
C=[subsetofD(:,1:pointdim-1),ones(numofpoint,1)];
d=subsetofD(:,pointdim);
coeffi=lsqlin(C,d);

% ridge regression
% C=subsetofD(:,1:pointdim-1);
% d=subsetofD(:,pointdim);
% coeffi=ridge(d,C,5,0);
% coeffi=[coeffi(end);coeffi(1:end-1)];