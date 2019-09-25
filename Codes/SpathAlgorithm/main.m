clear
warning off

load 'E:\BaiduCloud\ClusterwiseLinearRegression\paper_ReviewCLR\Data\LargeScaleDateSets\forestfires.txt'
D=forestfires;
% D=D(:,[2:size(D,2) 1]);
[numofpoint,pointdim]=size(D);
% numofcluster=input('Please input numofcluster: ');
numofcluster=5;


[sumoferror,coefficient,indi_point_k]=MultstartSpathAlgorithm(D,numofcluster);