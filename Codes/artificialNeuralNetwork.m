function PredY = artificialNeuralNetwork(trainX,trainY,testX)

% trainX: input for training, size = #dimension * #points
% trainY: output for training, size = #points * 1
% testX: input for testing, size = #dimension * #points

%%
net = fitnet(10);
% net = feedforwardnet(10);
net = train(net,trainX,trainY');
% view(net)
PredY = net(testX);