testImage = trainData(1,:);

firstLayerW = dbn.rbm{1,1}.W;
firstLayerb = dbn.rbm{1,1}.b;

secondLayerW = dbn.rbm{2,1}.W;
secondLayerb = dbn.rbm{2,1}.b;

thirdLayerW = dbn.rbm{3,1}.W;
thirdLayerb = dbn.rbm{3,1}.b;

hidden1 = sigmoid(testImage*firstLayerW + firstLayerb); % 1x150
hidden2 = sigmoid(hidden1*secondLayerW + secondLayerb); % 1x100
% hiddenoutput = sigmoid(hidden2*thirdLayerW + thirdLayerb); % 1 x 784


Initial_weight = zeros(size(input, 2), 1);
