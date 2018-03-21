%% 3.1 RBM
clear all; clc; close all;
load binMNIST

%--------------------------------------------------------------------
% test different eta, epochs (10, 20), different number of hidden nodes
% (50, 75, 100, 150)
%--------------------------------------------------------------------
rng(1)
trainData = bindata_trn;
trainLabel = digtargets_trn;
testData = bindata_tst;
testLabel = digtargets_tst;

N = length(trainData);              % 8000
V = length(trainData(1,:));         % 784 = 28 x 28
H = 150;             % should test 50, 75, 100, 150.

% params (dimV, dimH, type) type - default: 'BBRBM'
rbm = randRBM(V, H);
rbmWeights = rbm.W;
rbmBias = rbm.b;

% Setting of the model
opts.MaxIter = 20;                   % should begin with 10 and 20
opts.InitialMomentumIter = 1;
opts.Verbose = true;
opts.StepRatio = 15;
opts.BatchSize = 100;
pretrainedRBM = pretrainRBM(rbm, trainData, opts);
trainedW = pretrainedRBM.W;
trainedB = pretrainedRBM.b;
% trainedC = pretrainedRBM.c;


%--------------------------------------------------------------------
% image of digits 0-9
%--------------------------------------------------------------------
%{
% Getting the digit image from 0-9
digitNumberIndex = [12 3 9 16 7 4 1 5 32 10];

figure(1);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(1,10,i); 
    imshow(reshape(image, 28, 28));
end 
figure(2);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(1,10,i); 
    hiddenLayer = sigmoid(image*trainedW + trainedB);
    reconstructedImage = trainedW*hiddenLayer';
    imshow(reshape(reconstructedImage, 28, 28));
end
%}
%--------------------------------------------------------------------
% image of the weights
%--------------------------------------------------------------------
%{
figure;
for i = 1:H         % 50 or 100
    subplot(5, 10, i);
    weights = trainedW(:,i);
    imshow(reshape(weights, 28, 28));
end
%}  

%% 3.1 autoencoder (Need to debug, wrong reconstrution image)
clear all; clc; close all;
load binMNIST

%--------------------------------------------------------------------
% test different eta, epochs (10, 20), different number of hidden nodes
% (50, 75, 100, 150)
%--------------------------------------------------------------------

trainData = bindata_trn;
trainLabel = digtargets_trn;
testData = bindata_tst;
testLabel = digtargets_tst;

% N = length(trainData);              % 8000
% V = length(trainData(1,:));         % 784 = 28 x 28
% H = 50;             % should test 50, 75, 100, 150.

maxEpochs = 20;
hiddenSize = 25;        % 50, 75, 100, 150
% trainData = trainData(16,:);
autoenc = trainAutoencoder(trainData, hiddenSize, 'MaxEpochs', maxEpochs);
reconstructedImage = predict(autoenc, trainData);

% error 
mseError = mse(trainData - reconstructedImage);

% Getting the digit image from 0-9
digitNumberIndex = [12 3 9 16 7 4 1 5 32 10];
%{
autoenc.EncoderWeights
autoenc.EncoderBiases
autoenc.DecoderWeights
autoenc.EncoderBiases
%}

%image = trainData(12,:)
%{
disp('here')
encod = sigmoid(autoenc.EncoderWeights*trainData+autoenc.EncoderBiases);
decod = sigmoid(autoenc.DecoderWeights*encod + autoenc.DecoderBiases);

imshow(reshape(decod, 28, 28));
%}
% sigmoid(autoenc.DecoderWeights*image


figure(1);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(1,10,i); 
    imshow(reshape(image, 28, 28));
end 
figure(2);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(1,10,i); 
    %hiddenLayer = sigmoid(image*trainedW + trainedB);
    %reconstructedImage = trainedW*hiddenLayer';
    imshow(reshape(reconstructedImage(index,:), 28, 28));
end

%% 3.2.1 Classification with deeper architectures
%--------------------------------------------------------------------
% * compare performance obtained with different number of hidden layers (1,2
% and 3) 
%
% * Taking the the optimal number of nodes from the experiment of 3.1 then
% decide on the size of the other layers within a similar range with
% tendency to have less and less units.
% 
% 
%--------------------------------------------------------------------
 
clear all; clc; close all;
load binMNIST

%--------------------------------------------------------------------
% test different eta, epochs (10, 20), different number of hidden nodes
% (50, 75, 100, 150)
%--------------------------------------------------------------------

trainData = bindata_trn;
trainLabel = digtargets_trn;
testData = bindata_tst;
testLabel = digtargets_tst;

N = length(trainData);              % 8000
V = length(trainData(1,:));         % 784 = 28 x 28
dims = [V 150 100 V];                             % should test 50, 75, 100, 150.

% params (dimV, dimH, type) type - default: 'BBRBM'
initial_dbn = randDBN(dims);            % default type, Bernoulli-Bernoulli RBMs?

opts.MaxIter = 20;                   % should begin with 10 and 20
opts.InitialMomentumIter = 1;
opts.Verbose = true;
opts.StepRatio = 15;
opts.BatchSize = 100;

dbn = pretrainDBN(initial_dbn, trainData, opts);

testImage = trainData(1,:);

output1 = dbn.rbm{1,1}.W*(trainData(1,:))'
dbn.rbm{1,1}.b
dbn.rbm{2,1}.W
dbn.rbm{2,1}.b
dbn.rbm{3,1}.W
dbn.rbm{3,1}.b




