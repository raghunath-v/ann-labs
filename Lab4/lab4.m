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
H = 50;             % should test 50, 75, 100, 150.

% params (dimV, dimH, type) type - default: 'BBRBM'
rbm = randRBM(V, H);
%rbmWeights = rbm.W;
%rbmBias = rbm.b;

% Setting of the model
opts.MaxIter = 200;                   % should begin with 10 and 20
opts.InitialMomentumIter = 1;
opts.Verbose = true;
opts.StepRatio = 15;
opts.BatchSize = 100;
pretrainedRBM = pretrainRBM(rbm, trainData, opts);
trainedW = pretrainedRBM.W;
trainedB = pretrainedRBM.b;
% trainedC = pretrainedRBM.c;

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

maxEpochs = 100;
hiddenSize = 25;        % 50, 75, 100, 150
% trainData = trainData(16,:);
autoenc = trainAutoencoder(trainData', hiddenSize, 'MaxEpochs', maxEpochs);
reconstructedImage = predict(autoenc, trainData')';

% error 
mseError = mse(trainData - reconstructedImage);

% Getting the digit image from 0-9
digitNumberIndex = [12 3 9 16 7 4 1 5 32 10];

figure(1);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(2,10,i); 
    imshow(reshape(image, 28, 28));
    subplot(2,10,i+10); 
    imshow(reshape(reconstructedImage(index,:), 28, 28));
end 

%% 3.2 Classification with deeper architectures



