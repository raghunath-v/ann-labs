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
H = 75;             % should test 50, 75, 100, 150.

% params (dimV, dimH, type) type - default: 'BBRBM'
rbm = randRBM(V, H);
%rbmWeights = rbm.W;
%rbmBias = rbm.b;

% Setting of the model
opts.MaxIter = 100;                   % should begin with 10 and 20
opts.InitialMomentumIter = 1;
opts.Verbose = true;
opts.StepRatio = 0.1;
opts.BatchSize = 1000;
% pretrainedRBM = pretrainRBM(rbm, trainData, opts);
% trainedW = pretrainedRBM.W;
% trainedB = pretrainedRBM.b;
% trainedC = pretrainedRBM.c;

% Getting the digit image from 0-9
digitNumberIndex = [12 3 9 16 7 4 1 5 32 10];

figure(1);
hold on
title('Error against epochs');
for i = 100
    rbm = randRBM(V, i);
    pretrainedRBM = pretrainRBM(rbm, trainData, opts);
    plot(pretrainedRBM.errorPlot(:,1),pretrainedRBM.errorPlot(:,2));
end
hold off

figure(2);
for i = 1:10
    index = digitNumberIndex(i);
    image = trainData(index,:);
    subplot(2,10,i); 
    imshow(reshape(image, 28, 28)');
    hiddenLayer = sigmoid(image*pretrainedRBM.W + pretrainedRBM.b);
    reconstructedImage = pretrainedRBM.W*hiddenLayer';
    subplot(2,10,i+10); 
    imshow(reshape(reconstructedImage, 28, 28)');
end 

figure(3);
title(Weights)
for i = 1:H         % 50 or 100
    subplot(H/10, 10, i);
    weights = pretrainedRBM.W(:,i);
    imshow(reshape(weights, 28, 28)');
end


%% 3.1 autoencoder 
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
hiddenSize = 100;        % 50, 75, 100, 150
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
    imshow(reshape(image, 28, 28)');
    subplot(2,10,i+10); 
    imshow(reshape(reconstructedImage(index,:), 28, 28)');
end 

%% 3.2 Classification with deeper architectures
clear all; clc; close all;
load binMNIST

trainData = bindata_trn;
%trainLabel = digtargets_trn;
testData = bindata_tst;
%testLabel = digtargets_tst;

trainLabel = zeros([length(bindata_trn) 10]);
for i = 1:8000
    trainLabel(i,digtargets_trn(i)+1) = 1;
end
testLabel = zeros([length(bindata_tst) 10]);
for i = 1:2000
    testLabel(i,digtargets_tst(i)+1) = 1;
end

N = length(trainData);              % 8000
V = length(trainData(1,:));         % 784 = 28 x 28

dims = [V 150 100 50 10]; 
initial_dbn = randDBN(dims);

opts.MaxIter = 100;                   % should begin with 10 and 20
opts.InitialMomentumIter = 1;
opts.Verbose = true;
opts.StepRatio = 0.1;
opts.BatchSize = 1000;

dbn = pretrainDBN(initial_dbn, trainData, opts);

dbn= SetLinearMapping(dbn, trainData, trainLabel);

opts.Layer = 0;
dbn = trainDBN(dbn, trainData, trainLabel, opts);


rmse= CalcRmse(dbn, trainData, trainLabel);
ErrorRate= CalcErrorRate(dbn, trainData, trainLabel);
fprintf( 'For training data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );
fprintf( 'Accuracy: %g\n', (1-ErrorRate)*100 );

rmse= CalcRmse(dbn, testData, testLabel);
ErrorRate= CalcErrorRate(dbn, testData, testLabel);
fprintf( 'For test data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );
fprintf( 'Accuracy: %g\n', (1-ErrorRate)*100 );


