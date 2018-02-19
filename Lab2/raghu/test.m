%% RBF Batchlearning
clc; close all; clear;

units = 70;

xTrain = (0:0.1:2*pi)';
xTest = (0.05:0.1:2*pi)';
sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);
sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);
train_size=length(sinTrainTarget);
test_size = length(sinTestTarget);

%plot( xTrain, squareTrainTarget);
%W = rand(units,1);
mu = 0.5;
var = 0.01;
sDev = sqrt(var);

max_error = 0.1;

xTrainMat = repmat(xTrain,[1,units]);
mean = (0:(2*pi/(units-1)):2*pi);
muMat = repmat(mean, [train_size, 1]);
sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);

Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));

newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
sum((Phi*newW-sinTrainTarget).^2)/length(sinTrainTarget)
    
%plot(Phi*newW)
%hold on
%plot(sinTrainTarget)
%hold off


