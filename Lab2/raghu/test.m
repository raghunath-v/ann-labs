%% RBF Batchlearning
clc; close all; clear;

units = 63;

xTrain = (0:0.1:2*pi)';
xTest = (0.05:0.1:2*pi)';
sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);
sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);
train_size=length(sinTrainTarget);
test_size = length(sinTestTarget);

%plot( xTrain, squareTrainTarget);
W = rand(units,1);
mu = 0.5;
var = 0.5;
sDev = sqrt(var);
%eta = 0.001;
max_error = 0.1;

epoch = 10000;

xTrainMat = repmat(xTrain,[1,units]);
muMat = repmat(rand(1,units), [train_size, 1]);
sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);

Phi = exp((-(xTrainMat-mu).^2)./(2*sDev.^2));
%Phi = exp((-(xTrainMat-muMat).^2)/(2*sDev^2));
%output = sum(Phi'*W, 2);

f_calc = Phi*W;
newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
sum(Phi*newW-sinTrainTarget)
plot(Phi*newW)
hold on
plot(sinTrainTarget)
hold off
