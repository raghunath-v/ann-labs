%% RBF Batchlearning
clc; close all; clear;

units = ;


xTrain = (0:0.1:2*pi)';
noise = normrnd(0, sqrt(0.1), [length(xTrain), 1]);
xTrain = xTrain + noise;

xTest = (0.05:0.1:2*pi)';
noise1 = normrnd(0, sqrt(0.1), [length(xTest), 1]);
xTest = xTest + noise1;

sinTrainTarget = sin(2*xTrain)+noise1;
% squareTrainTarget = square(2*xTrain);
sinTestTarget = sin(2*xTest);
% squareTestTarget = square(2*xTest);
train_size=length(sinTrainTarget);
test_size = length(sinTestTarget);
noise = normrnd(0, sqrt(0.1), [length(xTrain), 1]);
noisesinTrain = sinTestTarget + noise;

%plot( xTrain, squareTrainTarget);
%W = rand(units,1);0,1
mu = 0.5;
var = 0.01;
sDev = sqrt(var);

max_error = 0.01;

xTrainMat = repmat(xTrain,[1,units]);
mean = (0:(2*pi/(units-1)):2*pi);
muMat = repmat(mean, [train_size, 1]);
sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);

Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));

newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
newPhi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));
sum((Phi*newW-sinTrainTarget).^2)/length(sinTrainTarget)

%plot(Phi*newW)
%hold on
%plot(sinTrainTarget)
%hold off

%{
W = zeros(1,units);
for n = 1:epoch
    output = zeros(1,units);
    Phi = exp(-(xTrain-mu).^2/(2*sDev^2));
    output = sum(Phi'*W, 2);
    dW = -eta*(output'-sinTrainTarget)*Phi';
    W = W + dW;
    V = output' - sinTrainTarget;
    error = sum(V)/length(V);
    if (error < 0.1)
        disp('Number of iterations:');
        disp(n);
        break;
    end
    oldW = W;
end
%}
