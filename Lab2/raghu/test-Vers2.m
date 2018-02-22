%% RBF Batchlearning
clc; close all; clear;

units = 29;
rng(4)

xTrain = (0:0.1:2*pi)';
noise = normrnd(0, sqrt(0.1), [length(xTrain), 1]);
% xTrain = xTrain ;


xTest = (0.05:0.1:2*pi)';
noise1 = normrnd(0, sqrt(0.1), [length(xTest), 1]);
%xTest = xTest;

sinTrainTarget = sin(2*(xTrain)) + noise;
% squareTrainTarget = square(2*xTrain);
sinTestTarget = sin(2*(xTest ));
% squareTestTarget = square(2*xTest);
train_size=length(sinTrainTarget);
test_size = length(sinTestTarget);

%plot( xTrain, squareTrainTarget);
%W = rand(units,1);0,1
%mu = 0.5;
var = 0.10;
sDev = sqrt(var);

xTrainMat = repmat(xTrain,[1,units]);
range = 2*pi;
mean = (0:(range/(units-1)):range);
mean = sort(2*pi*rand([1 units]));
%mean = mean/0.9
muMat = repmat(mean, [train_size, 1]);
% sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);

Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));

newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
xTestMat = repmat(xTest,[1,units]);
newPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
sum((newPhi*newW-sinTestTarget).^2)/length(sinTestTarget)


figure;
hold on
%yyaxis left           % plot against left y-axis 
%plot(xTrain, sinTrainTarget)        
%plot(xTrain, Phi*newW)
plot(xTest, sinTestTarget)
plot(xTest, newPhi*newW)

legend('TestData','Testfit')
hold off
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
