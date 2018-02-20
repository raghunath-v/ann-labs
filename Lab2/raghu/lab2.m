%% Section 3.1 
% RBF Batchlearning
clc; close all; clear;

units = 26;

genData;
load('data')

%plot( xTrain, squareTrainTarget);
mu = 0.5;
var = 0.01;
sDev = sqrt(var);

max_error = 0.1;

% RBF parameters
xTrainMat = repmat(xTrain,[1,units]);
xTestMat = repmat(xTest,[1,units]);
mean = (0:(2*pi/(units-1)):2*pi);
muMat = repmat(mean, [train_size, 1]);
sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);
Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));


newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
error = sum((Phi*newW-sinTrainTarget).^2)/length(sinTrainTarget);

% Running the algorithm for and checking for different errors if they
% have reached below 0.1, 0.01, 0.001 etc. Change units everytime.



%% Section 3.2 
%Data
noise = normrnd(0, sqrt(0.1), size(xTrain));
sinTrainTarget_noisy = sinTrainTarget + noise;
%plot(sinTrainTarget_noisy)

W = zeros(units,1);
eta = 0.01; %Learning rate
MAX_ERRROR = 0.1;
MAX_EPOCHS = 10000;
%Online mode learning
for n = 1:MAX_EPOCHS
    error = 0;
    for i=1:length(xTrain)
        phi = exp((-(xTrain(i)-mean).^2)/(2*sDev.^2));
        output = phi*W;
        dW = -eta*(output-sinTrainTarget_noisy(i))*phi;
        W = W + dW';
        error = error + (output - sinTrainTarget_noisy(i))^2;
    end
    error = error/length(xTrain);
    if error<MAX_ERRROR
        disp('Number of epochs')
        disp(n)
        break;
    end
end

testPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
plot(testPhi*W);

% Two layer perceptron

