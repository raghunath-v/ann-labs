clear all 
close all
% 3.1 Assignment

DEBUG_MODE=[0 1];

N_FEATURES=8;
load('data8.mat')

%Plot Data
figure();
hold on
plot(c_1(:, 1), c_1(:,2), 'o');
plot(c_2(:, 1), c_2(:,2), '+');
hold off

% 3.1.2
% How quickly do the algorithms converge? 
% Plot the learning curves for each variant of learning
% Visualise the learning process by plotting a separating line (decision
% boundary) after each epoch of training

% TODO, make trainingdata integer.


trainPercent = 1;
trainingDataSize = size(shuffledData,1)*trainPercent;

trainData = shuffledData(1:trainingDataSize,:);
testData = shuffledData((trainingDataSize+1):end,:);

N_TRAINDATA=int16(trainPercent*2*N_DATA_PER_CLASS);
N_TESTDATA=int16((1-trainPercent)*2*N_DATA_PER_CLASS);

%For auto-encoder
N_TRAINDATA=8;
N_TESTDATA=0;
% 1. Select random sample from training set as input
%trainData = trainingData_label(:,1:2);
%trainLabel = trainingData_label(:,3:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delta rule
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bias=ones(size(trainData(:,1)))';
X= [trainData(:,1:N_FEATURES)'; bias];
bias_test=ones(size(testData(:,1)))';
X_test=[testData(:,1:N_FEATURES)'; bias_test];

T= trainData(:,N_FEATURES+1)';
T_test= testData(:,N_FEATURES+1)';

Eta=0.0001; % Learning Rate

W=zeros(1,N_FEATURES+1);
x_axis=-5:0.1:5;
hold on
for i=1:20
    dW= -Eta*(W*X-T)*X';
    dW= -Eta*(2*(W*X-T)-1)*X';
    %if(dW<0 && )
    W=W+dW;
    y_axis=-W(1)*x_axis/W(2)-W(3)/W(2);
    if(DEBUG_MODE(1)==1)
        plot(x_axis,y_axis)
    end
end

if(DEBUG_MODE(1)==1)
        %plot(x_axis,y_axis)
end

hold off
 %plot(x_axis,y_axis)

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-layer Perceptron
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%Parameters
N_HIDDEN=3; %Size of Hidden Layer
N_OUTPUTS=8;
alpha = 0.7; %Momentum
Eta=0.0001; % Learning Rate
MaxEpochs = 120;
T=X(1:8,:);
%W=zeros(N_HIDDEN,3);
%V=zeros(N_OUTPUTS, N_HIDDEN+1); 
W = normrnd(0,1,[N_HIDDEN N_FEATURES+1]);
V = normrnd(0,1,[N_OUTPUTS N_HIDDEN+1]);
delW = zeros(size(W));
delV = zeros(size(V));

%X(3,:)=-X(3,:);


ErrorList=[];
ErrorListTest=[];
for Epochs= 1:MaxEpochs
    for i=1:Epochs
        %Forward Pass
        Hin=W*X;
        H=[Phi(Hin); ones(1,N_TRAINDATA)]
        Oin=V*H;
        O=Phi(Oin);

        %Backward pass
        delO=(O-T).*dPhi(O);
        delH=(V'*delO).*dPhi(H);
        delH=delH(1:N_HIDDEN,:);

        %Weight Update
        %delW=Eta*delH*X';
        %delV=Eta*delO*H';
        delW = (delW .* alpha) - (delH * X') .* (1-alpha);
        delV = (delV .* alpha) - (delO * H') .* (1-alpha);
        W = W + delW .* Eta;
        V = V + delV .* Eta;
        W;
        %Training Evaluation        
        Result=2*(O>0)-1;
        Error = sum(sum(abs(Result-T)))/(2*double(N_TRAINDATA));
        
        %Testing Evaluation
        Hin_test=W*X_test;
        H_test=[Phi(Hin_test); ones(1,N_TESTDATA)];
        Oin_test=V*H_test;
        O_test=Phi(Oin_test);
        ResultTest=2*(O_test>0)-1;
        %ErrorTest = sum(sum(abs(ResultTest-T_test)))/(2*double(N_TESTDATA));
    end
    ErrorList=[ErrorList Error];
    %ErrorListTest=[ErrorListTest ErrorTest];
end


%Plot
figure();
hold on
plot([1:Epochs],ErrorList(1:Epochs));
%plot([1:Epochs],ErrorListTest(1:Epochs));
legend('Training Error','Test Error','Location','northeast')
hold off



