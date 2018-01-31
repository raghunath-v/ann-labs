% 3.1 Assignment
clear all
close all

DEBUG_MODE=[0 1];
N_DATA_PER_CLASS=100;


% 3.1.1
rng default

%Generate Data
% class 1
mu_1 = [3, 3];
sigma_1 = [1, -1.5; -1.5, 10];
c_1 = mvnrnd(mu_1, sigma_1, N_DATA_PER_CLASS);

c1_label = ones(size(c_1,1), 1);

c_1 = [c_1 c1_label]; % the third column is the label


% class 2
mu_2 = [-3, -3];
sigma_2 = [1, 1.5; 1.5, 10];
c_2 = mvnrnd(mu_2, sigma_2, N_DATA_PER_CLASS);
c_2_size = size(c_2);
c2_label = -1*(ones(size(c_2, 1), 1));
c_2 = [c_2 c2_label];

% shuffle data into one dataset
data = [c_1; c_2];
shuffledData = data(randperm(size(data,1)), :);

if(DEBUG_MODE(1)==1)
    hold on
    plot(c_1(:, 1), c_1(:,2), 'o');
    plot(c_2(:, 1), c_2(:,2), '+');
    
end

% 3.1.2
% How quickly do the algorithms converge? 
% Plot the learning curves for each variant of learning
% Visualise the learning process by plotting a separating line (decision
% boundary) after each epoch of training

% TODO, make trainingdata integer.


trainPercent = 0.7;
trainingDataSize = size(shuffledData,1)*trainPercent;

trainData = shuffledData(1:trainingDataSize,:);
testData = shuffledData((trainingDataSize+1):end,:);

N_TRAINDATA=trainPercent*2*N_DATA_PER_CLASS;
% 1. Select random sample from training set as input
%trainData = trainingData_label(:,1:2);
%trainLabel = trainingData_label(:,3:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delta rule
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bias=ones(size(trainData(:,1)))';
X= [trainData(:,1:2)'; bias];

T= trainData(:,3)';

Eta=0.0001; % Learning Rate

W=zeros(1,3);
x_axis=-5:0.1:5;

for i=1:20
    dW= -Eta*(W*X-T)*X';
    %if(dW<0 && )
    W=W+dW;
    y_axis=-W(1)*x_axis/W(2)-W(3)/W(2);
    if(DEBUG_MODE(1)==1)
        plot(x_axis,y_axis)    
    end
end
 %plot(x_axis,y_axis)

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2-layer Perceptron
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%Size of Hidden Layer
N_HIDDEN=3;
N_OUTPUTS=1;
%W=zeros(N_HIDDEN,3);
%V=zeros(N_OUTPUTS, N_HIDDEN+1); 
W = normrnd(0,1,[N_HIDDEN 3])
V = normrnd(0,1,[N_OUTPUTS N_HIDDEN+1])
alpha = 0.9;     
dw = zeros(size(W));
dv = zeros(size(V));
epochs = 200;
eta = 0.0001;


<<<<<<< HEAD
%{
delta_o = (O-T).*((1+O) .*(1-O))*0.5;
delta_h = (V' * delta_o) .* ((1+H) .* (1-H))*0.5;
delta_h = delta_h(1:N_HIDDEN, :);
%}
=======

% delta_o = (O-T).*((1+O) .*(1-O))*0.5;
% delta_h = (V' * delta_o) .* ((1+H) .* (1-H))*0.5;
% delta_h = delta_h(1:N_HIDDEN, :);
>>>>>>> b97707a0838adb1f2cc7f3130f604f0163e18641
for epoch = 1:epochs
    H_in=W*X;        
    H=[Phi(H_in); ones(1,N_TRAINDATA)];
    O_in=V*H;       
    O=Phi(O_in);    
    delta_o = (O-T).*((1+O) .*(1-O))*0.5;
    delta_h = (V' * delta_o) .* ((1+H) .* (1-H))*0.5;
    delta_h = delta_h(1:N_HIDDEN, :);
    dw = (dw .* alpha) - (delta_h * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * H') .* (1-alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;
    
end

O






