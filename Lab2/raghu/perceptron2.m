function out = perceptron2(X,T)
    %Parameters
    N_FEATURES = 1;
    N_TRAINDATA = length(X);
    N_HIDDEN = 20; %Size of Hidden Layer
    N_OUTPUTS = 1;
    alpha = 0.7; %Momentum
    Eta =0.01; % Learning Rate
    MaxEpochs = 1000;
    %W=zeros(N_HIDDEN,3);
    %V=zeros(N_OUTPUTS, N_HIDDEN+1); 
    W = normrnd(0,1,[N_HIDDEN N_FEATURES+1]);
    V = normrnd(0,1,[N_OUTPUTS N_HIDDEN+1]);
    delW = zeros(size(W));
    delV = zeros(size(V));
    out = MaxEpochs;
    
    %Add bias
    bias = ones(size(X(:,1)))';
    X= [X'; bias];
    T=T';
    for i=1:MaxEpochs
        %Forward Pass
        Hin=W*X;
        H=[func_phi(Hin); ones(1,N_TRAINDATA)];
        Oin=V*H;
        O=func_phi(Oin)

        %Backward pass
        delO=(O-T).*func_dPhi(O);
        delH=(V'*delO).*func_dPhi(H);
        delH=delH(1:N_HIDDEN,:);

        %Weight Update
        delW=delH*X';
        delV=delO*H';
        %delW = (delW .* alpha) - (delH * X') .* (1-alpha);
        %delV = (delV .* alpha) - (delO * H') .* (1-alpha);
        W = W + delW .* Eta;
        V = V + delV .* Eta;
  
        %Training Evaluation        
        Hin_train=W*X;
        H_train=[func_phi(Hin_train); ones(1,length(X))];
        Oin_train=V*H_train;
        O_train=func_phi(Oin_train);
        errorTrain=sum((O_train-T).^2);
        if (errorTrain < 0.1)
            break
        end
    end
end

function out = func_dPhi(x)
    out = (1+func_phi(x)) .* (1-func_phi(-x)) / 2;
end

function out = func_phi(x)
    out = 2./(1+exp(-x))-1;
end