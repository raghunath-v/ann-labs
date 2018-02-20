function out = dPhi(x)
    out = (1+Phi(x)) .* (1-Phi(-x)) / 2;
end

function out = Phi(x)
    out = 2./ (1+exp(-x))-1;
end

function out = perceptron2(X)
    N_HIDDEN=3; %Size of Hidden Layer
    N_OUTPUTS=1;
    W = normrnd(0,1,[N_HIDDEN 3]);
    V = normrnd(0,1,[N_OUTPUTS N_HIDDEN+1]);
    alpha = 0.9;     
    dw = zeros(size(W));
    dv = zeros(size(V));
    MAX_EPOCHS = 1000;
    eta = 0.01;
    for epoch = 1:MAX_EPOCHS
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
end