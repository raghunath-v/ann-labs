function out = isAttractor(x)
    N = length(x);
    MAX_EPOCHS = 10;
    Xmat = repmat(x,[N,1]); 
    Xd = Xmat.*(-sgn(eye(N)-1));
    Xd = Xd(randperm(size(Xd,1)),:);
    
    W = x'*x;
    out = false;
    for epoch = 1:MAX_EPOCHS
        error = sum(sum(abs(Xmat-sgn(Xd*W))));
        for i=1:size(Xd,1)
            W = W + Xd(i,:)'*Xd(i,:);
        end
        if(error==0)
            disp('Num of epochs');
            disp(epoch);
            out = true;
            break;
        end
    end
end