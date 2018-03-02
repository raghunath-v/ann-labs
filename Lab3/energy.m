function out = energy(w,x)
    N = size(x,2); %Number of patterns x Number of units
    E=zeros(size(x,1),1);
    for i=1:N
        for j=1:N
            E = E - (w(i,j)*x(:,i).*x(:,j));
        end
    end
    out = E;
end