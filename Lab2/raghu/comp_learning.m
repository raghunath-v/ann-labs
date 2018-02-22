%For trial use this simulate_comp_learning
function out = comp_learning(W, data)
    PLOT = true; %Turn this on to see the graph
    num_winners = 7;
    move_rate = 0.2;
    num_iter = 5;
    data = data(:,randperm(size(data,2)));
    if(PLOT == true)
        figure;
        hold on
    end
    for n = 1:num_iter
        for i = 1:size(data,2)
           dist = sqrt(sum((repmat(data(:,i),[1,size(W,2)])-W).^2,1));
           %min_vals = zeros(num_winners,1);
           bestDist = 1;
           for k=1:num_winners
              [winnerDist,idx] = min(dist);
              if (k==1)
                  bestDist = winnerDist;
                  bestdiff = data(:,i)-W(:,idx);
              end
              %min_vals(k) = [winnerDist,idx];
              %dist(idx)
              diff = data(:,i)-W(:,idx);
              dW = (move_rate*diff/(k^2));
              W(:,idx) = W(:,idx) + dW;
              dist(:,idx) = inf(size(dist(:,idx)));
           end
           if(PLOT == true)
               hold on
               scatter(data(1,:)', data(2,:)','green');
               scat=scatter(W(1,:)',W(2,:)','d','filled','red');
               pause(0.1)
               delete(scat)
           end
        end
    end
    out = W;
end