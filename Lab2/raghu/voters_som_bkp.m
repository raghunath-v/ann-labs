%% Section 4.3
%Analysing voter behaviour
clear all;clc;close all
load votes.dat
load mpsex.dat
load mpparty.dat
load mpdistrict.dat

num_voters = 349;
nodes = 100;
attributes = 31;
eta = 0.2;
props = reshape(votes,attributes,length(votes)/attributes)';
weights = double(rand(nodes,attributes)>0.5);
dist_list = zeros(nodes,num_voters);
MAX_EPOCHS = 20;
for epoch = 1:MAX_EPOCHS
    for voter = 1:num_voters
        diff = repmat(props(voter,:),[nodes,1])-weights;
        dist = sqrt(sum((diff).^2,2));
        [bestDist, bestInd] = min(dist);
        neib_size = (round(nodes/2)-round((nodes/2)*epoch/MAX_EPOCHS));
        idx=(max(1,bestInd-neib_size):min(nodes,bestInd+neib_size))';
        weights(idx,:) = weights(idx,:) + eta*(diff(idx,:));
        dist_list(:,voter) = dist;
        %plot(dist_list(:,[2,11,18]))
        %pause(0.1)
    end
end


order = [];
for voter = 1:num_voters
    testdiff = repmat(props(voter,:),[nodes,1])-weights;
    testdist = sqrt(sum((testdiff).^2,2));
    [bestDist,idx] = min(testdist);
    order = [order;idx];
end
[index, voter_order] = sort(order,'ascend');
voter_party=mpparty(voter_order,:);
voter_gender=mpsex(voter_order,:);
voter_district=mpdistrict(voter_order,:);
plot(voter_district)
