%% Section 4.1
%Classifying animals
clear all;clc;close all
load animals.dat

nodes = 100;
attributes = 84;
eta = 0.2;
props = reshape(animals,attributes,length(animals)/attributes)';
weights = double(rand(nodes,attributes)>0.5);
dist_list = zeros(100,32);
for epoch = 1:20
    for animal = 1:32
        diff = repmat(props(animal,:),[nodes,1])-weights;
        dist = sqrt(sum((diff).^2,2));
        [bestDist, bestInd] = min(dist);
        neib_size = (50-round(2.5*epoch));
        idx=[max(1,bestInd-neib_size):min(100,bestInd+neib_size)]';
        weights(idx,:) = weights(idx,:) + eta*(diff(idx,:));
        dist_list(:,animal) = dist;
        %plot(dist_list(:,[2,11,18]))
        %pause(0.1)
    end
end

animalNames = textread('animalnames.txt','%s', 'delimiter','\n','whitespace','');
animalNames = regexprep(animalNames,'[''\t]','');
order = [];
for animal = 1:32
    testdiff = repmat(props(animal,:),[nodes,1])-weights;
    testdist = sqrt(sum((testdiff).^2,2));
    [bestDist,idx] = min(testdist);
    order = [order;idx];
end
[index, anim_order] = sort(order,'descend');
animalNames(anim_order)

%% Section 4.2
%Travelling Salesman Problem
clear all;clc;close all
load cities.dat

nodes = 10;
attributes = 2;
eta = 0.2;
props=cities;
weights = double(rand(nodes,attributes)>0.5);
dist_list = zeros(nodes,10);
MAX_EPOCHS = 20;
for epoch = 1:MAX_EPOCHS
    for city = 1:10
        diff = repmat(props(city,:),[nodes,1])-weights;
        dist = sqrt(sum((diff).^2,2));
        [bestDist, bestInd] = min(dist);
        neib_size = (3-round(((3/MAX_EPOCHS)*epoch)^2));
        idx=(bestInd-neib_size:bestInd+neib_size)';
        idx = mod(10+idx-1,10)+1;
        weights(idx,:) = weights(idx,:) + eta*(diff(idx,:));
        dist_list(:,city) = dist;
    end
end


order = [];
for city = 1:10
    testdiff = repmat(props(city,:),[nodes,1])-weights;
    testdist = sqrt(sum((testdiff).^2,2));
    [bestDist,idx] = min(testdist);
    order = [order;idx];
end
[index, city_order] = sort(order,'ascend');
new_cities=cities(city_order,:);

hold on
scatter(cities(:,1),cities(:,2),100,'fill','blue')
line(new_cities(:,1),new_cities(:,2))
scatter(weights(:,1),weights(:,2),'d','red','fill')

%% Section 4.3
%Analysing voter behaviour
clear all;clc;close all
load votes.dat
load mpsex.dat
load mpparty.dat
load mpdistrict.dat

num_voters = 349;
nodes = 100;
map = [10 10];
attributes = 31;
eta = 0.2;
props = reshape(votes,attributes,length(votes)/attributes)';
%weights = double(rand(nodes,attributes)>0.5);
weights = double(rand(nodes,attributes)>0.5);
dist_list = zeros(nodes,num_voters);
MAX_EPOCHS = 20;
for epoch = 1:MAX_EPOCHS
    for voter = 1:num_voters
        diff = repmat(props(voter,:),[nodes,1])-weights;
        dist = sqrt(sum((diff).^2,2));
        [bestDist, bestInd] = min(dist);
        neib_size = floor((round(nodes/2)-round((nodes/2)*epoch/MAX_EPOCHS))/10);
        %idx=(max(1,bestInd-neib_size):min(nodes,bestInd+neib_size))';
        idx = getNeighbourIndices(bestInd,map,neib_size);
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
%[index, voter_order] = sort(order,'ascend');
voter_party=mpparty(voter_order,:);
voter_gender=mpsex(voter_order,:);
voter_district=mpdistrict(voter_order,:);
plot(voter_party)
r
