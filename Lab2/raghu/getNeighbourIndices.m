function out = getNeighbourIndices(ind,size,offset)
%     N=length(size);
%     [c1{1:N}]=ndgrid(1:(2*offset+1));
%     c2(1:N)={offset+1};
%     offsets=sub2ind(size,c1{:}) - sub2ind(size,c2{:})
%     offset_mat = ind+offsets
%     offset_vect = reshape(offset_mat,[1 (2*offset+1)^2]);
%     valid_indices =  find(offset_vect<=(size(1)*size(2))) & (offset_vect>=1 );
%     out = offset_vect(valid_indices);
    index_list = [];
    max_dist = offset * sqrt(2);
    for i=1: size(1)*size(2)
        [x1,y1]=ind2sub([10,10], 13);
        [x2,y2]=ind2sub([10,10], 13);
        dist=sqrt((x1-x2)^2+(y1-y2)^2);
        if (dist<=max_dist) 
            index_list=[index_list;i];
        end
    end
    out = index_list;
end