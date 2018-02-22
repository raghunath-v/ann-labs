function out = getNeighbourIndices(ind,size,offset)
    N=length(size);
    [c1{1:N}]=ndgrid(1:(2*offset+1));
    c2(1:N)={offset+1};
    offsets=sub2ind(size,c1{:}) - sub2ind(size,c2{:})
    offset_mat = ind+offsets
    offset_vect = reshape(offset_mat,[1 (2*offset+1)^2]);
    valid_indices =  find(offset_vect<=(size(1)*size(2))) & (offset_vect>=1 );
    out = offset_vect(valid_indices);
end