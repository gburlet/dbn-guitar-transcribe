function [yshift] = shiftoctaves(y)
%SHIFTOCTAVES takes binary indicator matrix and wherever a one is present,
%   a one is added at integer multiple of 12 columns.
    N = size(y,2);
    num_octaves = uint8(floor(N/12));
    yshift = y;
    for i = 1:num_octaves
        yshift = yshift + shiftright(y, i*12);
    end
    yshift(yshift > 1) = 1;
end