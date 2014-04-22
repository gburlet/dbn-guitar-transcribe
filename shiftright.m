function [Xshift] = shiftright(X, shift)
%SHIFTRIGHT Shifts a matrix to the right (non-circular)
    Xshift = circshift(X, [0, shift]);
    Xshift(:, 1:min(shift,size(X,2))) = 0;
end

