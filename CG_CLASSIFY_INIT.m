% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


function [f, df] = CG_CLASSIFY_INIT(VV,Dim,w3probs,target);
l1 = Dim(1);
l2 = Dim(2);
num_labels = Dim(end);

N = size(w3probs,1);
% Do decomversion.
w_class = reshape(VV,l1+1,l2);
w3probs = [w3probs  ones(N,1)];  

% sigmoid output with matching cross-entropy error function
targetout = 1./(1+exp(-w3probs*w_class));
f = -sum(sum(target.*log(targetout) + (1-target).*log(1-targetout)));

%  targetout = exp(w3probs*w_class);
%  targetout = targetout./repmat(sum(targetout,2),1,num_labels);
%  f = -sum(sum( target(:,1:end).*log(targetout))) ;

% derivative is simply yhat - y, since we are using the matching loss
IO = targetout - target;
Ix_class=IO; 
dw_class =  w3probs'*Ix_class; 

df = [dw_class(:)']'; 

