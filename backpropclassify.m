% NOTE: this code is a modification of the deep learning code provided by Ruslan and Geoff
% for the purpose of guitar transcription instead of MNIST digit classification.

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

function [] = backpropclassify(Xtrainb, ytrainb, Xtestb, ytestb, vishid, hidrecbiases, hidpen, penrecbiases, hidpen2, penrecbiases2)
% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

% initialize experiment variables
maxepoch=200;
test_recall = zeros(1, maxepoch);
train_recall = zeros(1, maxepoch);

% preinitialize weights of the discriminative model
num_labels = size(ytestb{1},2);
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w_class = 0.1*randn(size(w3,2)+1,num_labels);

% cache layer dimensions
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w_class,1)-1;
l5=num_labels;

for epoch = 1:maxepoch
    % Calculate training and testing recall
    [train_recall(epoch), train_cerr] = calcerror(Xtrainb, ytrainb, w1, w2, w3, w_class);
    [test_recall(epoch), test_cerr] = calcerror(Xtestb, ytestb, w1, w2, w3, w_class);
    fprintf(1,'Before epoch %d; Training recall: %.2f%%. Testing recall: %.2f%% \n',...
            epoch, train_recall(epoch)*100, test_recall(epoch)*100);

    %{
    tt=0;
    for batch = 1:numbatches/10
        fprintf(1,'epoch %d batch %d\r',epoch,batch);

        %%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1; 
        data=[];
        targets=[]; 
        for kk=1:10
        data=[data 
            batchdata(:,:,(tt-1)*10+kk)]; 
        targets=[targets
            batchtargets(:,:,(tt-1)*10+kk)];
        end 

        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;

        if epoch<6  % First update top-level weights holding other weights fixed. 
            N = size(data,1);
            XX = [data ones(N,1)];
            w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
            w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
            w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];

            VV = [w_class(:)']';
            Dim = [l4; l5];
            [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w3probs,targets);
            w_class = reshape(X,l4+1,l5);
        else
            VV = [w1(:)' w2(:)' w3(:)' w_class(:)']';
            Dim = [l1; l2; l3; l4; l5];
            [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets);

            w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
            xxx = (l1+1)*l2;
            w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
            xxx = xxx+(l2+1)*l3;
            w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
            xxx = xxx+(l3+1)*l4;
            w_class = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
        end
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end

    save classify_weights w1 w2 w3 w_class
    %save classify_error test_err test_crerr train_err train_crerr;
    %}
end

end
