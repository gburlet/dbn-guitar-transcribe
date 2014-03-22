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

maxepoch=200;
fprintf(1,'\nTraining discriminative model by minimizing cross entropy error. \n');

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_labels = size(ytestb{1},2);
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w_class = 0.1*randn(size(w3,2)+1,num_labels);

%%%%%%%%%% END OF PREINITIALIZATION OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w_class,1)-1;
l5=num_labels; 
test_err=[];
train_err=[];

train_numbatches = length(Xtrainb);
%train_numsamples = sum(cellfun(@(x) size(x,1), Xtrainb));
test_numbatches = length(Xtestb);
%test_numsamples = sum(cellfun(@(x) size(x,1), Xtestb));

for epoch = 1:maxepoch
%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0; 
    err_cr=0;
    counter=0;
    % allocate sparse matrix for training note predictions
    yhattrainb = cellfun(@(x) sparse(size(x,1), size(x,2)), ytrainb, 'UniformOutput', false);
    for batch = 1:train_numbatches
        [N, numdims] = size(Xtrainb{batch});
        data = [Xtrainb{batch}, ones(N,1)];
        target = ytrainb{batch};
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs, ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs, ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs, ones(N,1)];
        targetout = exp(w3probs*w_class);
        targetout = targetout./repmat(sum(targetout, 2), 1, num_labels);

        % TODO: polyphony estimation
        % possibly train neural network here for polyphony estimation:
        % X = probability output of dbn on class labels [numsamples x numlabels]
        % y = {1, ..., 6} for polyphony
        %{
        [probs, nidx] = sort(targetout, 2, 'descend');
        probs(1, 1:8)
        nidx(1, 1:8)
        %polyphony = sum(target, 2);
        %polyphony(1:2)
        [i,j] = find(target(1,:));
        notes = accumarray(i', j', [], @(x) {x'});
        celldisp(notes)
        %}

        % for now, assume polyphony is known
        polyphony = sum(target, 2);
        [~, nidx] = sort(targetout, 2, 'descend');
        yhattrainb{batch}(nidx(:,1:polyphony)) = 1;

        counter = counter + sum(sum(yhattrainb{batch} & ytrainb{batch}));
        %err_cr = err_cr - sum(sum(target(:,1:end).*log(targetout)));
    end
    num_ytrainnotes = sum(cellfun(@(x) sum(nonzeros(x)), ytrainb));
    train_err(epoch) = num_ytrainnotes - counter;
    %train_crerr(epoch) = err_cr/train_numbatches;
%%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    yhattestb = cellfun(@(x) sparse(size(x,1), size(x,2)), ytestb, 'UniformOutput', false);
    for batch = 1:test_numbatches
        [N, numdims] = size(Xtestb{batch});
        data = [Xtestb{batch}, ones(N,1)];
        target = ytestb{batch};
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs, ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs, ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs, ones(N,1)];
        targetout = exp(w3probs*w_class);
        targetout = targetout./repmat(sum(targetout, 2), 1, num_labels);

         % for now, assume polyphony is known
        polyphony = sum(target, 2);
        [~, nidx] = sort(targetout, 2, 'descend');
        yhattestb{batch}(nidx(:,1:polyphony)) = 1;

        counter = counter + sum(sum(yhattestb{batch} & ytestb{batch}));
        %err_cr = err_cr - sum(sum(target(:,1:end).*log(targetout)));       
    end
    num_ytestnotes = sum(cellfun(@(x) sum(nonzeros(x)), ytestb));
    test_err(epoch) = num_ytestnotes - counter;
    %test_crerr(epoch) = err_cr/test_numbatches;
    fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
            epoch, train_err(epoch), num_ytrainnotes, test_err(epoch), num_ytestnotes);
%%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
