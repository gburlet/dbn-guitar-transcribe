function [recall, recall_lowpoly, recall_highpoly, cerr] = calcerror(Xb, yb, w1, w2, w3, w_class, nooctaveerr)
%CALCERROR calculates the prediction error of the model on the batch dataset Xb
%   in relation to the batch targets yb.
%   nooctaveerr is a boolean which ignores octave errors if enabled

%initialize counters
err = 0; 
err_cr = 0;
num_correct = 0;
num_correct_lowpoly = 0;
num_correct_highpoly = 0;

% cache dimensions
numbatches = length(Xb);
num_labels = size(yb{1},2);
num_lowpoly_notes = 0;
num_highpoly_notes = 0;

num_lowpoly_frames = 0;
num_highpoly_frames = 0;

% allocate sparse matrix for note predictions
yhatb = cellfun(@(y) sparse(size(y,1), size(y,2)), yb, 'UniformOutput', false);
for batch = 1:numbatches
    [N, numdims] = size(Xb{batch});
    data = [Xb{batch}, ones(N,1)];
    target = yb{batch};
    w1probs = [1./(1 + exp(-data*w1)), ones(N,1)];
    w2probs = [1./(1 + exp(-w1probs*w2)), ones(N,1)];
    w3probs = [1./(1 + exp(-w2probs*w3)), ones(N,1)];

    % sigmoid output unit activation
    targetout = 1./(1+exp(-w3probs*w_class));

    % softmax output unit activation
    %targetout = exp(w3probs*w_class);
    %targetout = targetout./repmat(sum(targetout,2),1,num_labels);

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
    for fidx = 1:N
        yhatb{batch}(fidx, nidx(fidx, 1:polyphony(fidx))) = 1;
    end

    lowpoly_target = target(polyphony <= 3,:);
    lowpoly_targethat = yhatb{batch}(polyphony <= 3,:);
    highpoly_target = target(polyphony > 3,:);
    highpoly_targethat = yhatb{batch}(polyphony > 3,:);
 
    % calculate number of correct note predictions
    if nooctaveerr
        target_oct = shiftoctaves(target);
        target_oct_lowpoly = shiftoctaves(lowpoly_target);
        target_oct_highpoly = shiftoctaves(highpoly_target);
        num_correct = num_correct + sum(sum(yhatb{batch} & target_oct));
        num_correct_lowpoly = num_correct_lowpoly + sum(sum(lowpoly_targethat & target_oct_lowpoly));
        num_correct_highpoly = num_correct_highpoly + sum(sum(highpoly_targethat & target_oct_highpoly));
    else
        num_correct = num_correct + sum(sum(yhatb{batch} & target));
        num_correct_lowpoly = num_correct_lowpoly + sum(sum(lowpoly_targethat & lowpoly_target));
        num_correct_highpoly = num_correct_highpoly + sum(sum(highpoly_targethat & highpoly_target));
    end
    
    num_lowpoly_notes = num_lowpoly_notes + sum(sum(lowpoly_target));
    num_highpoly_notes = num_highpoly_notes + sum(sum(highpoly_target));
    
    num_lowpoly_frames = num_lowpoly_frames + sum(polyphony <= 3);
    num_highpoly_frames = num_highpoly_frames + sum(polyphony > 3);
    %err_cr = err_cr - sum(sum(target(:,1:end).*log(targetout)));
end

num_notes = sum(cellfun(@(y) sum(nonzeros(y)), yb));

recall = num_correct / num_notes;
recall_lowpoly = num_correct_lowpoly / num_lowpoly_notes;
recall_highpoly = num_correct_highpoly / num_highpoly_notes;
%num_lowpoly_frames
%num_highpoly_frames

cerr = err_cr / numbatches;

end
