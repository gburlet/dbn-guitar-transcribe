function [recall, cerr] = calcerror(Xb, yb, w1, w2, w3, w_class)
%CALCERROR calculates the prediction error of the model on the batch dataset Xb
%   in relation to the batch targets yb.

%initialize counters
err = 0; 
err_cr = 0;
num_correct = 0;

% cache dimensions
numbatches = length(Xb);
num_labels = size(yb{1},2);

% allocate sparse matrix for note predictions
yhatb = cellfun(@(y) sparse(size(y,1), size(y,2)), yb, 'UniformOutput', false);
for batch = 1:numbatches
    [N, numdims] = size(Xb{batch});
    data = [Xb{batch}, ones(N,1)];
    target = yb{batch};
    w1probs = [1./(1 + exp(-data*w1)), ones(N,1)];
    w2probs = [1./(1 + exp(-w1probs*w2)), ones(N,1)];
    w3probs = [1./(1 + exp(-w2probs*w3)), ones(N,1)];
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
    yhatb{batch}(nidx(:,1:polyphony)) = 1;

    % calculate number of correct note predictions
    num_correct = num_correct + sum(sum(yhatb{batch} & yb{batch}));
    %err_cr = err_cr - sum(sum(target(:,1:end).*log(targetout)));
end

num_notes = sum(cellfun(@(y) sum(nonzeros(y)), yb));
%fprintf('num correct: %d, num notes: %d \n', num_correct, num_notes);
recall = num_correct / num_notes;
cerr = err_cr / numbatches;

end
