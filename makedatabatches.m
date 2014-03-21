function [Xtrainb, ytrainb, Xtestb, ytestb] = makedatabatches(Xtrain, ytrain, Xtest, ytest)
%MAKEDATABATCHES partition the training and testing data into batches for dbn training

[Ntrain, F] = size(Xtrain);
[Ntest, ycard] = size(ytest);

batch_size = 100;
[train_batch_size, last_train_batch] = deal(floor(Ntrain/batch_size), mod(Ntrain, batch_size));
[test_batch_size, last_test_batch] = deal(floor(Ntest/batch_size), mod(Ntest, batch_size));
Xtrainb = mat2cell(Xtrain, [batch_size*ones(1,train_batch_size), last_train_batch]);
ytrainb = mat2cell(ytrain, [batch_size*ones(1,train_batch_size), last_train_batch]);
Xtestb = mat2cell(Xtest, [batch_size*ones(1,test_batch_size), last_test_batch]);
ytestb = mat2cell(ytest, [batch_size*ones(1,test_batch_size), last_test_batch]);

%{
Earlier implementation using truncated matrices
[num_train_batches, train_batch_size] = deal(floor(sqrt(Ntrain)));
[num_test_batches, test_batch_size] = deal(floor(sqrt(Ntest)));
assert(num_train_batches > 0 && num_test_batches > 0, 'Not enough training/testing data for one batch');
Xtrainb = permute(reshape(Xtrain(1:num_train_batches^2,:)', [F, num_train_batches, train_batch_size]), [2,1,3]);
%}

end
