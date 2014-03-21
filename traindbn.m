mirverbose(0);
mirwaitbar(0);

% experiment parameters
Fs = 11025;
w = 2048;               % window size (samples)
win_type = 'hamming';   % window type
h = 0.75;               % hop size (ratio wrt frame length)
train_percent = 0.8;
maxepoch=50; 
numhid=500; numpen=500; numpen2=2000; 

% load audio file and resample to lower sampling rate
a = miraudio('data/delilah.wav', 'Sampling', Fs);

% compute STFT, normalize wrt energy
afs = mirspectrum(a, 'Normal', 'Window', win_type, 'Frame', w/Fs, h);

% frame time (2 x numFrames): start, stop (seconds) 
ftimes = get(afs, 'FramePos');
ftimes = ftimes{1}{1}';

[y, ymin, ymax] = getframelabels('data/delilah.mid', ftimes);

X = get(afs, 'Data');
X = X{1}{1}';
numdims = size(X,2);

% partition dataset into batches
[Xtrainb, ytrainb, Xtestb, ytestb] = makedatabatches(X, y, train_percent);
clear X y a afs ftimes;

fprintf('Pretraining Layer 1 with RBM: %d-%d \n', numdims, numhid);
[vishid, hidbiases, visbiases] = rbm(Xtrainb, numhid, maxepoch, 1);
hidrecbiases = hidbiases;
save mnistvhclassify vishid hidrecbiases visbiases;

%{
fprintf('\nPretraining Layer 2 with RBM: %d-%d \n', numhid, numpen);
batchdata = batchposhidprobs;
numhid = numpen;
restart = 1;
rbm;
hidpen = vishid; penrecbiases = hidbiases; hidgenbiases = visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;

fprintf('\nPretraining Layer 3 with RBM: %d-%d \n', numpen, numpen2);
batchdata = batchposhidprobs;
numhid = numpen2;
restart = 1;
rbm;
hidpen2 = vishid; penrecbiases2 = hidbiases; hidgenbiases2 = visbiases;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;
%}
