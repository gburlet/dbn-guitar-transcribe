mirverbose(0);
mirwaitbar(0);

% experiment parameters
Fs = 11025;
w = 2048;               % window size (samples)
win_type = 'hamming';   % window type
h = 0.75;               % hop size (ratio wrt frame length)
train_percent = 0.8;

% load audio file and resample to lower sampling rate
a = miraudio('data/delilah.wav', 'Sampling', Fs);

% compute STFT, normalize wrt energy
afs = mirspectrum(a, 'Normal', 'Window', win_type, 'Frame', w/Fs, h);

% frame time (2 x numFrames): start, stop (seconds) 
ftimes = get(afs, 'FramePos');
ftimes = ftimes{1}{1}';

flabels = getframelabels('data/delilah.mid', ftimes, true);

data = get(afs, 'Data');
data = data{1}{1}';

[N, F] = size(data);

% shuffle data, create train/test dataset
shuffle = randperm(N);
Ntrain = ceil(train_percent * N);
Ntest = N - Ntrain;
Xtrain = data(shuffle(1:Ntrain),:);
Xtest = data(shuffle(Ntrain+1:end),:)
ytrain = 
