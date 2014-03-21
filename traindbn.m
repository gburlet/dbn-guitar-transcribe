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

[y, ymin, ymax] = getframelabels('data/delilah.mid', ftimes);

X = get(afs, 'Data');
X = X{1}{1}';

% partition dataset into batches
[Xtrainb, ytrainb, Xtestb, ytestb] = makedatabatches(X, y, train_percent);
clear X y a afs ftimes;
