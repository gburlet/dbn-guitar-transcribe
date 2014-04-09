mirverbose(0);
mirwaitbar(0);
mirparallel(0); % do parallel processing

% experiment parameters
project_path = '~/Documents/CMPUT656/dbn-guitar-transcribe/';
wav_path = [project_path, 'data/wav/'];
midi_path = [project_path, 'data/mid/'];

Fs = 11025;
w = 2048;               % window size (samples)
win_type = 'hamming';   % window type
h = 0.75;               % hop size (ratio wrt frame length)
freq_res = 2;           % frequency resolution of 1 Hz
train_percent = 0.8;
maxepoch = 80; 
numhid = 500; numpen = 500; numpen2 = 2000;

% load audio file and resample to lower sampling rate
a = miraudio('Design', 'Sampling', Fs);

% compute STFT, normalize wrt energy
cd(wav_path);
%spectrograms = mirspectrum(a, 'Window', win_type, 'MinRes', freq_res, 'Frame', w/Fs, h);
spectrograms = mirspectrum(a, 'Window', win_type, 'Frame', w/Fs, h);
afs = mireval(spectrograms, 'Folder');
cd(project_path);

song_names = get(afs{1}, 'Name');
num_songs = length(song_names);

% vector of frame times (2 x numFrames) for each song: start, stop (seconds) 
frame_times = get(afs{1}, 'FramePos');
song_frame_count = cellfun(@(f) size(f{1},2), frame_times);
total_frames = sum(song_frame_count);

frame_freqs = get(afs{1}, 'Data');
numdims = size(frame_freqs{1}{1}, 1);

% concatenate training data across songs 
X = zeros(total_frames, numdims);   % preallocate for speed
% dropped C tuning for lower bound MIDI number and standard tuning for upper bound MIDI number
% MIDI number 36 (C2: 65.406Hz) -- MIDI number 86 (D6: 1174.7Hz)
y = sparse(total_frames, 51);    % preallocate for speed
for sind = 1:num_songs
    midi_file = [midi_path, song_names{sind}(1:end-4), '.mid'];
    start_frame = sum(song_frame_count(1:sind-1)) + 1;
    end_frame = start_frame + song_frame_count(sind) - 1;  
    y(start_frame:end_frame,:) = getframelabels(midi_file, frame_times{sind}{1}');
    X(start_frame:end_frame,:) = frame_freqs{sind}{1}';
end

% remove silence and unannotated frames
silence = find(max(X,[],2) == 0);
unannotated = find(max(y,[],2) == 0);
remove_frames = vertcat(silence, unannotated);
X(remove_frames,:) = [];
y(remove_frames,:) = [];

% preprocess spectrogram: normalize energy between [0,1]
X = bsxfun(@rdivide, X, max(X,[],2));

% partition dataset into batches
[Xtrainb, ytrainb, Xtestb, ytestb] = makedatabatches(X, y, train_percent);
clearvars -except Xtrainb ytrainb Xtestb ytestb numhid numpen numpen2 maxepoch numdims;

save data.mat

fprintf('Pretraining Layer 1 with RBM: %d-%d \n', numdims, numhid);
[vishid, hidbiases, visbiases, batchposhidprobs] = rbm(Xtrainb, numhid, maxepoch, 1, true);
hidrecbiases = hidbiases;
%save mnistvhclassify vishid hidrecbiases visbiases;

fprintf('\nPretraining Layer 2 with RBM: %d-%d \n', numhid, numpen);
[hidpen, penrecbiases, hidgenbiases, batchposhidprobs] = rbm(batchposhidprobs, numpen, maxepoch, 1, true);
%save mnisthpclassify hidpen penrecbiases hidgenbiases;

fprintf('\nPretraining Layer 3 with RBM: %d-%d \n', numpen, numpen2);
[hidpen2, penrecbiases2, hidgenbiases2, batchposhidprobs] = rbm(batchposhidprobs, numpen2, maxepoch, 1, true);
%save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;

save pretrain.mat vishid hidrecbiases hidpen penrecbiases hidpen2 penrecbiases2 

fprintf(1,'\nTraining discriminative model by minimizing cross entropy error. \n');
backpropclassify(Xtrainb, ytrainb, Xtestb, ytestb, vishid, hidrecbiases, hidpen, penrecbiases, hidpen2, penrecbiases2);
