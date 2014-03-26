function [y] = getframelabels(midi_path, ftimes)
%GETFRAMELABELS given a midi file (midi_path) and a list of frame start
% and stop times (ftimes), returns a matrix of num_frames x |y|, where the columns
% are an indicator vector with ones to indicate the midi note number.
num_frames = size(ftimes,1);

m = readmidi(midi_path);
% change duration to note offset time
m(:,7) = m(:,6) + m(:,7);
% guitar pro adds 0.025 seconds of silence to beginning of audio files
m(:,6:7) = m(:,6:7) + 0.025;
% m(:,4) = MIDI pitch
% m(:,6) = note onset (sec)
% m(:,7) = note offset (sec)

y = sparse(num_frames, 128);  % midi numbers 0-127
for i = 1:num_frames
    % gather notes occuring in this frame from MIDI file
    nidx = (m(:,6) < ftimes(i,1) & m(:,7) > ftimes(i,2)) | ...
           (m(:,6) > ftimes(i,1) & m(:,6) < ftimes(i,2)) | ...
           (m(:,7) > ftimes(i,1) & m(:,7) < ftimes(i,2));
    y(i, unique(m(nidx,4))) = 1;
end

end
