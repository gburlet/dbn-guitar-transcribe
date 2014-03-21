function [flabels] = getframelabels(midi_path, ftimes, binaryvec)
%GETFRAMELABELS given a midi file (midi_path) and a list of frame start
% and stop times (ftimes), return a cell array of midi numbers for each frame
% if binaryvec == false, otherwise returns a matrix of num_frames x |y|, where the columns
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

flabels = cell(num_frames,1);
for i = 1:num_frames
    % gather notes occuring in this frame from MIDI file
    nidx = (m(:,6) < ftimes(i,1) & m(:,7) > ftimes(i,2)) | ...
           (m(:,6) > ftimes(i,1) & m(:,6) < ftimes(i,2)) | ...
           (m(:,7) > ftimes(i,1) & m(:,7) < ftimes(i,2));
    flabels{i} = unique(m(nidx,4));
end

if binaryvec
    % convert to indicator matrix
    % get label bounds
    ymin = min(cellfun(@min, flabels));
    ymax = max(cellfun(@max, flabels));
end

end
