function [flabels] = getframelabels(midi_path, ftimes)
%GETFRAMELABELS given a midi file and a list of frame start
% and stop times, return a cell array of midi numbers for each frame.

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
    %{
    nidx = (m(:,6) > ftimes(i,1) & m(:,7) < ftimes(i,2)) | ...
           (m(:,6) < ftimes(i,1) & m(:,7) > ftimes(i,1) & m(:,7) < ftimes(i,2)) | ...
           (m(:,6) > ftimes(i,1) & m(:,6) < ftimes(i,2) & m(:,7) > ftimes(i,2)) | ...
           (m(:,6) < ftimes(i,1) & m(:,7) > ftimes(i,2));
    %}
    nidx = (m(:,6) < ftimes(i,1) & m(:,7) > ftimes(i,2)) | ...
           (m(:,6) > ftimes(i,1) & m(:,6) < ftimes(i,2)) | ...
           (m(:,7) > ftimes(i,1) & m(:,7) < ftimes(i,2));
    flabels{i} = unique(m(nidx,4));
end

end
