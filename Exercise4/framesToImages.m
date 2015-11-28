
clc;
clear all;
close all;

% convert video to movie object
movieObj = VideoReader('movie01.avi');

% get number of frames
frames = movieObj.NumberOfFrames;

% display the first 3 frames
for i = 1 : 3
   
    img = read(movieObj, i);
    figure(i);
    imshow(img, []);
    title(sprintf('Frame %d', i));
    
end

