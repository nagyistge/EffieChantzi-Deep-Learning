           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %% Chantzi Efthymia - Deep Learning - Exercises 5,6  %%
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function creates an object array for storing the video objects,  %
% which are parsed from the used-defined file directory 'moviesFolder'. %
% This is implemented by the contribution of the ObjectArray class,     %
% which creates an object array that is the same size as the input      %
% array. This function requires that all the video files are in .avi    %
% format and stored in a separate folder of the current working         %
% parent directory.                                                     %
%                                                                       %
%                                                                       %
% %%%% Inputs %%%%                                                      %
% movieFolder: name of the file directory that contains all the .avi    %
% movie files                                                           %
%                                                                       %
%                                                                       %
% %%%% Outputs %%%%                                                     %
% movieObjArray: object array of movie objects from the movie dataset   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function movieObjArray = createMovieObjects(movieFolder)

% current parent working directory
basicDir = pwd;

% file directory containing the whole movie dataset
dirMovies = fullfile(basicDir, movieFolder);
cd(dirMovies);

% load all video .avi files 
videos = dir('*.avi');

% total number of movies in the respective directory
numOfVideos = numel(videos);

% initialize object array that stores the video reader objects from movies
movieObjArray = ObjectArray(numOfVideos);

for i = 1 : numOfVideos
   
    movieObjArray(i) = VideoReader(videos(i).name);
    
end

% back to parent working directory
cd(basicDir);

end