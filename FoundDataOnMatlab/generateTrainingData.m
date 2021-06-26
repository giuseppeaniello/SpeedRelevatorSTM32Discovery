% This script is used to establish connection with your Android phone
% (after the required MATLAB Mobile App) is installed. After establishing
% connection from MATLAB to your phone, this App will turn on the
% accelerometer and start collecting acceleration data from your phone. 
% Follow instructions displayed during execution of this code to
% successfully record training data.
%
% Copyright 2014 The MathWorks, Inc.

%connector on;

%mobileSensor = mobiledev();
%mobileSensor.SampleRate = 'High';

activity = ['walk          '; ...
            'run           '; ...
            'idle          '];
        
        
fileNames = ['walk.csv         '; ...
             'run.csv          '; ...
             'idle.csv         '];        
        
activityName = char(cellstr(activity));
windowLength = 2.558;
uniformSampleRate = 100;

%for i = 1:size(activityName, 1)
%    display(['Put your F3 discovery in pocket and register ', ...
%           deblank(activityName(i, :)), ' for at least 20 seconds.', ...
%           'Save the obtained data in a file with name: ' deblank(fileNames(i, :))]);
%end


for i = 1:size(activityName, 1)

    values = readmatrix(deblank(fileNames(i, :)));
    
   



    % mobileSensor.stop;
    % [a, t] = accellog(mobileSensor);
    a = values(:, [2:4]);
    t = values(:, 1);
    
    
    %plot(t, a);
    
    %%startTime = input('Use data cursor to select the start point in the plot, key in the x coordinate (time).');
    %stopTime = input('Use data cursor to select the stop point in the plot, key in the x coordinate (time).');

    %close all;
    %indexValidData = find(t > startTime & t < stopTime);
    %t = t(indexValidData) - t(indexValidData(1));
    %a = a(indexValidData, :);
    
    % Checking tWindow is monotonically increasing
    dt = diff(t);
    deleteIndex = find(dt <= 0);
    while (sum(deleteIndex) > 0)
        t(deleteIndex + 1) = [];
        a(deleteIndex + 1,:) = [];
        dt = diff(t);
        deleteIndex = find(dt <= 0);
    end
    
    plot(t, a);
    xlim([t(1), t(end)]);

    fileName = [deblank(activityName(i, :)), '.mat'];
    save(fileName, 'a', 't');
    
    feature{i} = extractTrainingFeature(fileName, windowLength, ...
                                            uniformSampleRate);
end

featureWalk = feature{1};
featureRun = feature{2};
featureIdle = feature{3};
save('userTrainingData.mat','featureWalk', 'featureRun', 'featureIdle');

display('Congratulations! You finished recording training data.');