function [indicator]=next(x, valrc)
  %x = uigetfile('F:\Project\projects\testings\videos\working\*.*','All Files (*.*)');
   y = which(x);
videoFileReader = vision.VideoFileReader(y);
faceDetector = vision.CascadeObjectDetector();
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));
% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
   valrc = cornerPoints(valrc);
 d = [points;valrc];
points = d;
%points = bbox;
% Display the detected points.
figure, imshow(videoFrame), hold on, title('Detected features');
plot(points);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);


%initialize video player to display the results
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);


% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
cnt=0;
frames=0;
while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    frames=frames+1;
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
   
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        cnt=cnt+1;
        i=10000000;
        %delay
        while(i>=0)
            i=i-1;
        end
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);

        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);

        % Display tracked points
        %videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
         %   'Color', 'white');

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
end
release(videoFileReader);
release(videoPlayer);
release(pointTracker);
assignin('base','cnt',cnt);
assignin('base','frames',frames);
assignin('base','isFound',isFound);