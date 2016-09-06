function [mean] = edge()
x = uigetfile('F:\Project\projects\testings\videos\working\*.*','All Files (*.*)');
addpath('F:\Project\projects\testings\videos');
addpath('F:\Project\projects\testings\videos\working');
addpath('C:\Program Files\MATLAB\MATLAB Production Server\R2015a\bin\videos');

%--------------------Detect face using viola jones ---------------------------------%
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(x);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);

figure;
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Human Face');
imshow(videoOut), title('Detected face');
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for i=1:4 
    p(i)=bbox(i);
end

%  Display the image and store its handle:
figure;
h_im = image(videoFrame);

%  Create an rectangle defining a ROI:

e = imrect(gca,[p(1) p(2) p(3) p(4)]);

%  Create a mask from the rectangle:

BW = createMask(e,h_im);

%  (For color images only) Augment the mask to three channels:

BW(:,:,2) = BW;

BW(:,:,3) = BW(:,:,1);

%  Use logical indexing to set area outside of ROI to zero:

ROI = videoFrame;

ROI(BW == 0) = 0;

%  Display extracted portion:

figure, imshow(ROI),title('ROI');

%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ROI= rgb2gray(ROI);
fx = [-1 0 1;-1 0 1;-1 0 1];
Ix = filter2(fx,ROI);
fy = [1 1 1;0 0 0;-1 -1 -1];
Iy = filter2(fy,ROI); 

Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;
clear Ix;
clear Iy;

%applying gaussian filter on the computed value
h= fspecial('gaussian',[7 7],2); 
Ix2 = filter2(h,Ix2);
Iy2 = filter2(h,Iy2);
Ixy = filter2(h,Ixy);
height = size(ROI,1);
width = size(ROI,2);
result = zeros(height,width); 
R = zeros(height,width);

Rmax = 0; 
for i = 1:height
for j = 1:width
M = [Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)]; 
R(i,j) = det(M)-0.01*(trace(M))^2;
if R(i,j) > Rmax
Rmax = R(i,j);
end;
end;
end;
cnt = 0;
for i = 2:height-1
for j = 2:width-1
if R(i,j) > 0.1*Rmax && R(i,j) > R(i-1,j-1) && R(i,j) > R(i-1,j) && R(i,j) > R(i-1,j+1) && R(i,j) > R(i,j-1) && R(i,j) > R(i,j+1) && R(i,j) > R(i+1,j-1) && R(i,j) > R(i+1,j) && R(i,j) > R(i+1,j+1)
result(i,j) = 1;
cnt = cnt+1;
end;
end;
end;
[posc, posr] = find(result == 1);
cnt ;
imshow(ROI);
hold on;
plot(posr,posc,'r+');

%detected points corrdinates
for i=1:size(posc)
    row(i)=posr(i);
    col(i)=posc(i);
end
%-------------------------------------------------------------------------------------------------------------------------------%

%Finding centroid by calculating coordinates of image and intersection of
%tr and tc gives centroid
minr = posr(1);
minc = posc(1);
maxr = 0;
maxc = 0;
for i=1:size(posc)
    if posc(i) > maxc
        maxc = posc(i);
    end
    if posc(i) < minc
        minc = posc(i);
    end
    if posr(i) > maxr
        maxr = posr(i);
    end
    if posr(i) < minr
        minr = posr(i);
    end
end
tr = (maxr-minr)/2;
tc = (maxc-minc)/2;

figure;
imshow(videoFrame);
hold on;
plot(minr+tr,minc+tc,'r+');

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Human Face');
imshow(videoOut), title('Detected face');
plot(minr+tr,minc+tc,'rx');
plot(posr,posc,'rx');



%---------------------------------------------------------------------------------------------------------------------------%