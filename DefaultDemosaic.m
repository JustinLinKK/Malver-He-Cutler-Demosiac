% Read the input image
img = imread('test2.png');
%img = rgb2gray(img);   % convert to grayscale

% Demosaic the image 
demosaic_img = demosaic(img, 'rggb');

% Read the ground truth image
%gt_img = imread('GroundTruthSample3.png');


% Calculate the root-mean-square error (RMSE) between the demosaiced image and the ground truth image
%rmse = sqrt(mean((gt_img(:) - demosaic_img(:)).^2));

% Display the demosaiced image and save it as a PNG file named 'DefaultDemosaiced_Sample1.png'
figure, imshow(demosaic_img), title('Default Demosaiced Image');
imwrite(demosaic_img, 'DefaultDemosaiced_test2.png');

% Dispaly the RMSE value
disp(rmse);
