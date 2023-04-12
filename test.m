% Load the coefficient from files call coefficients.mat

load('coefficients.mat');

% Load the test bayer image call test_image.png, the variable should be stored as a 2D matrix
 
test_image =  imread('test2.png');

% Convert the test image to double

test_image = double(test_image);

% Create a blank index Bayer matrix with the same size as the test image
index_bayer = zeros(size(test_image, 1), size(test_image, 2));

% Assign color indices to each pixel based on the Bayer pattern
for row = 1:size(index_bayer, 1)
    for col = 1:size(index_bayer, 2)
        if mod(row, 2) == 1 % odd row
            if mod(col, 2) == 1 % odd column
                index_bayer(row, col) = 1; % red
            else % even column
                index_bayer(row, col) = 3; % green type 1
            end
        else % even row
            if mod(col, 2) == 1 % odd column
                index_bayer(row, col) = 2; % green type 2
            else % even column
                index_bayer(row, col) = 4; % blue
            end
        end
    end
end

% Create a blank RGB image
rgb_image = zeros(size(index_bayer, 1), size(index_bayer, 2), 3);

% Assign the color values to each pixel based on the pixel's type and the coefficients

for i = 1:size(test_image, 1)
    for j = 1:size(test_image, 2)
        % Check if the current pixel is not on the edge of the image
        if i > 2 && i < size(test_image, 1) - 2 && j > 2 && j < size(test_image, 2) - 2
            % Create a 5x5 matrix around the current pixel, record the values of this patch
            patch = test_image(i - 2:i + 2, j - 2:j + 2);
            % Identify the type of pixel in the center of the patch
        else % If it is on the edge of the image, fill the blank positions with medium gray
            patch = zeros(5, 5);
            patch(3, 3) = 128;
        end
        % First, fill the position that not out of the bayer image
        for k = 1:5
            for l = 1:5
                if i - 3 + k > 0 && i - 3 + k <= size(test_image, 1) && j - 3 + l > 0 && j - 3 + l <= size(test_image, 2)
                    patch(k, l) = test_image(i - 3 + k, j - 3 + l);
                end
            end
        end
        % Then, fill the position that out of the bayer image using medium gray
        for k = 1:5
            for l = 1:5
                if patch(k, l) == 0
                    patch(k, l) = 128;
                end
            end
        end
        center_pixel = index_bayer(i, j);
        % Determine which dataset to use based on the type of pixel in the center of the patch
        % For the type one pixel, use the dataset_one and dataset_two
        if center_pixel == 1
            % Transfer the patch to a 1x25 vector
            patch_vector = reshape(patch, 1, 25);
            % Convert the patch vector to a double type
            patch_vector = double(patch_vector);
            % Predict the value of the green for the pixel in the center of the patch
            green = patch_vector * coeff_one;
            % Select the first row element as the green predict value
            green_predict = green(1);
            % Predict the value of the blue for the pixel in the center of the patch
            blue = patch_vector * coeff_two;
            % Select the first row element as the blue predict value
            blue_predict = blue(1);
            % Extract the red value from the bayer image
            red = test_image(i, j);
            % Assign the color values to the pixel in the RGB image
            rgb_image(i, j, 1) = red;
            rgb_image(i, j, 2) = green_predict;
            rgb_image(i, j, 3) = blue_predict;

        % For the type two pixel, use the dataset_three and dataset_four
        elseif center_pixel == 2
            % Transfer the patch to a 1x25 vector
            patch_vector = reshape(patch, 1, 25);
            % Convert the patch vector to a double type
            patch_vector = double(patch_vector);
            % Predict the value of the red for the pixel in the center of the patch
            red = patch_vector * coeff_three;
            % Select the first row element as the red predict value
            red_predict = red(1);
            % Predict the value of the blue for the pixel in the center of the patch
            blue = patch_vector * coeff_four;
            % Select the first row element as the blue predict value
            blue_predict = blue(1);
            % Extract the green value from the bayer image
            green = test_image(i, j);
            % Assign the color values to the pixel in the RGB image
            rgb_image(i, j, 1) = red_predict;
            rgb_image(i, j, 2) = green;
            rgb_image(i, j, 3) = blue_predict;
            
        % For the type three pixel, use the dataset_five and dataset_six
        elseif center_pixel == 3
            % Transfer the patch to a 1x25 vector
            patch_vector = reshape(patch, 1, 25);
            % Convert the patch vector to a double type
            patch_vector = double(patch_vector);
            % Predict the value of the red for the pixel in the center of the patch
            red = patch_vector * coeff_five;
            % Select the first row element as the red predict value
            red_predict = red(1);
            % Predict the value of the blue for the pixel in the center of the patch
            blue = patch_vector * coeff_six;
            % Select the first row element as the green predict value
            blue_predict = blue(1);
            % Extract the green value from the bayer image
            green = test_image(i, j);
            % Assign the color values to the pixel in the RGB image
            rgb_image(i, j, 1) = red_predict;
            rgb_image(i, j, 2) = green;
            rgb_image(i, j, 3) = blue_predict;
             
            

        % For the type four pixel, use the dataset_seven and dataset_eight
        elseif center_pixel == 4
            % Transfer the patch to a 1x25 vector
            patch_vector = reshape(patch, 1, 25);
            % Convert the patch vector to a double type
            patch_vector = double(patch_vector);
            % Predict the value of the red for the pixel in the center of the patch
            red = patch_vector * coeff_seven;
            % Select the first row element as the red predict value
            red_predict = red(1);
            % Predict the value of the green for the pixel in the center of the patch
            green = patch_vector * coeff_eight;
            % Select the first row element as the green predict value
            green_predict = green(1);
            % Extract the blue value from the bayer image
            blue = test_image(i, j);
            % Assign the color values to the pixel in the RGB image
            rgb_image(i, j, 1) = red_predict;
            rgb_image(i, j, 2) = green_predict;
            rgb_image(i, j, 3) = blue;
           
        end
    end
end


% Convert the RGB image to the range of 0 to 255 integer of uint8
rgb_image = uint8(rgb_image);

% Display the RGB image and save it as a .png file, file name is the same as the bayer image with the suffix '_rgb'
figure;
imshow(rgb_image);
imwrite(rgb_image, strcat('matlabRGB_test2', '.png'));
 
 
 