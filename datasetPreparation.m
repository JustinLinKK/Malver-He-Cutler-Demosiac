train_folder = 'train/';
% Get a list of all the image files in the folder
image_files = dir(fullfile(train_folder, '*.png'));
% Loop through each image file in the folder

for i = 1:length(image_files)
    % Read the image using imread function
    image = imread(fullfile(train_folder, image_files(i).name));



    % Check if the image is already in RGB format
    if size(image, 3) == 3
        % The image is already in RGB format, so no need to convert it
        rgb_image = image;
    else
        % The image is in grayscale format, so convert it to RGB
        rgb_image = cat(3, image, image, image);
    end

    % Define the Bayer pattern (GRBG)
    bayer_pattern = [1 2; 2 3];

    % Create a blank Bayer mosaic image with the same size as the RGB image
    bayer_image = zeros(size(rgb_image, 1), size(rgb_image, 2));

    % Assign color values to each pixel based on the Bayer pattern of rggb
    for row = 1:size(bayer_image, 1)
        for col = 1:size(bayer_image, 2)
            if mod(row, 2) == 1 % odd row
                if mod(col, 2) == 1 % odd column
                    bayer_image(row, col) = rgb_image(row, col, 1); % red
                else % even column
                    bayer_image(row, col) = rgb_image(row, col, 2); % green type 1
                end
            else % even row
                if mod(col, 2) == 1 % odd column
                    bayer_image(row, col) = rgb_image(row, col, 2); % green type 2
                else % even column
                    bayer_image(row, col) = rgb_image(row, col, 3); % blue
                end
            end
        end
    end

    % Convert the Bayer image to uint8 format
    bayer_image = uint8(bayer_image);

    % Save the Bayer image with a "bayer_" prefix
    imwrite(bayer_image, fullfile(train_folder, ['bayer_' image_files(i).name]));
    disp("image bayer created");


    % Create a blank index Bayer matrix with the same size as the RGB image
    index_bayer = zeros(size(rgb_image, 1), size(rgb_image, 2));

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


    % Create eight coefficient matrixs
    coefficient_matrix_one = zeros(25, 1);
    coefficient_matrix_two = zeros(25, 1);
    coefficient_matrix_three = zeros(25, 1);
    coefficient_matrix_four = zeros(25, 1);
    coefficient_matrix_five = zeros(25, 1);
    coefficient_matrix_six = zeros(25, 1);
    coefficient_matrix_seven = zeros(25, 1);
    coefficient_matrix_eight = zeros(25, 1);

    % Create eight different datasets
    dataset_one = zeros(1, 25);
    dataset_two = zeros(1, 25);
    dataset_three = zeros(1, 25);
    dataset_four = zeros(1, 25);
    dataset_five = zeros(1, 25);
    dataset_six = zeros(1, 25);
    dataset_seven = zeros(1, 25);
    dataset_eight = zeros(1, 25);

    % Create eight different ground truth matrices
    ground_truth_one = zeros(1, 1);
    ground_truth_two = zeros(1, 1);
    ground_truth_three = zeros(1, 1);
    ground_truth_four = zeros(1, 1);
    ground_truth_five = zeros(1, 1);
    ground_truth_six = zeros(1, 1);
    ground_truth_seven = zeros(1, 1);
    ground_truth_eight = zeros(1, 1);


    % loop through the Bayer image matrix to extract the patches
    for i = 1:size(bayer_image, 1)
        for j = 1:size(bayer_image, 2)
            % Check if the current pixel is not on the edge of the image
            if i > 2 && i < size(bayer_image, 1) - 2 && j > 2 && j < size(bayer_image, 2) - 2
                % Create a 5x5 matrix around the current pixel, record the values of this patch
                patch = bayer_image(i - 2:i + 2, j - 2:j + 2);
                % Identify the type of pixel in the center of the patch
            else % If it is on the edge of the image, fill the blank positions with medium gray
                patch = zeros(5, 5);
                patch(3, 3) = 128;
            end
            % First, fill the position that not out of the bayer image
            for k = 1:5
                for l = 1:5
                    if i - 3 + k > 0 && i - 3 + k <= size(bayer_image, 1) && j - 3 + l > 0 && j - 3 + l <= size(bayer_image, 2)
                        patch(k, l) = bayer_image(i - 3 + k, j - 3 + l);
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
                % Extract the ground truth value of the blue for the pixel in the center of the patch
                ground_truth_blue = rgb_image(i, j, 3);
                % Extract the ground truth value of the green for the pixel in the center of the patch
                ground_truth_green = rgb_image(i, j, 2);
                % Append the patch vector to the dataset_one and dataset_two as bottom row
                dataset_one = [dataset_one; patch_vector];
                dataset_two = [dataset_two; patch_vector];
                % Append the ground truth value of the blue to the ground_truth_one and ground_truth_two as bottom row
                ground_truth_one = [ground_truth_one; ground_truth_blue];
                ground_truth_two = [ground_truth_two; ground_truth_blue];

            % For the type two pixel, use the dataset_three and dataset_four
            elseif center_pixel == 2
                % Transfer the patch to a 1x25 vector
                patch_vector = reshape(patch, 1, 25);
                % Extract the ground truth value of the red for the pixel in the center of the patch
                ground_truth_red = rgb_image(i, j, 1);
                % Extract the ground truth value of the blue for the pixel in the center of the patch
                ground_truth_blue = rgb_image(i, j, 3);
                % Append the patch vector to the dataset_three and dataset_four as bottom row
                dataset_three = [dataset_three; patch_vector];
                dataset_four = [dataset_four; patch_vector];
                % Append the ground truth value of the green to the ground_truth_three and ground_truth_four as bottom row
                ground_truth_three = [ground_truth_three; ground_truth_red];
                ground_truth_four = [ground_truth_four; ground_truth_blue];

            % For the type three pixel, use the dataset_five and dataset_six
            elseif center_pixel == 3
                % Transfer the patch to a 1x25 vector
                patch_vector = reshape(patch, 1, 25);
                % Extract the ground truth value of the red for the pixel in the center of the patch
                ground_truth_red = rgb_image(i, j, 1);
                % Extract the ground truth value of the blue for the pixel in the center of the patch
                ground_truth_blue = rgb_image(i, j, 3);
                % Append the patch vector to the dataset_five and dataset_six as bottom row
                dataset_five = [dataset_five; patch_vector];
                dataset_six = [dataset_six; patch_vector];
                % Append the ground truth value of the green to the ground_truth_five and ground_truth_six as bottom row
                ground_truth_five = [ground_truth_five; ground_truth_red];
                ground_truth_six = [ground_truth_six; ground_truth_green];

            % For the type four pixel, use the dataset_seven and dataset_eight
            elseif center_pixel == 4
                % Transfer the patch to a 1x25 vector
                patch_vector = reshape(patch, 1, 25);
                % Extract the ground truth value of the red for the pixel in the center of the patch
                ground_truth_red = rgb_image(i, j, 2);
                % Extract the ground truth value of the green for the pixel in the center of the patch
                ground_truth_green = rgb_image(i, j, 1);
                % Append the patch vector to the dataset_seven and dataset_eight as bottom row
                dataset_seven = [dataset_seven; patch_vector];
                dataset_eight = [dataset_eight; patch_vector];
                % Append the ground truth value of the green to the ground_truth_seven and ground_truth_eight as bottom row
                ground_truth_seven = [ground_truth_seven; ground_truth_red];
                ground_truth_eight = [ground_truth_eight; ground_truth_green];
            end
        end
    end
    disp("image dataset created");
end

% Remove the first row of zeros in the dataset and ground truth
dataset_one = dataset_one(2:end, :);
dataset_two = dataset_two(2:end, :);
dataset_three = dataset_three(2:end, :);
dataset_four = dataset_four(2:end, :);
dataset_five = dataset_five(2:end, :);
dataset_six = dataset_six(2:end, :);
dataset_seven = dataset_seven(2:end, :);
dataset_eight = dataset_eight(2:end, :);
ground_truth_one = ground_truth_one(2:end, :);
ground_truth_two = ground_truth_two(2:end, :);
ground_truth_three = ground_truth_three(2:end, :);
ground_truth_four = ground_truth_four(2:end, :);
ground_truth_five = ground_truth_five(2:end, :);
ground_truth_six = ground_truth_six(2:end, :);
ground_truth_seven = ground_truth_seven(2:end, :);
ground_truth_eight = ground_truth_eight(2:end, :);

% Output "preparation done! Start training..."
disp("preparation done! ");
% Save the dataset and ground truth to the file "dataset.mat"


 


