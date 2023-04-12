
% Load the dataset and ground_truth from the mat file named "dataset.mat"
load('dataset.mat');

% Convert all the data to double type
dataset_one = double(dataset_one);
dataset_two = double(dataset_two);
dataset_three = double(dataset_three);
dataset_four = double(dataset_four);
dataset_five = double(dataset_five);
dataset_six = double(dataset_six);
dataset_seven = double(dataset_seven);
dataset_eight = double(dataset_eight);
ground_truth_one = double(ground_truth_one);
ground_truth_two = double(ground_truth_two);
ground_truth_three = double(ground_truth_three);
ground_truth_four = double(ground_truth_four);
ground_truth_five = double(ground_truth_five);
ground_truth_six = double(ground_truth_six);
ground_truth_seven = double(ground_truth_seven);
ground_truth_eight = double(ground_truth_eight);



% Get the coefficient of the linear regression model form the dataset and ground_truth
coeff_one = inv(transpose(dataset_one)*dataset_one)*transpose(dataset_one)*ground_truth_one;
coeff_two = inv(transpose(dataset_two)*dataset_two)*transpose(dataset_two)*ground_truth_two;
coeff_three = inv(transpose(dataset_three)*dataset_three)*transpose(dataset_three)*ground_truth_three;
coeff_four = inv(transpose(dataset_four)*dataset_four)*transpose(dataset_four)*ground_truth_four;
coeff_five = inv(transpose(dataset_five)*dataset_five)*transpose(dataset_five)*ground_truth_five;
coeff_six = inv(transpose(dataset_six)*dataset_six)*transpose(dataset_six)*ground_truth_six;
coeff_seven = inv(transpose(dataset_seven)*dataset_seven)*transpose(dataset_seven)*ground_truth_seven;
coeff_eight = inv(transpose(dataset_eight)*dataset_eight)*transpose(dataset_eight)*ground_truth_eight;


% Save the coefficient of the linear regression model to the mat file named "coefficients.mat"
save('coefficients.mat', 'coeff_one', 'coeff_two', 'coeff_three', 'coeff_four', 'coeff_five', 'coeff_six', 'coeff_seven', 'coeff_eight');