close all; % Close all open figures

%% Main script for computing Dice scores
% Define paths for input, saving, and output folders
inputFolder = fullfile(userpath, 'Ball_frames'); % Path to the input image folder
savePath = fullfile(userpath,'29364727 Reeya Shrestha', 'Assets'); % Path where results and masks will be saved
finalMaskFolder = fullfile(savePath, 'Final_Mask'); % Folder to save final masks
segmentationDifference = fullfile(savePath, 'Segmentation_Difference'); % Folder to save segmentation differences

% Create the Final_Mask folder if it does not exist
if ~exist(finalMaskFolder, 'dir')
    mkdir(finalMaskFolder); % Create the folder
    disp('Final Mask folder created.'); % Notify user
end

% Create the Segmentation_Difference folder if it does not exist
if ~exist(segmentationDifference, 'dir')
    mkdir(segmentationDifference); % Create the folder
    disp('Segmentation Difference folder created.'); % Notify user
end

% Get a list of all PNG files in the input folder, excluding those with '_GT' in the name
images = dir(fullfile(inputFolder, '*.png')); % List of all PNG files in the input folder
images = images(~contains({images.name}, '_GT')); % Exclude ground truth files (_GT)

dice_scores = []; % Initialize an array to store Dice scores for averaging
processed_images = {}; % Cell array to store paths of processed (segmented) images
masks = {}; % Cell array to store paths of ground truth images
image_names = {}; % Cell array to store the names of processed images for plotting

% Loop through each image in the input folder
for i = 1:length(images)
    [~, name] = fileparts(images(i).name); % Extract the image name (without extension)
    
    % Skip files with '_GT' in the name (ground truth files)
    if contains(name, '_GT')
        continue; % Skip ground truth images
    end
    
    % Generate the final mask for the image
    final_mask = generate_binary_masks(fullfile(inputFolder, images(i).name), finalMaskFolder, false);
    
    % Generate the path for the ground truth image corresponding to the current image
    gt_name = fullfile(inputFolder, [name, '_GT.png']);
    
    % Check if the ground truth file exists
    if exist(gt_name, 'file')
        % Read the ground truth image
        ground_truth = imread(gt_name);
        
        % Process ground truth if it's a color image (convert to grayscale)
        if size(ground_truth, 3) == 3
            ground_truth = rgb2gray(ground_truth); % Convert to grayscale
        end
        ground_truth = imbinarize(im2gray(ground_truth)); % Binarize the ground truth image
        
        % Save the processed mask to the Final_Mask folder
        imwrite(final_mask, fullfile(finalMaskFolder, [name, '_final_mask.png']));
        
        % Store paths for Dice score calculation
        processed_images{end + 1} = fullfile(finalMaskFolder, [name, '_final_mask.png']);
        masks{end + 1} = gt_name;
        
        % Store image names for plotting later
        image_names{end + 1} = name;
        
        % Compute Dice score for the current image
        dice_value = dice_coefficient(final_mask, ground_truth); % Calculate the Dice score
        dice_scores = [dice_scores; dice_value]; % Append Dice score to the array
        
        % Display the Dice score for the current image
        fprintf('Dice Score for %s: %.4f\n', images(i).name, dice_value);

        % Compute and save difference between the processed mask and the ground truth
        compute_and_save_difference(final_mask, ground_truth, name, segmentationDifference);
        
    else
        % If ground truth is not found, notify the user
        fprintf('Ground truth not found for %s. Skipping Dice computation.\n', images(i).name);
    end
end

% Compute and display average Dice score across all images
if ~isempty(dice_scores)
    avg_dice = mean(dice_scores); % Calculate the average Dice score
    fprintf('\nAverage Dice Score across all images: %.4f\n', avg_dice); % Display the average Dice score
    std_dice = std(dice_scores);
    fprintf('Standard Deviation of Dice Scores: %.4f\n', std_dice);
else
    % If no Dice scores were computed due to missing ground truth images
    fprintf('\nNo Dice scores were computed due to missing ground truth images.\n');
end

% Compute and save the Dice scores for all images
dice_score(processed_images, masks, savePath);

% After all images have been processed, plot the Dice scores for each image
plot_dice_scores(dice_scores, image_names, savePath, finalMaskFolder, inputFolder, segmentationDifference)