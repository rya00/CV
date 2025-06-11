function [top_dice_scores, worst_dice_scores] = dice_score(processed_images, masks, save_path)
    % This function computes the Dice similarity coefficient (Dice score)
    % for a set of processed images (predicted masks) and their corresponding ground truth masks.
    % It saves the computed Dice scores and average score to a text file in the specified save path.
    
    % Initialize an array to hold Dice scores for each image
    eval = [];
    
    % Initialize a map to store Dice scores with image names as keys
    score_dict = containers.Map;
    
    % Ensure the directory for saving results exists, if not create it
    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    % Loop through each pair of processed (predicted) and ground truth masks
    for idx = 1:length(processed_images)
        % Read the predicted mask (segmented image)
        pred_mask = imread(processed_images{idx});
        
        % Read the ground truth mask
        gt_mask = imread(masks{idx});
        
        % If the predicted mask is in RGB format, convert it to grayscale
        if size(pred_mask, 3) == 3
            pred_mask = rgb2gray(pred_mask);
        end
        
        % If the ground truth mask is in RGB format, convert it to grayscale
        if size(gt_mask, 3) == 3
            gt_mask = rgb2gray(gt_mask);
        end

        % Convert both the predicted and ground truth masks to double precision
        % This is necessary for binarization and Dice score computation
        pred_mask = im2double(pred_mask);
        gt_mask = im2double(gt_mask);

        % Ensure both masks are binarized using automatic thresholding (Otsu's method)
        pred_mask = imbinarize(pred_mask, graythresh(pred_mask));
        gt_mask = imbinarize(gt_mask, graythresh(gt_mask));

        % Compute the Dice similarity coefficient for the current pair of masks
        score = dice_coefficient(pred_mask, gt_mask);
        
        % Store the Dice score in the map with the image name as the key
        [~, name, ~] = fileparts(processed_images{idx}); % Extract image name (without extension)
        score_dict(name) = score;
        
        % Append the score to the eval array for later averaging
        eval = [eval, score];
    end
    
    % Compute the average Dice score across all images
    avg_score = mean(eval);
    
    % Display the average Dice score in the console
    disp(['Average Dice Score: ', num2str(avg_score)]);
    
    % Define the path for saving the Dice scores in a text file
    score_file = fullfile(save_path, 'dice_score.txt');
    
    % Open the file for writing
    fileID = fopen(score_file, 'w');
    
    % Check if the file opened successfully
    if fileID == -1
        error(['Error opening file: ', score_file]);
    end
    
    % Write the average Dice score to the file
    fprintf(fileID, 'Avg Score: %f\n', avg_score);
    
    % Write each individual Dice score along with the corresponding mask file name
    for idx = 1:length(eval)
        fprintf(fileID, '\t%f\t%s\n', eval(idx), masks{idx});
    end
    
    % Close the file after writing
    fclose(fileID);
    
    % Display a message indicating that the Dice scores have been saved
    disp(['Dice scores saved to: ', score_file]);

    % Sort the Dice scores and get the top 5 and worst 5
    scores = cell2mat(values(score_dict)); % Convert the map values (Dice scores) to an array
    names = keys(score_dict); % Get the image names

    % Sort scores in descending order
    [sorted_scores, sorted_idx] = sort(scores, 'descend');
    top_dice_scores = sorted_scores(1:5); % Top 5 highest scores
    top_names = names(sorted_idx(1:5)); % Corresponding names for top 5

    worst_dice_scores = sorted_scores(end-4:end); % Bottom 5 lowest scores
    worst_names = names(sorted_idx(end-4:end)); % Corresponding names for worst 5
    
    % Return the top and worst scores and names for further plotting
    disp('Top 5 Best DSC Scores:');
    disp(table(top_names', top_dice_scores', 'VariableNames', {'ImageName', 'DiceScore'}));
    
    disp('Top 5 Worst DSC Scores:');
    disp(table(worst_names', worst_dice_scores', 'VariableNames', {'ImageName', 'DiceScore'}));
end
