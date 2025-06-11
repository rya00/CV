function plot_dice_scores(dice_scores, image_names, savePath, finalMaskFolder, inputFolder, segmentationDifference)
    % This function generates and saves three types of plots:
    % 1. Bar plot for Dice scores with color coding
    % 2. Violin plot for the distribution of Dice scores
    % 3. Box plot for the Dice scores
    % 4. Top and Worst 5 Dice Score Visualization

    %% Bar Plot of Dice Scores
    % Ensure figure is clear
    clf;

    % Number of bars (images) in the bar plot
    numBars = length(dice_scores);

    % Bar width
    barWidth = 0.5;

    % X-coordinates for bars (from 1 to numBars)
    xPositions = 1:numBars;

    % Plot bars manually using `fill` function
    hold on;
    for i = 1:numBars
        % Define the rectangle (bar) corners based on the current position and width
        x = [xPositions(i)-barWidth/2, xPositions(i)+barWidth/2, xPositions(i)+barWidth/2, xPositions(i)-barWidth/2];
        y = [0, 0, dice_scores(i), dice_scores(i)];

        % Assign colors based on Dice score values
        if dice_scores(i) > 0.8
            color = [0, 0.7, 0]; % Green for high scores
        elseif dice_scores(i) > 0.5
            color = [0.9, 0.6, 0]; % Orange for medium scores
        else
            color = [0.8, 0, 0]; % Red for low scores
        end

        % Plot the bar using `fill`
        fill(x, y, color, 'EdgeColor', 'k');
    end
    hold off;

    % Customize axes limits
    xlim([0 numBars+1]); % X-axis range
    ylim([0 1]); % Dice score range is from 0 to 1

    % Set labels and title for the bar plot
    % Extract only the numeric part from filenames
    numeric_labels = cellfun(@(x) regexp(x, '\d+', 'match', 'once'), image_names, 'UniformOutput', false);
    
    % Set the x-axis labels using extracted numbers
    set(gca, 'XTick', xPositions, 'XTickLabel', numeric_labels, 'XTickLabelRotation', 45, 'FontSize', 10);
    xlabel('Image Index');

    ylabel('Dice Score');
    title('Dice Score Comparison');

    % Display grid for better visibility
    grid on;

    % Save the bar plot to a file inside the 'Assets' folder
    saveas(gcf, fullfile(savePath, 'dice_score_bar_plot.png'));

    %% Violin Plot for Dice Scores
    % Step 1: Compute Kernel Density Estimation (KDE) using ksdensity
    [PDF, x_values] = ksdensity(dice_scores);

    % Step 2: Create the figure for the violin plot
    figure;
    hold on;

    % Plot the "left" side of the violin (mirrored)
    fill([-PDF, fliplr(PDF)], [x_values, fliplr(x_values)], 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.5);

    % Plot the "right" side of the violin (mirrored)
    fill([PDF, -fliplr(PDF)], [x_values, fliplr(x_values)], 'b', 'EdgeColor', 'k', 'FaceAlpha', 0.5);

    % Customize the plot labels and title
    xlabel('Density');
    ylabel('Dice Score');
    title('Violin Plot of Dice Scores');
    grid on;
    hold off;

    % Save the violin plot to a file inside the 'Assets' folder
    saveas(gcf, fullfile(savePath, 'dice_score_violin_plot.png'));

    %% Box Plot for Dice Scores
    % Create the figure for the box plot
    figure;

    % Create a horizontal boxplot for Dice Scores
    boxplot(dice_scores, 'Orientation', 'horizontal', 'Widths', 0.5);

    % Customize the plot labels and title
    xlabel('Dice Score');
    title('Box Plot of Dice Scores');
    grid on;

    % Save the box plot to a file inside the 'Assets' folder
    saveas(gcf, fullfile(savePath, 'dice_score_box_plot.png'));

    %% Top 5 and Bottom 5 Dice Scores with Mask Images
    % Sort Dice scores and get indices for top 5 best and bottom 5 worst segmentations
    [sorted_scores, sorted_indices] = sort(dice_scores, 'descend'); % Sort in descending order
    
    best_indices = sorted_indices(1:5); % Top 5 best (highest Dice scores)
    worst_indices = sorted_indices(end-4:end); % Bottom 5 worst (lowest Dice scores)
    
    % Plot Best Segmentations
    figure;
    set(gcf, 'Position', [100, 100, 3500, 3000]);
    t_best = tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_best, 'Top 5 Best Segmentations', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Loop through top 5 best segmentations
    for i = 1:length(best_indices)
        idx = best_indices(i);
        img_name = image_names{idx};
    
        % Load images
        segmented_img = imread(fullfile(finalMaskFolder, [img_name, '_final_mask.png']));
        ground_truth_img = imread(fullfile(inputFolder, [img_name, '_GT.png']));
    
        % Convert to grayscale if needed
        if size(ground_truth_img, 3) == 3
            ground_truth_img = rgb2gray(ground_truth_img);
        end
        if size(segmented_img, 3) == 3
            segmented_img = rgb2gray(segmented_img);
        end
    
        % Ensure both images have the same class type (convert to uint8)
        if ~isa(segmented_img, 'uint8')
            segmented_img = im2uint8(segmented_img);
        end
        if ~isa(ground_truth_img, 'uint8')
            ground_truth_img = im2uint8(ground_truth_img);
        end
    
        % Compute difference mask
        difference_mask = imabsdiff(segmented_img, ground_truth_img);
    
        % Plot images
        nexttile;
        imshow(ground_truth_img);
        title(['GT: ', img_name], 'Interpreter', 'none', 'FontSize', 12);
    
        nexttile;
        imshow(segmented_img);
        title(['Seg: ', num2str(dice_scores(idx), '%.3f')], 'FontSize', 12);
    
        nexttile;
        imshow(difference_mask, []);
        title('Difference Mask', 'FontSize', 12);
    end
    
    % Save the figure for the best segmentations
    saveas(gcf, fullfile(savePath, 'Top_5_Best_Segmentations.png'));
    
    disp('Top 5 Best Segmentations plotted and saved.');
    
    % Plot Worst Segmentations
    figure;
    set(gcf, 'Position', [100, 100, 3500, 3000]);
    t_worst = tiledlayout(5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t_worst, 'Bottom 5 Worst Segmentations', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Loop through bottom 5 worst segmentations
    for i = 1:length(worst_indices)
        idx = worst_indices(i);
        img_name = image_names{idx};
    
        % Load images
        segmented_img = imread(fullfile(finalMaskFolder, [img_name, '_final_mask.png']));
        ground_truth_img = imread(fullfile(inputFolder, [img_name, '_GT.png']));
    
        % Convert to grayscale if needed
        if size(ground_truth_img, 3) == 3
            ground_truth_img = rgb2gray(ground_truth_img);
        end
        if size(segmented_img, 3) == 3
            segmented_img = rgb2gray(segmented_img);
        end
    
        % Ensure both images have the same class type (convert to uint8)
        if ~isa(segmented_img, 'uint8')
            segmented_img = im2uint8(segmented_img);
        end
        if ~isa(ground_truth_img, 'uint8')
            ground_truth_img = im2uint8(ground_truth_img);
        end
    
        % Compute difference mask
        difference_mask = imabsdiff(segmented_img, ground_truth_img);
    
        % Plot images
        nexttile;
        imshow(ground_truth_img);
        title(['GT: ', img_name], 'Interpreter', 'none', 'FontSize', 12);
    
        nexttile;
        imshow(segmented_img);
        title(['Seg: ', num2str(dice_scores(idx), '%.3f')], 'FontSize', 12);
    
        nexttile;
        imshow(difference_mask, []);
        title('Difference Mask', 'FontSize', 12);
    end
    
    % Save the figure for the worst segmentations
    saveas(gcf, fullfile(savePath, 'Bottom_5_Worst_Segmentations.png'));
    
    disp('Bottom 5 Worst Segmentations plotted and saved.');
end
