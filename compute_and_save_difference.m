function compute_and_save_difference(final_mask, ground_truth, name, segmentationDifference)
    % This function computes the difference between the predicted mask (final_mask) 
    % and the ground truth mask (ground_truth). It calculates the percentage of differing 
    % pixels, displays the result, and saves a difference image.
    
    % Step 1: Compute the difference mask
    % The difference mask highlights the pixels where the final mask differs from the ground truth.
    difference_mask = imabsdiff(final_mask, ground_truth); % Absolute difference between the two masks
    
    % Step 2: Count the number of differing pixels
    % Sum the number of non-zero pixels in the difference mask to count differing pixels.
    num_different_pixels = sum(difference_mask(:)); 
    
    % Step 3: Calculate the total number of pixels in the final mask
    % The total number of pixels is obtained using numel, which returns the number of elements in the mask.
    total_pixels = numel(final_mask); 
    
    % Step 4: Calculate the percentage difference
    % The percentage difference is the ratio of differing pixels to the total number of pixels.
    percentage_difference = (num_different_pixels / total_pixels) * 100;
        
    % Step 5: Display the difference percentage
    % Print the difference percentage for the current image (identified by the 'name' argument).
    fprintf('Difference Percentage for %s: %.2f%%\n', name, percentage_difference);
        
    % Step 6: Visualize the results
    % The following lines, if uncommented, would visualize the ground truth, final mask, and the difference mask in subplots.
    % figure;
    % sgtitle(['Difference: ', name], 'FontSize', 14, 'FontWeight', 'bold'); % Set title at the top
    % subplot(1,3,1); imshow(ground_truth); title('Ground Truth'); % Display ground truth image
    % subplot(1,3,2); imshow(final_mask); title('Final Mask'); % Display final mask image
    % subplot(1,3,3); imshow(difference_mask, []); title('Difference Mask'); % Display the difference mask

    % Step 7: Save the difference image
    % Save the difference mask as a PNG file with a name based on the 'name' argument in the specified directory.
    save_name = fullfile(segmentationDifference, [name, '_segmentation_difference.png']);
    
    % Save the current figure as a PNG image
    saveas(gcf, save_name); % Save as PNG in the specified directory
end
