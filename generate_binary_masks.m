%% Function to Generate Binary Masks
function final_mask = generate_binary_masks(image_path, finalMaskFolder, debug_mode)
    % This function processes an input image to generate a binary mask for segmentation. 
    % It uses HSV color space for object detection, applies glare reduction, 
    % enhances the image, and performs several morphological operations 
    % to create a refined mask. The final mask is saved and optionally visualized for debugging.

    % Step 1: Read the input image
    img = imread(image_path); % Load the image from the provided path
    
    % Step 2: Convert to HSV color space
    % The HSV color space is useful for color-based segmentation.
    hsv_img = rgb2hsv(img); % Convert the RGB image to HSV
    hue = hsv_img(:,:,1); % Extract hue channel
    sat = hsv_img(:,:,2); % Extract saturation channel
    val = hsv_img(:,:,3); % Extract value (brightness) channel
    
    % Step 3: Detect orange objects using color thresholding in the HSV space
    % This step creates a binary mask for orange-colored objects based on hue, 
    % saturation, and value thresholds.
    orange_mask = (hue >= 0.02 & hue <= 0.10) & (sat > 0.38) & (val > 0.2); % Define the color range for orange
    
    % Step 4: Glare Reduction (Top-hat filtering)
    % The glare removal reduces bright spots in the image that may be considered as noise.
    gray = rgb2gray(img); % Convert the image to grayscale
    se = strel('disk', 15); % Create a disk-shaped structuring element
    glare_removed = imtophat(gray, se); % Apply top-hat transform to remove glare
    
    % Step 5: Enhancement and Thresholding
    % The glare-reduced image is then enhanced and thresholded to create a binary mask.
    enhanced = imadjust(medfilt2(glare_removed, [5 5])); % Apply median filter and adjust intensity
    binary = imbinarize(enhanced, graythresh(enhanced) * 0.85); % Threshold the enhanced image
    
    % Step 6: Invert the binary mask if the image is too bright
    % If the mean of the binary mask is greater than 0.5 (i.e., the mask is mostly white),
    % invert the binary mask to improve the result.
    if mean(binary(:)) > 0.5
        binary = imcomplement(binary); % Invert the binary mask
    end
    
    % Step 7: Combine the HSV-based mask and the binary mask
    % The orange object detection mask is combined with the binary mask from thresholding.
    combined_mask = binary | orange_mask; % Combine both masks
    
    % Step 8: Noise Removal (Morphological Opening)
    % This step removes small objects that are likely to be noise.
    cleaned_mask = imopen(combined_mask, strel('disk', 5)); % Apply morphological opening to remove noise
    
    % Step 9: Size Filtering (Area-based Filtering)
    % This step filters objects based on their area. Only objects with a size between 
    % 300 and 10000 pixels are kept.
    filtered_mask = bwareafilt(cleaned_mask, [300, 10000]);  % Filter by object size
    cleaned_mask = imopen(filtered_mask, strel('disk', 3)); % Apply further morphological opening
    
    % Step 10: Hole Filling
    % This step fills any holes within the detected objects to ensure solid regions.
    filled_mask = imfill(cleaned_mask, 'holes'); % Fill holes inside objects
    cleaned_mask = imopen(filled_mask, strel('disk', 6)); % Apply opening again to refine
    
    % Step 11: Morphological Processing
    % Several morphological operations are applied to refine the mask further:
    % - Dilation to expand objects.
    % - Closing to close small gaps.
    % - Erosion to remove small artifacts.
    processed_mask = imdilate(cleaned_mask, strel('disk', 2)); % Dilate to expand
    processed_mask = imclose(processed_mask, strel('disk', 5)); % Close gaps in objects
    processed_mask = imerode(processed_mask, strel('disk', 2)); % Erode to refine edges
    
    % Step 12: Feature Extraction and Filtering
    % Extract features from the processed mask such as bounding box, eccentricity, 
    % solidity, and area, then filter objects based on these features.
    stats = regionprops(processed_mask, 'BoundingBox', 'Eccentricity', 'Solidity', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'EquivDiameter');
    [height, width] = size(processed_mask); % Get the size of the processed mask
    refined_mask = false(height, width); % Initialize an empty refined mask
    
    % Step 13: Object Refinement Based on Shape and Size Criteria
    % Filter objects based on shape properties (eccentricity, solidity, aspect ratio, circularity).
    for i = 1:length(stats)
        aspect_ratio = stats(i).MajorAxisLength / stats(i).MinorAxisLength; % Aspect ratio (roundness)
        circularity = (4 * pi * stats(i).Area) / (stats(i).EquivDiameter^2); % Circularity measure
        
        % Only keep round and well-defined objects
        if stats(i).Area > 300 && stats(i).Eccentricity < 0.85 && stats(i).Solidity > 0.6 ...
                && aspect_ratio < 2.5 && circularity > 0.6  % Keep round objects
            % Extract bounding box coordinates and update the refined mask
            y1 = max(1, round(stats(i).BoundingBox(2)));
            y2 = min(height, round(stats(i).BoundingBox(2) + stats(i).BoundingBox(4)));
            x1 = max(1, round(stats(i).BoundingBox(1)));
            x2 = min(width, round(stats(i).BoundingBox(1) + stats(i).BoundingBox(3)));
            temp_mask = false(height, width);
            temp_mask(y1:y2, x1:x2) = processed_mask(y1:y2, x1:x2);
            refined_mask = refined_mask | temp_mask; % Refine the mask by adding valid objects
        end
    end
    
    % Step 14: Remove Excessively Large Objects
    % Identify and remove objects that are too large (greater than 8000 pixels).
    CC = bwconncomp(refined_mask); % Find connected components
    numPixels = cellfun(@numel, CC.PixelIdxList); % Get the pixel count for each component
    largeObjects = find(numPixels > 8000); % Find large objects
    for i = 1:numel(largeObjects)
        refined_mask(CC.PixelIdxList{largeObjects(i)}) = 0; % Remove large objects from the mask
    end
    
    % Step 15: Apply Convex Hull for Final Shape
    % Apply convex hull to smooth the mask and make its boundary convex.
    convex_mask = bwconvhull(refined_mask, 'objects'); % Apply convex hull operation
    final_mask = convex_mask; % Set the final mask
    
    % Step 16: Debugging Mode (Optional)
    % If debug_mode is true, visualize intermediate steps for inspection.
    if debug_mode
        figure;
        [~, img_name, ~] = fileparts(image_path); % Extract image name from the file path
        sgtitle(['Processing: ', img_name], 'FontSize', 14, 'FontWeight', 'bold'); % Set title
        % Display the images in subplots
        subplot(2,5,1); imshow(img); title('Original');
        subplot(2,5,2); imshow(glare_removed); title('Glare Reduced');
        subplot(2,5,3); imshow(enhanced); title('Enhanced');
        subplot(2,5,4); imshow(combined_mask); title('Combined Mask');
        subplot(2,5,5); imshow(cleaned_mask); title('Noise Removed'); 
        subplot(2,5,6); imshow(filtered_mask); title('Filtered Objects');
        subplot(2,5,7); imshow(filled_mask); title('Filled Holes');
        subplot(2,5,8); imshow(refined_mask); title('Refined Mask');
        subplot(2,5,9); imshow(final_mask); title('Final Mask');
    end

    % Step 17: Save the Final Mask
    % Save the final processed binary mask to the specified folder.
    [~, name] = fileparts(image_path); % Extract the base name of the image file
    imwrite(final_mask, fullfile(finalMaskFolder, [name, '_final_mask.png'])); % Save the mask
end
