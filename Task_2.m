% Directory setup
inputDir = 'data/ball_frames/'; % Input directory
outputDir = 'Assets/Features/';  % Output directory for extracted features and images

% Create output directory if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get list of indexed images
imageFiles = dir(fullfile(inputDir, '*_indexed.png'));
numImages = numel(imageFiles);

% Struct to store all features
allFeatures = struct();
featureIdx = 1;

function rangeVal = safe_range(x)
    x = x(~isnan(x));
    if isempty(x)
        rangeVal = NaN;
    else
        rangeVal = max(x) - min(x);
    end
end

ballTypes = {'Tennis Ball', 'Football', 'American Football'};

% Loop through all images
for i = 1:numImages
    imgPathIndexed = fullfile(inputDir, imageFiles(i).name);
    indexedImg = imread(imgPathIndexed);

    rgbImgName = strrep(imageFiles(i).name, '_indexed', '');
    imgPathRGB = fullfile(inputDir, rgbImgName);

    if ~isfile(imgPathRGB)
        warning('RGB image %s not found, skipping...', rgbImgName);
        continue;
    end
    
    rgbImg = imread(imgPathRGB);
    combinedImg = zeros(size(rgbImg), 'uint8');

    % Process each ball type
    ballRegions = unique(indexedImg);
    ballRegions(ballRegions == 0) = []; 

    for j = 1:length(ballRegions)
        mask = (indexedImg == ballRegions(j));
        mask = bwareaopen(mask, 100);
        
        maskedImg = rgbImg .* uint8(mask);
        BW = mask;
        
        % Extract shape features
        properties = regionprops(BW, 'Area', 'Perimeter', 'Solidity', 'MajorAxisLength', 'MinorAxisLength');
        if isempty(properties)
            continue;
        end
        
        % Select largest object
        [~, largestIdx] = max([properties.Area]);
        props = properties(largestIdx);
        
        area = props.Area;
        perimeter = props.Perimeter; % Use perimeter from regionprops
        solidity = props.Solidity;
        
        % Print area and perimeter for debugging
        fprintf('Object %d: Area = %.2f, Perimeter = %.2f\n', j, area, perimeter);
        
        % Validate perimeter to avoid division errors
        if perimeter > 0 && area > 50
            circularity = (4 * pi * area) / max(perimeter^2, eps);
            non_compactness = max(perimeter^2, eps) / max(area, eps);

            % Print for debug
            fprintf('   Circularity = %.4f, Non-Compactness = %.4f\n', circularity, non_compactness);
        else
            fprintf('   Skipping due to invalid perimeter/area.\n');
            continue;
        end
        
        if props.MajorAxisLength > 0
            eccentricity = sqrt(1 - (props.MinorAxisLength / props.MajorAxisLength)^2);
        else
            eccentricity = NaN;
        end

        % Texture Features - With manual Haralick calculations
        textureFeatures = struct();
        for channel = 1:3
            channelData = rgbImg(:,:,channel);
            channelData(~mask) = 0;
            channelData = uint8(channelData); % Assuming input is already uint8
            
            offsets = [0 1; -1 1; -1 0; -1 -1];
            glcm = graycomatrix(channelData, ...
                'Offset', offsets, ...
                'Symmetric', true, ...
                'NumLevels', 256, ...
                'GrayLimits', [0 255]);
            
            % Initialize storage for orientation features
            numOrientations = size(glcm, 3);
            ASM_vals = zeros(1, numOrientations);
            contrast_vals = zeros(1, numOrientations);
            correlation_vals = zeros(1, numOrientations);
            
            % Process each orientation
            for orient = 1:numOrientations
                glcm_slice = glcm(:,:,orient);
                glcm_slice = glcm_slice ./ sum(glcm_slice(:));  % Normalize GLCM manually for Haralick feature calculation
                
                [numLevels, ~] = size(glcm_slice);
                [iGrid, jGrid] = meshgrid(1:numLevels, 1:numLevels);
                
                % Angular Second Moment (Energy)
                ASM_vals(orient) = sum(glcm_slice(:).^2);
                
                % Contrast
                contrast_vals(orient) = sum(glcm_slice(:) .* (iGrid(:) - jGrid(:)).^2);
                
                % Correlation
                mu_i = sum(iGrid(:) .* glcm_slice(:));
                mu_j = sum(jGrid(:) .* glcm_slice(:));
                sigma_i = sqrt(sum(glcm_slice(:) .* (iGrid(:) - mu_i).^2));
                sigma_j = sqrt(sum(glcm_slice(:) .* (jGrid(:) - mu_j).^2));
                if sigma_i * sigma_j ~= 0
                    correlation_vals(orient) = sum(glcm_slice(:) .* (iGrid(:) - mu_i) .* (jGrid(:) - mu_j)) / (sigma_i * sigma_j);
                else
                    correlation_vals(orient) = NaN;
                end
            end
            
            % Store average and range across orientations
            textureFeatures(channel).ASM_avg = mean(ASM_vals);
            textureFeatures(channel).ASM_range = safe_range(ASM_vals);
            textureFeatures(channel).Contrast_avg = mean(contrast_vals);
            textureFeatures(channel).Contrast_range = safe_range(contrast_vals);
            textureFeatures(channel).Correlation_avg = mean(correlation_vals);
            textureFeatures(channel).Correlation_range = safe_range(correlation_vals);
        end
        
        switch ballRegions(j)
            case 1
                ball_type = 'Tennis Ball';
            case 2
                ball_type = 'Football';
            case 3
                ball_type = 'American Football';
            otherwise
                ball_type = sprintf('Unknown_%d', ballRegions(j));
        end
        
        combinedImg = min(combinedImg + maskedImg, 255);
        
        % Store Features
        allFeatures(featureIdx).ImageName = imageFiles(i).name;
        allFeatures(featureIdx).BallType = ball_type;
        allFeatures(featureIdx).ShapeFeatures = struct( ...
            'NonCompactness', non_compactness, ...
            'Circularity', circularity, ...
            'Solidity', solidity, ...
            'Eccentricity', eccentricity ...
        );
        allFeatures(featureIdx).TextureFeatures = struct( ...
            'Red', textureFeatures(1), ...
            'Green', textureFeatures(2), ...
            'Blue', textureFeatures(3) ...
        );
        
        featureIdx = featureIdx + 1;
    end
    
    % Save the combined image
    [~, nameOnly, ~] = fileparts(rgbImgName);
    combinedImgName = sprintf('combined_%s.png', nameOnly);
    imwrite(combinedImg, fullfile(outputDir, combinedImgName));
end

% Combined feature plots
shapeFeatures = {'NonCompactness', 'Circularity', 'Solidity', 'Eccentricity'};

% Create combined feature distributions
for k = 1:numel(shapeFeatures)
    fig = figure('Visible', 'off');
    hold on;
    title(['Combined Feature Distribution - ', shapeFeatures{k}], 'FontSize', 14);
    xlabel(shapeFeatures{k}, 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);

    colors = lines(numel(ballTypes));
    legendEntries = cell(1, numel(ballTypes));

    for b = 1:numel(ballTypes)
        values = arrayfun(@(x) x.ShapeFeatures.(shapeFeatures{k}), ...
            allFeatures(strcmp({allFeatures.BallType}, ballTypes{b})));
        
        histogram(values, ...
            'FaceColor', colors(b,:), ...
            'FaceAlpha', 0.6, ...
            'EdgeColor', 'none');
        
        legendEntries{b} = ballTypes{b};
    end

    legend(legendEntries, 'Location', 'best');
    grid on;
    hold off;

    saveas(fig, fullfile(outputDir, sprintf('Combined_FeatureDist_%s.png', shapeFeatures{k})));
    close(fig);
end

% Per-ball-type feature distributions
ballTypes = {'Tennis Ball', 'Football', 'American Football'};
shapeFeatures = {'NonCompactness', 'Circularity', 'Solidity', 'Eccentricity'};... % Individual feature plots per ball type
for b = 1:numel(ballTypes)
    currentType = ballTypes{b};
    typeFeatures = allFeatures(strcmp({allFeatures.BallType}, currentType));
    
    if isempty(typeFeatures)
        continue;
    end
    
    for k = 1:numel(shapeFeatures)
        fig = figure('Visible', 'off');
        
        % Explicitly extract values
        featureValues = arrayfun(@(x) x.ShapeFeatures.(shapeFeatures{k}), typeFeatures);
        
        histogram(featureValues, ...
            'FaceColor', [0.2 0.6 0.9], ...
            'EdgeColor', 'none', ...
            'NumBins', 15);
        
        title(sprintf('%s - %s', currentType, shapeFeatures{k}), 'FontSize', 14);
        xlabel('Value', 'FontSize', 12);
        ylabel('Frequency', 'FontSize', 12);
        grid on;
        
        saveas(fig, fullfile(outputDir, sprintf('%s_%s.png', strrep(currentType, ' ', '_'), shapeFeatures{k})));
        close(fig);
    end
end

% Combined texture feature distributions for all ball types
selectedFeatures = {
    {'Blue', 'ASM_avg', 'Angular Second Moment (Blue)'}, ...
    {'Red', 'Contrast_range', 'Contrast Range (Red)'},  ...
    {'Green', 'Correlation_avg', 'Correlation (Green)'}
};

colors = lines(numel(ballTypes)); % Distinct colors for each ball type

% Create combined plots for each texture feature
% Separate combined texture feature histograms
for fIdx = 1:numel(selectedFeatures)
    channel = selectedFeatures{fIdx}{1};
    featureName = selectedFeatures{fIdx}{2};
    featureTitle = selectedFeatures{fIdx}{3};
    
    for bIdx = 1:numel(ballTypes)
        fig = figure('Visible', 'off');
        hold on;
        
        % Extract features for current ball type
        typeFeatures = allFeatures(strcmp({allFeatures.BallType}, ballTypes{bIdx}));
        if isempty(typeFeatures)
            continue;
        end
        values = arrayfun(@(x) x.TextureFeatures.(channel).(featureName), typeFeatures);
        
        histogram(values, ...
            'FaceColor', colors(bIdx,:), ...
            'FaceAlpha', 0.6, ...
            'EdgeColor', 'none', ...
            'NumBins', 15);
        
        title(sprintf('%s - %s', ballTypes{bIdx}, featureTitle), 'FontSize', 14);
        xlabel('Feature Value', 'FontSize', 12);
        ylabel('Frequency', 'FontSize', 12);
        grid on;
        
        saveas(fig, fullfile(outputDir, sprintf('%s_Texture_%s_%s.png', ...
            strrep(ballTypes{bIdx}, ' ', '_'), channel, featureName)));
        close(fig);
    end
end

% Define custom colors for each ball type=
ballTypeColors = containers.Map( ...
    {'Tennis Ball', 'Football', 'American Football'}, ...
    {[1, 1, 0], [1, 1, 1], [1, 0.5, 0]} ...
);


for fIdx = 1:numel(selectedFeatures)
    channel = selectedFeatures{fIdx}{1};
    featureName = selectedFeatures{fIdx}{2};
    featureTitle = selectedFeatures{fIdx}{3};

    % Collect data and group labels
    data = [];
    groupLabels = {};
    groupIndices = [];

    for bIdx = 1:numel(ballTypes)
        % Extract features for current ball type
        typeFeatures = allFeatures(strcmp({allFeatures.BallType}, ballTypes{bIdx}));
        if isempty(typeFeatures)
            continue;
        end
        values = arrayfun(@(x) x.TextureFeatures.(channel).(featureName), typeFeatures);

        data = [data; values(:)];
        groupLabels = [groupLabels; repmat(ballTypes(bIdx), numel(values), 1)];
        groupIndices = [groupIndices; bIdx * ones(numel(values), 1)];
    end

    if isempty(data)
        continue;
    end

    % Create boxplot
    fig = figure('Visible', 'off');
    boxHandle = boxplot(data, groupLabels, 'Colors', 'k', 'Symbol', 'o', 'OutlierSize', 4);
    title([featureTitle ' Boxplot'], 'FontSize', 14);
    ylabel('Feature Value', 'FontSize', 12);
    grid on;

    % Customize box colors using labels
    boxes = findobj(gca, 'Tag', 'Box');
    labels = get(gca, 'XTickLabel');
    
    % Reverse the order of boxes
    for j = 1:length(boxes)
        label = strtrim(labels{j}); % Label from left to right
        boxIndex = length(boxes) - j + 1; % Reverse index to match label order
        if isKey(ballTypeColors, label)
            c = ballTypeColors(label); % RGB triplet
            patch(get(boxes(boxIndex), 'XData'), get(boxes(boxIndex), 'YData'), c, ...
                'FaceAlpha', 0.6, 'EdgeColor', 'k');
        end
    end

    % Save plot
    saveas(fig, fullfile(outputDir, sprintf('Boxplot_Texture_%s_%s.png', channel, featureName)));
    close(fig);
end