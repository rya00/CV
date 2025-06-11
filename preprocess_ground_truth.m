function ground_truth = preprocess_ground_truth(ground_truth)
    if size(ground_truth, 3) == 3
        ground_truth = rgb2gray(ground_truth);
    end
    ground_truth = imbinarize(im2gray(ground_truth));
end
