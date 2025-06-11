%% Function to Compute Dice Coefficient
function dice = dice_coefficient(pred_mask, gt_mask)
    pred_mask = logical(pred_mask);
    gt_mask = logical(gt_mask);
    intersection = sum(pred_mask(:) & gt_mask(:));
    total_pixels = sum(pred_mask(:)) + sum(gt_mask(:));
    
    if total_pixels == 0
        dice = 1;
    else
        dice = (2 * intersection) / total_pixels;
    end
end