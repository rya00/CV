% Load ground-truth and noisy position data
xTrue = csvread('./data/coordinates/x.csv');
yTrue = csvread('./data/coordinates/y.csv');
xNoisy = csvread('./data/coordinates/na.csv');
yNoisy = csvread('./data/coordinates/nb.csv');

% Directory setup
savePath = 'Assets/Tracking/'; % Specify the output directory where images will be saved

% Create output folder if it doesn't exist
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Initial Setup
timeStep = 0.5;
stateTransition = [1 timeStep 0 0; 0 1 0 0; 0 0 1 timeStep; 0 0 0 1]; % Constant velocity model
observationMatrix = [1 0 0 0; 0 0 1 0]; 
processNoise = diag([0.4^2, 0.6^2, 0.4^2, 0.6^2]);  % Process noise covariance Q
measurementNoise = 0.25 * eye(2);  % Measurement noise covariance R
validationThreshold = 1.3e6; % Gating threshold

% Initial state vector: [x, vx, y, vy]
initialState = [xNoisy(1); 0; yNoisy(1); 0];
P = eye(4); % Initial estimate covariance

% Initialise storage
n = length(xTrue);
estimatedX = zeros(n, 1);
estimatedY = zeros(n, 1);
rmseKalman = zeros(n, 1);
rmseMeasured = zeros(n, 1);

% Baseline Kalman Filter Loop
for t = 1:n
    % Predict next state and covariance
    predictedState = stateTransition * initialState;
    predictedCov = stateTransition * P * stateTransition' + processNoise;

    % Measurement update
    z = [xNoisy(t); yNoisy(t)]; % Noisy observation
    predictedObs = observationMatrix * predictedState;
    S = observationMatrix * predictedCov * observationMatrix' + measurementNoise;
    K = predictedCov * observationMatrix' / S; % Kalman gain

    % Mahalanobis Distance for outlier rejection
    mahalanobis = (z - predictedObs)' / S * (z - predictedObs);

    if mahalanobis <= validationThreshold
        % Update
        initialState = predictedState + K * (z - predictedObs);
        P = (eye(4) - K * observationMatrix) * predictedCov;
    else
        % Reject update
        disp(['Frame ' num2str(t) ': Measurement rejected (outside gate)']);
        initialState = predictedState;
    end

    % Store Results
    estimatedX(t) = initialState(1);
    estimatedY(t) = initialState(3);
    rmseMeasured(t) = sqrt((xNoisy(t) - xTrue(t))^2 + (yNoisy(t) - yTrue(t))^2);
    rmseKalman(t) = sqrt((estimatedX(t) - xTrue(t))^2 + (estimatedY(t) - yTrue(t))^2);
end

% Final summary stats for baseline
baseline_rmse_estimated_mean = mean(rmseKalman);
baseline_rmse_estimated_std = std(rmseKalman);
noisy_rmse_estimated_mean = mean(rmseMeasured);
noisy_rmse_estimated_std = std(rmseMeasured);

% Parameter Tuning
dt = 0.4;  % New timestep for fine-tuned model
F = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]; % Update state transition
H = observationMatrix;

% Define tuning range
qRange = linspace(0.001, 0.05, 10); % Process noise scaling
rRange = linspace(0.1, 0.2, 100); % Measurement noise scaling

% Initialise best performance tracker
bestRMSE = inf;
optimalQ = [];
optimalR = [];

for q = qRange
    for r = rRange
        Q = q * eye(4);
        R = r * eye(2);
        state = [xNoisy(1); 0; yNoisy(1); 0];
        P = eye(4);
        xOut = zeros(n, 1);
        yOut = zeros(n, 1);
        rmseTemp = zeros(n, 1);

        for t = 1:n
            % Predict
            statePred = F * state;
            P_pred = F * P * F' + Q;
            z = [xNoisy(t); yNoisy(t)];
            S = H * P_pred * H' + R;
            K = P_pred * H' / S;
            state = statePred + K * (z - H * statePred);
            P = (eye(4) - K * H) * P_pred;

            %Update
            xOut(t) = state(1);
            yOut(t) = state(3);
            rmseTemp(t) = sqrt((xOut(t) - xTrue(t))^2 + (yOut(t) - yTrue(t))^2);
        end

        % Track best performing Q/R values
        avgRMSE = mean(rmseTemp);
        if avgRMSE < bestRMSE
            bestRMSE = avgRMSE;
            optimalQ = Q;
            optimalR = R;
            x_estimated_trajectory = xOut;
            y_estimated_trajectory = yOut;
            rmse_estimated_finetuned_mean = avgRMSE;
            rmse_estimated_finetuned_std = std(rmseTemp);
            rmse_estimated_finetuned_mean_noisy = noisy_rmse_estimated_mean;
            
            % Direction specific errors
            x_std_error_baseline = std(estimatedX - xTrue);
            y_std_error_baseline = std(estimatedY - yTrue);
            x_std_error = std(x_estimated_trajectory - xTrue);
            y_std_error = std(y_estimated_trajectory - yTrue);
        end
    end
end

% Print Results
fprintf('\nBaseline RMSE: %.4f ± %.4f\n', baseline_rmse_estimated_mean, baseline_rmse_estimated_std);
fprintf('Noisy RMSE: %.4f ± %.4f\n', noisy_rmse_estimated_mean, noisy_rmse_estimated_std);
fprintf('\nTuned Kalman RMSE: %.4f\n', rmse_estimated_finetuned_mean);
fprintf('Optimal Q value: %.4f\n', optimalQ(1));
fprintf('Optimal R value: %.4f\n', optimalR(1));

% Custom Plotting
% Custom Colours
true_colour = [0.1 0.4 0.8];
noisy_colour = [1.0 0.6 0.2];
estimated_colour = [0.4 0.8 0.4];
outputDir = savePath;

% Baseline Trajectory Plot
baseline_rmse_str = sprintf('Estimated Trajectory, RMSE = %.4f', baseline_rmse_estimated_mean);
noisy_rmse_str = sprintf('Noisy Measurement, RMSE = %.4f', noisy_rmse_estimated_mean);
fig = figure('Visible', 'off');
plot(xTrue, yTrue, 'Color', true_colour, 'LineWidth', 2); hold on;
plot(xNoisy, yNoisy, 'Color', noisy_colour, 'LineWidth', 1.5);
plot(estimatedX, estimatedY, 'Color', estimated_colour, 'LineWidth', 1.5);
plot(estimatedX, estimatedY, 'x', 'Color', 'k', 'MarkerSize', 4, 'LineWidth', 1);
stylePlot('X Position', 'Y Position', 'Baseline Kalman Filter Trajectory', ...
          {'True Trajectory', noisy_rmse_str, baseline_rmse_str});
saveas(fig, fullfile(outputDir, 'Baseline_Trajectories.png'));

% Fine-Tuned Trajectory Plot
finetuned_rmse_str = sprintf('Estimated Trajectory, RMSE = %.4f', rmse_estimated_finetuned_mean);
noisy_rmse_str = sprintf('Noisy Measurement, RMSE = %.4f', rmse_estimated_finetuned_mean_noisy);
fig = figure('Visible', 'off');
plot(xTrue, yTrue, 'Color', true_colour, 'LineWidth', 2); hold on;
plot(xNoisy, yNoisy, 'Color', noisy_colour, 'LineWidth', 1.5);
plot(x_estimated_trajectory, y_estimated_trajectory, 'Color', estimated_colour, 'LineWidth', 1.5);
plot(x_estimated_trajectory, y_estimated_trajectory, 'x', 'Color', 'k', 'MarkerSize', 4, 'LineWidth', 1);
stylePlot('X Position', 'Y Position', 'Fine-tuned Kalman Filter Trajectory', ...
          {'True Trajectory', noisy_rmse_str, finetuned_rmse_str});
saveas(fig, fullfile(outputDir, 'Fine_Tuned_Trajectory.png'));

% Mean RMSE Comparison Bar Plot
fig = figure('Visible', 'off');
barData = [baseline_rmse_estimated_mean, rmse_estimated_finetuned_mean];
bar(barData, 0.4, 'FaceColor', [0.3 0.5 0.7]);
hold on;
errorbar(1:2, barData, ...
         [baseline_rmse_estimated_std, rmse_estimated_finetuned_std], ...
         'r.', 'LineWidth', 2);
xticks(1:2); xticklabels({'Baseline', 'Fine-tuned'});
ylabel('RMSE');
title('Mean RMSE Comparison');
legend('Mean RMSE', 'Standard Deviation', 'Location', 'northeast');
grid on;
print(fig, fullfile(outputDir, 'Mean_RMSE_Comparison_Bar_Graph.png'), '-dpng', '-r300');

% X and Y Error Comparison Bar Plot
fig = figure('Visible', 'off');
errors = [x_std_error_baseline, y_std_error_baseline; x_std_error, y_std_error];
bar(errors, 'grouped');
colormap([0.3 0.6 0.9; 0.9 0.5 0.2]);  % Blue and orange shades
set(gca, 'XTickLabel', {'Baseline', 'Fine-tuned'});
ylabel('Standard Deviation Error');
legend({'X Direction', 'Y Direction'}, 'Location', 'northeast');
title('X and Y Direction Error');
grid on;
print(fig, fullfile(outputDir, 'X_and_Y_Direction_Error_Bar_Plot.png'), '-dpng', '-r300');

% Plot Styling Function
function stylePlot(xlab, ylab, plotTitle, legendLabels)
    xlabel(xlab, 'FontSize', 12);
    ylabel(ylab, 'FontSize', 12);
    title(plotTitle, 'FontSize', 14, 'FontWeight', 'bold');
    legend(legendLabels, 'Location', 'Best', 'FontSize', 10);
    grid on;
    axis tight;
end
