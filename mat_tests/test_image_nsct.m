% test_image_nsct.m
% MATLAB script for NSCT decomposition and reconstruction
% Purpose: Verify NSCT perfect reconstruction on test image
% Output: Saves only input and output images (not coefficients)

clear all;
close all;

fprintf('========================================\n');
fprintf('NSCT Image Test - MATLAB Version\n');
fprintf('========================================\n\n');

% Add NSCT MATLAB toolbox to path
addpath('../nsct_matlab');

%% Load Test Image
fprintf('Loading test image...\n');
img_path = '../test_image.jpg';
img = imread(img_path);

% Convert to grayscale if RGB
if size(img, 3) == 3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end

% Convert to double
img_input = double(img_gray);
fprintf('  Image size: %d x %d\n', size(img_input, 1), size(img_input, 2));
fprintf('  Value range: [%.2f, %.2f]\n\n', min(img_input(:)), max(img_input(:)));

%% NSCT Parameters
fprintf('NSCT Parameters:\n');
levels = [2, 3, 4];  % Decomposition levels
dfilt = 'dmaxflat7'; % Directional filter
pfilt = 'maxflat';   % Pyramidal filter

fprintf('  Levels: [%s]\n', num2str(levels));
fprintf('  Directional filter: %s\n', dfilt);
fprintf('  Pyramidal filter: %s\n\n', pfilt);

%% NSCT Decomposition
fprintf('Performing NSCT decomposition...\n');
tic;
coeffs = nsctdec(img_input, levels, dfilt, pfilt);
time_dec = toc;
fprintf('  Decomposition time: %.4f seconds\n\n', time_dec);

%% NSCT Reconstruction
fprintf('Performing NSCT reconstruction...\n');
tic;
img_recon = nsctrec(coeffs, dfilt, pfilt);
time_rec = toc;
fprintf('  Reconstruction time: %.4f seconds\n\n', time_rec);

%% Calculate Error Metrics
fprintf('Calculating error metrics...\n');
mse = mean((img_input(:) - img_recon(:)).^2);
max_val = max(img_input(:));
if mse > 0
    psnr_val = 10 * log10(max_val^2 / mse);
else
    psnr_val = Inf;
end
max_error = max(abs(img_input(:) - img_recon(:)));
nrmse = sqrt(mse) / (max(img_input(:)) - min(img_input(:)));

fprintf('  MSE: %.10e\n', mse);
fprintf('  PSNR: %.4f dB\n', psnr_val);
fprintf('  Max Error: %.10e\n', max_error);
fprintf('  NRMSE: %.10e\n\n', nrmse);

%% Save Results (only input and output)
fprintf('Saving results...\n');
results = struct();
results.input_image = img_input;
results.reconstructed_image = img_recon;
results.parameters = struct('levels', levels, 'dfilt', dfilt, 'pfilt', pfilt);
results.timing = struct('decomposition', time_dec, 'reconstruction', time_rec);
results.metrics = struct('mse', mse, 'psnr', psnr_val, 'max_error', max_error, 'nrmse', nrmse);
results.image_size = size(img_input);
results.test_date = datestr(now);

output_file = '../data/matlab_nsct_results.mat';
save(output_file, 'results');
fprintf('  Results saved to: %s\n', output_file);

% Check file size
file_info = dir(output_file);
fprintf('  File size: %.2f MB\n\n', file_info.bytes / 1024 / 1024);

%% Create Visualization
fprintf('Creating visualization...\n');
figure('Position', [100, 100, 1400, 400]);

% Original image
subplot(1, 4, 1);
imshow(img_input, []);
title('Input Image');
colorbar;

% Reconstructed image
subplot(1, 4, 2);
imshow(img_recon, []);
title('Reconstructed Image');
colorbar;

% Error image
subplot(1, 4, 3);
error_img = abs(img_input - img_recon);
imshow(error_img, []);
title(sprintf('Absolute Error (Max: %.2e)', max_error));
colorbar;

% Error histogram
subplot(1, 4, 4);
histogram(error_img(:), 50);
title('Error Distribution');
xlabel('Error Magnitude');
ylabel('Frequency');
grid on;

% Save figure
fig_file = '../data/matlab_nsct_comparison.png';
saveas(gcf, fig_file);
fprintf('  Figure saved to: %s\n\n', fig_file);

%% Summary
fprintf('========================================\n');
fprintf('Test Summary\n');
fprintf('========================================\n');
fprintf('Image: %s (%dx%d)\n', img_path, size(img_input, 1), size(img_input, 2));
fprintf('Levels: [%s]\n', num2str(levels));
fprintf('Time: %.4fs (dec) + %.4fs (rec) = %.4fs\n', time_dec, time_rec, time_dec + time_rec);
fprintf('MSE: %.10e\n', mse);
fprintf('PSNR: %.4f dB\n', psnr_val);
fprintf('Max Error: %.10e\n', max_error);
fprintf('\n✓ MATLAB test completed!\n');
fprintf('========================================\n');
