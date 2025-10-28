% Test script for nsdfbrec (Nonsubsampled Directional Filter Bank Reconstruction)
% This script generates test data for validating the Python implementation

clear all;
close all;

% Add path to NSCT MATLAB implementation
addpath('../nsct_matlab');

fprintf('Testing nsdfbrec (Nonsubsampled DFB Reconstruction)...\n\n');

%% Test 1: Perfect reconstruction - Level 1
fprintf('Test 1: Perfect reconstruction - Level 1\n');
x1 = rand(64, 64);
clevels1 = 1;
y1 = nsdfbdec(x1, 'pkva', clevels1);
x1_rec = nsdfbrec(y1, 'pkva');
error1 = max(abs(x1(:) - x1_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x1, 1), size(x1, 2), clevels1);
fprintf('  Subbands: %d\n', length(y1));
fprintf('  Reconstruction error: %.2e\n', error1);
if error1 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 2: Perfect reconstruction - Level 2
fprintf('Test 2: Perfect reconstruction - Level 2\n');
x2 = rand(64, 64);
clevels2 = 2;
y2 = nsdfbdec(x2, 'pkva', clevels2);
x2_rec = nsdfbrec(y2, 'pkva');
error2 = max(abs(x2(:) - x2_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x2, 1), size(x2, 2), clevels2);
fprintf('  Subbands: %d\n', length(y2));
fprintf('  Reconstruction error: %.2e\n', error2);
if error2 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 3: Perfect reconstruction - Level 3
fprintf('Test 3: Perfect reconstruction - Level 3\n');
x3 = rand(128, 128);
clevels3 = 3;
y3 = nsdfbdec(x3, 'pkva', clevels3);
x3_rec = nsdfbrec(y3, 'pkva');
error3 = max(abs(x3(:) - x3_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x3, 1), size(x3, 2), clevels3);
fprintf('  Subbands: %d\n', length(y3));
fprintf('  Reconstruction error: %.2e\n', error3);
if error3 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 4: Perfect reconstruction - dmaxflat7 filter
fprintf('Test 4: Perfect reconstruction - dmaxflat7 filter\n');
x4 = rand(64, 64);
clevels4 = 2;
y4 = nsdfbdec(x4, 'dmaxflat7', clevels4);
x4_rec = nsdfbrec(y4, 'dmaxflat7');
error4 = max(abs(x4(:) - x4_rec(:)));
fprintf('  Input: %dx%d, Levels: %d, Filter: dmaxflat7\n', size(x4, 1), size(x4, 2), clevels4);
fprintf('  Subbands: %d\n', length(y4));
fprintf('  Reconstruction error: %.2e\n', error4);
if error4 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 5: Non-square image
fprintf('Test 5: Non-square image\n');
x5 = rand(64, 96);
clevels5 = 2;
y5 = nsdfbdec(x5, 'pkva', clevels5);
x5_rec = nsdfbrec(y5, 'pkva');
error5 = max(abs(x5(:) - x5_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x5, 1), size(x5, 2), clevels5);
fprintf('  Subbands: %d\n', length(y5));
fprintf('  Reconstruction error: %.2e\n', error5);
if error5 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 6: Level 0 (no decomposition)
fprintf('Test 6: Level 0 (no decomposition)\n');
x6 = rand(32, 32);
clevels6 = 0;
y6 = nsdfbdec(x6, 'pkva', clevels6);
x6_rec = nsdfbrec(y6, 'pkva');
error6 = max(abs(x6(:) - x6_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x6, 1), size(x6, 2), clevels6);
fprintf('  Subbands: %d\n', length(y6));
fprintf('  Reconstruction error: %.2e\n', error6);
if error6 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 7: Large image - Level 4
fprintf('Test 7: Large image - Level 4\n');
x7 = rand(256, 256);
clevels7 = 4;
y7 = nsdfbdec(x7, 'pkva', clevels7);
x7_rec = nsdfbrec(y7, 'pkva');
error7 = max(abs(x7(:) - x7_rec(:)));
fprintf('  Input: %dx%d, Levels: %d\n', size(x7, 1), size(x7, 2), clevels7);
fprintf('  Subbands: %d\n', length(y7));
fprintf('  Reconstruction error: %.2e\n', error7);
if error7 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Test 8: Constant image
fprintf('Test 8: Constant image\n');
x8 = ones(64, 64) * 5.0;
clevels8 = 2;
y8 = nsdfbdec(x8, 'pkva', clevels8);
x8_rec = nsdfbrec(y8, 'pkva');
error8 = max(abs(x8(:) - x8_rec(:)));
fprintf('  Input: %dx%d (constant value: 5.0), Levels: %d\n', size(x8, 1), size(x8, 2), clevels8);
fprintf('  Subbands: %d\n', length(y8));
fprintf('  Reconstruction error: %.2e\n', error8);
if error8 < 1e-10
    fprintf('  Perfect reconstruction: YES\n\n');
else
    fprintf('  Perfect reconstruction: NO\n\n');
end

%% Save all test data
fprintf('Saving test data to mat file...\n');
save('../data/test_nsdfbrec_results.mat', ...
    'x1', 'clevels1', 'x1_rec', 'error1', ...
    'x2', 'clevels2', 'x2_rec', 'error2', ...
    'x3', 'clevels3', 'x3_rec', 'error3', ...
    'x4', 'clevels4', 'x4_rec', 'error4', ...
    'x5', 'clevels5', 'x5_rec', 'error5', ...
    'x6', 'clevels6', 'x6_rec', 'error6', ...
    'x7', 'clevels7', 'x7_rec', 'error7', ...
    'x8', 'clevels8', 'x8_rec', 'error8');

fprintf('Test data saved successfully!\n');
fprintf('\nAll tests completed!\n');
