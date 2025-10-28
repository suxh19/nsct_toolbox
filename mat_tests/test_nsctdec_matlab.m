% Test script for nsctdec (Nonsubsampled Contourlet Transform Decomposition)
% This script generates test data for validating the Python implementation

clear all;
close all;

% Add path to NSCT MATLAB implementation
addpath('../nsct_matlab');

fprintf('Testing nsctdec (Nonsubsampled Contourlet Transform Decomposition)...\n\n');

%% Test 1: Single pyramid level, 2 directional levels
fprintf('Test 1: Single pyramid level, 2 directional levels\n');
x1 = rand(64, 64);
levels1 = [2];
y1 = nsctdec(x1, levels1, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x1, 1), size(x1, 2));
fprintf('  Pyramid levels: %d\n', length(levels1));
fprintf('  Directional levels: [%s]\n', num2str(levels1));
fprintf('  Output subbands: %d\n', length(y1));
fprintf('  y{1} (lowpass): %dx%d\n', size(y1{1}, 1), size(y1{1}, 2));
fprintf('  y{2} (bandpass): %d directional subbands\n', length(y1{2}));
fprintf('    Subband sizes: ');
for k = 1:length(y1{2})
    fprintf('%dx%d ', size(y1{2}{k}, 1), size(y1{2}{k}, 2));
end
fprintf('\n\n');

%% Test 2: Two pyramid levels, [2, 3] directional levels
fprintf('Test 2: Two pyramid levels, [2, 3] directional levels\n');
x2 = rand(64, 64);
levels2 = [2, 3];
y2 = nsctdec(x2, levels2, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x2, 1), size(x2, 2));
fprintf('  Pyramid levels: %d\n', length(levels2));
fprintf('  Directional levels: [%s]\n', num2str(levels2));
fprintf('  Output subbands: %d\n', length(y2));
fprintf('  y{1} (lowpass): %dx%d\n', size(y2{1}, 1), size(y2{1}, 2));
fprintf('  y{2}: %d directional subbands\n', length(y2{2}));
fprintf('  y{3}: %d directional subbands\n', length(y2{3}));
fprintf('\n');

%% Test 3: Three pyramid levels, [2, 3, 4] directional levels
fprintf('Test 3: Three pyramid levels, [2, 3, 4] directional levels\n');
x3 = rand(128, 128);
levels3 = [2, 3, 4];
y3 = nsctdec(x3, levels3, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x3, 1), size(x3, 2));
fprintf('  Pyramid levels: %d\n', length(levels3));
fprintf('  Directional levels: [%s]\n', num2str(levels3));
fprintf('  Output subbands: %d\n', length(y3));
fprintf('  y{1} (lowpass): %dx%d\n', size(y3{1}, 1), size(y3{1}, 2));
fprintf('  y{2}: %d directional subbands\n', length(y3{2}));
fprintf('  y{3}: %d directional subbands\n', length(y3{3}));
fprintf('  y{4}: %d directional subbands\n', length(y3{4}));
fprintf('\n');

%% Test 4: Level 0 (no directional decomposition)
fprintf('Test 4: Level 0 (no directional decomposition)\n');
x4 = rand(64, 64);
levels4 = [0];
y4 = nsctdec(x4, levels4, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x4, 1), size(x4, 2));
fprintf('  Pyramid levels: %d\n', length(levels4));
fprintf('  Directional levels: [%s]\n', num2str(levels4));
fprintf('  Output subbands: %d\n', length(y4));
fprintf('  y{1} (lowpass): %dx%d\n', size(y4{1}, 1), size(y4{1}, 2));
fprintf('  y{2} (bandpass, no directionality): %dx%d\n', size(y4{2}, 1), size(y4{2}, 2));
fprintf('\n');

%% Test 5: Mixed levels [0, 2]
fprintf('Test 5: Mixed levels [0, 2]\n');
x5 = rand(64, 64);
levels5 = [0, 2];
y5 = nsctdec(x5, levels5, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x5, 1), size(x5, 2));
fprintf('  Pyramid levels: %d\n', length(levels5));
fprintf('  Directional levels: [%s]\n', num2str(levels5));
fprintf('  Output subbands: %d\n', length(y5));
fprintf('  y{1} (lowpass): %dx%d\n', size(y5{1}, 1), size(y5{1}, 2));
fprintf('  y{2}: Single bandpass (no directionality)\n');
fprintf('  y{3}: %d directional subbands\n', length(y5{3}));
fprintf('\n');

%% Test 6: dmaxflat7 filter
fprintf('Test 6: dmaxflat7 filter\n');
x6 = rand(64, 64);
levels6 = [2, 3];
y6 = nsctdec(x6, levels6, 'dmaxflat7', 'maxflat');
fprintf('  Input: %dx%d\n', size(x6, 1), size(x6, 2));
fprintf('  Pyramid levels: %d\n', length(levels6));
fprintf('  Directional levels: [%s]\n', num2str(levels6));
fprintf('  Directional filter: dmaxflat7\n');
fprintf('  Output subbands: %d\n', length(y6));
fprintf('  y{1} (lowpass): %dx%d\n', size(y6{1}, 1), size(y6{1}, 2));
fprintf('  y{2}: %d directional subbands\n', length(y6{2}));
fprintf('  y{3}: %d directional subbands\n', length(y6{3}));
fprintf('\n');

%% Test 7: Non-square image
fprintf('Test 7: Non-square image\n');
x7 = rand(64, 96);
levels7 = [2, 3];
y7 = nsctdec(x7, levels7, 'pkva', 'maxflat');
fprintf('  Input: %dx%d\n', size(x7, 1), size(x7, 2));
fprintf('  Pyramid levels: %d\n', length(levels7));
fprintf('  Directional levels: [%s]\n', num2str(levels7));
fprintf('  Output subbands: %d\n', length(y7));
fprintf('  y{1} (lowpass): %dx%d\n', size(y7{1}, 1), size(y7{1}, 2));
fprintf('  y{2}: %d directional subbands\n', length(y7{2}));
fprintf('  y{3}: %d directional subbands\n', length(y7{3}));
fprintf('\n');

%% Test 8: Constant image
fprintf('Test 8: Constant image\n');
x8 = ones(64, 64) * 5.0;
levels8 = [2];
y8 = nsctdec(x8, levels8, 'pkva', 'maxflat');
fprintf('  Input: %dx%d (constant value: 5.0)\n', size(x8, 1), size(x8, 2));
fprintf('  Pyramid levels: %d\n', length(levels8));
fprintf('  Directional levels: [%s]\n', num2str(levels8));
fprintf('  Lowpass mean: %.6f\n', mean(y8{1}(:)));
fprintf('  Bandpass subbands: %d\n', length(y8{2}));
for k = 1:length(y8{2})
    fprintf('    Subband %d mean: %.6e\n', k, mean(y8{2}{k}(:)));
end
fprintf('\n');

%% Save all test data
fprintf('Saving test data to mat file...\n');

% For test 1
y1_lowpass = y1{1};
y1_num_dir = length(y1{2});
for k = 1:y1_num_dir
    eval(sprintf('y1_band_%d = y1{2}{%d};', k, k));
end

% For test 2
y2_lowpass = y2{1};
y2_num_dir1 = length(y2{2});
y2_num_dir2 = length(y2{3});
for k = 1:y2_num_dir1
    eval(sprintf('y2_band1_%d = y2{2}{%d};', k, k));
end
for k = 1:y2_num_dir2
    eval(sprintf('y2_band2_%d = y2{3}{%d};', k, k));
end

% For test 3
y3_lowpass = y3{1};
y3_num_dir1 = length(y3{2});
y3_num_dir2 = length(y3{3});
y3_num_dir3 = length(y3{4});

% For test 4
y4_lowpass = y4{1};
y4_bandpass = y4{2};

% For test 5
y5_lowpass = y5{1};
y5_bandpass = y5{2};
y5_num_dir = length(y5{3});

% For test 6
y6_lowpass = y6{1};
y6_num_dir1 = length(y6{2});
y6_num_dir2 = length(y6{3});

% For test 7
y7_lowpass = y7{1};
y7_num_dir1 = length(y7{2});
y7_num_dir2 = length(y7{3});

% For test 8
y8_lowpass = y8{1};
y8_num_dir = length(y8{2});

save('../data/test_nsctdec_results.mat', ...
    'x1', 'levels1', 'y1_lowpass', 'y1_num_dir', 'y1_band_1', 'y1_band_2', 'y1_band_3', 'y1_band_4', ...
    'x2', 'levels2', 'y2_lowpass', 'y2_num_dir1', 'y2_num_dir2', 'y2_band1_1', 'y2_band1_2', 'y2_band1_3', 'y2_band1_4', ...
    'y2_band2_1', 'y2_band2_2', 'y2_band2_3', 'y2_band2_4', 'y2_band2_5', 'y2_band2_6', 'y2_band2_7', 'y2_band2_8', ...
    'x3', 'levels3', 'y3_lowpass', 'y3_num_dir1', 'y3_num_dir2', 'y3_num_dir3', ...
    'x4', 'levels4', 'y4_lowpass', 'y4_bandpass', ...
    'x5', 'levels5', 'y5_lowpass', 'y5_bandpass', 'y5_num_dir', ...
    'x6', 'levels6', 'y6_lowpass', 'y6_num_dir1', 'y6_num_dir2', ...
    'x7', 'levels7', 'y7_lowpass', 'y7_num_dir1', 'y7_num_dir2', ...
    'x8', 'levels8', 'y8_lowpass', 'y8_num_dir');

fprintf('Test data saved successfully!\n');
fprintf('\nAll tests completed!\n');
