% Test script for nsfbdec function
% Tests nonsubsampled filter bank decomposition
% Results are saved to test_nsfbdec_results.mat for Python comparison

fprintf('=== Testing nsfbdec function ===\n\n');

% Get atrous filters
[h0, h1, g0, g1] = atrousfilters('maxflat');

% Test 1: Basic decomposition at level 0 (first level)
fprintf('Test 1: Level 0 decomposition (32x32 image)\n');
rng(42);  % Set seed for reproducibility
x1 = rand(32, 32);  % Need larger image for 19x19 filter
lev1 = 0;
[y0_1, y1_1] = nsfbdec(x1, h0, h1, lev1);
fprintf('Input size: %dx%d\n', size(x1, 1), size(x1, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_1, 1), size(y0_1, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_1, 1), size(y1_1, 2));

% Test 2: Level 1 decomposition
fprintf('\nTest 2: Level 1 decomposition (64x64 image)\n');
x2 = rand(64, 64);  % Larger for upsampled filters
lev2 = 1;
[y0_2, y1_2] = nsfbdec(x2, h0, h1, lev2);
fprintf('Input size: %dx%d\n', size(x2, 1), size(x2, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_2, 1), size(y0_2, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_2, 1), size(y1_2, 2));

% Test 3: Level 2 decomposition
fprintf('\nTest 3: Level 2 decomposition (128x128 image)\n');
x3 = rand(128, 128);  % Even larger for level 2
lev3 = 2;
[y0_3, y1_3] = nsfbdec(x3, h0, h1, lev3);
fprintf('Input size: %dx%d\n', size(x3, 1), size(x3, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_3, 1), size(y0_3, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_3, 1), size(y1_3, 2));

% Test 4: Level 3 decomposition
fprintf('\nTest 4: Level 3 decomposition (256x256 image)\n');
x4 = rand(256, 256);  % Very large for level 3
lev4 = 3;
[y0_4, y1_4] = nsfbdec(x4, h0, h1, lev4);
fprintf('Input size: %dx%d\n', size(x4, 1), size(x4, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_4, 1), size(y0_4, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_4, 1), size(y1_4, 2));

% Test 5: Non-square image at level 0
fprintf('\nTest 5: Non-square image (32x48) at level 0\n');
x5 = rand(32, 48);  % Larger non-square
lev5 = 0;
[y0_5, y1_5] = nsfbdec(x5, h0, h1, lev5);
fprintf('Input size: %dx%d\n', size(x5, 1), size(x5, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_5, 1), size(y0_5, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_5, 1), size(y1_5, 2));

% Test 6: Non-square image at level 1
fprintf('\nTest 6: Non-square image (64x96) at level 1\n');
x6 = rand(64, 96);  % Larger for level 1
lev6 = 1;
[y0_6, y1_6] = nsfbdec(x6, h0, h1, lev6);
fprintf('Input size: %dx%d\n', size(x6, 1), size(x6, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_6, 1), size(y0_6, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_6, 1), size(y1_6, 2));

% Test 7: Different filter - try '9-7'
fprintf('\nTest 7: Using 9-7 filters at level 0\n');
[h0_97, h1_97, g0_97, g1_97] = atrousfilters('9-7');
x7 = rand(32, 32);  % Larger for safety
lev7 = 0;
[y0_7, y1_7] = nsfbdec(x7, h0_97, h1_97, lev7);
fprintf('Input size: %dx%d\n', size(x7, 1), size(x7, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_7, 1), size(y0_7, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_7, 1), size(y1_7, 2));

% Test 8: Different filter at level 1
fprintf('\nTest 8: Using 9-7 filters at level 1\n');
x8 = rand(64, 64);  % Larger for level 1
lev8 = 1;
[y0_8, y1_8] = nsfbdec(x8, h0_97, h1_97, lev8);
fprintf('Input size: %dx%d\n', size(x8, 1), size(x8, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_8, 1), size(y0_8, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_8, 1), size(y1_8, 2));

% Test 9: Reasonable size image at level 0
fprintf('\nTest 9: Fixed pattern image (32x32) at level 0\n');
x9 = repmat([1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16], 8, 8);
lev9 = 0;
[y0_9, y1_9] = nsfbdec(x9, h0, h1, lev9);
fprintf('Input size: %dx%d\n', size(x9, 1), size(x9, 2));
fprintf('Output y0 size: %dx%d\n', size(y0_9, 1), size(y0_9, 2));
fprintf('Output y1 size: %dx%d\n', size(y1_9, 1), size(y1_9, 2));

% Test 10: Energy conservation check
fprintf('\nTest 10: Energy conservation check\n');
x10 = rand(64, 64);
lev10 = 1;
[y0_10, y1_10] = nsfbdec(x10, h0, h1, lev10);
energy_in = sum(x10(:).^2);
energy_out = sum(y0_10(:).^2) + sum(y1_10(:).^2);
fprintf('Input energy: %.6f\n', energy_in);
fprintf('Output energy (y0+y1): %.6f\n', energy_out);
fprintf('Energy ratio: %.6f\n', energy_out / energy_in);

% Save all results
fprintf('\nSaving results to test_nsfbdec_results.mat...\n');
save('data/test_nsfbdec_results.mat', ...
    'h0', 'h1', 'g0', 'g1', ...
    'h0_97', 'h1_97', 'g0_97', 'g1_97', ...
    'x1', 'lev1', 'y0_1', 'y1_1', ...
    'x2', 'lev2', 'y0_2', 'y1_2', ...
    'x3', 'lev3', 'y0_3', 'y1_3', ...
    'x4', 'lev4', 'y0_4', 'y1_4', ...
    'x5', 'lev5', 'y0_5', 'y1_5', ...
    'x6', 'lev6', 'y0_6', 'y1_6', ...
    'x7', 'lev7', 'y0_7', 'y1_7', ...
    'x8', 'lev8', 'y0_8', 'y1_8', ...
    'x9', 'lev9', 'y0_9', 'y1_9', ...
    'x10', 'lev10', 'y0_10', 'y1_10', ...
    'energy_in', 'energy_out');

fprintf('Done! All tests completed.\n');
