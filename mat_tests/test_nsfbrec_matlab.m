% Test script for nsfbrec function
% Tests nonsubsampled filter bank reconstruction
% Results are saved to test_nsfbrec_results.mat for Python comparison

fprintf('=== Testing nsfbrec function ===\n\n');

% Get atrous filters
[h0, h1, g0, g1] = atrousfilters('maxflat');

% Test 1: Perfect reconstruction at level 0 (32x32 image)
fprintf('Test 1: Perfect reconstruction at level 0 (32x32)\n');
rng(42);
x1 = rand(32, 32);
lev1 = 0;
[y0_1, y1_1] = nsfbdec(x1, h0, h1, lev1);
x_rec_1 = nsfbrec(y0_1, y1_1, g0, g1, lev1);
err1 = norm(x1(:) - x_rec_1(:)) / norm(x1(:));
fprintf('Original size: %dx%d\n', size(x1, 1), size(x1, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_1, 1), size(x_rec_1, 2));
fprintf('Relative error: %.6e\n', err1);

% Test 2: Perfect reconstruction at level 1 (64x64 image)
fprintf('\nTest 2: Perfect reconstruction at level 1 (64x64)\n');
x2 = rand(64, 64);
lev2 = 1;
[y0_2, y1_2] = nsfbdec(x2, h0, h1, lev2);
x_rec_2 = nsfbrec(y0_2, y1_2, g0, g1, lev2);
err2 = norm(x2(:) - x_rec_2(:)) / norm(x2(:));
fprintf('Original size: %dx%d\n', size(x2, 1), size(x2, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_2, 1), size(x_rec_2, 2));
fprintf('Relative error: %.6e\n', err2);

% Test 3: Perfect reconstruction at level 2 (128x128 image)
fprintf('\nTest 3: Perfect reconstruction at level 2 (128x128)\n');
x3 = rand(128, 128);
lev3 = 2;
[y0_3, y1_3] = nsfbdec(x3, h0, h1, lev3);
x_rec_3 = nsfbrec(y0_3, y1_3, g0, g1, lev3);
err3 = norm(x3(:) - x_rec_3(:)) / norm(x3(:));
fprintf('Original size: %dx%d\n', size(x3, 1), size(x3, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_3, 1), size(x_rec_3, 2));
fprintf('Relative error: %.6e\n', err3);

% Test 4: Perfect reconstruction at level 3 (256x256 image)
fprintf('\nTest 4: Perfect reconstruction at level 3 (256x256)\n');
x4 = rand(256, 256);
lev4 = 3;
[y0_4, y1_4] = nsfbdec(x4, h0, h1, lev4);
x_rec_4 = nsfbrec(y0_4, y1_4, g0, g1, lev4);
err4 = norm(x4(:) - x_rec_4(:)) / norm(x4(:));
fprintf('Original size: %dx%d\n', size(x4, 1), size(x4, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_4, 1), size(x_rec_4, 2));
fprintf('Relative error: %.6e\n', err4);

% Test 5: Non-square image at level 0 (32x48)
fprintf('\nTest 5: Non-square reconstruction at level 0 (32x48)\n');
x5 = rand(32, 48);
lev5 = 0;
[y0_5, y1_5] = nsfbdec(x5, h0, h1, lev5);
x_rec_5 = nsfbrec(y0_5, y1_5, g0, g1, lev5);
err5 = norm(x5(:) - x_rec_5(:)) / norm(x5(:));
fprintf('Original size: %dx%d\n', size(x5, 1), size(x5, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_5, 1), size(x_rec_5, 2));
fprintf('Relative error: %.6e\n', err5);

% Test 6: Non-square image at level 1 (64x96)
fprintf('\nTest 6: Non-square reconstruction at level 1 (64x96)\n');
x6 = rand(64, 96);
lev6 = 1;
[y0_6, y1_6] = nsfbdec(x6, h0, h1, lev6);
x_rec_6 = nsfbrec(y0_6, y1_6, g0, g1, lev6);
err6 = norm(x6(:) - x_rec_6(:)) / norm(x6(:));
fprintf('Original size: %dx%d\n', size(x6, 1), size(x6, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_6, 1), size(x_rec_6, 2));
fprintf('Relative error: %.6e\n', err6);

% Test 7: Using 9-7 filters at level 0
fprintf('\nTest 7: 9-7 filters at level 0 (32x32)\n');
[h0_97, h1_97, g0_97, g1_97] = atrousfilters('9-7');
x7 = rand(32, 32);
lev7 = 0;
[y0_7, y1_7] = nsfbdec(x7, h0_97, h1_97, lev7);
x_rec_7 = nsfbrec(y0_7, y1_7, g0_97, g1_97, lev7);
err7 = norm(x7(:) - x_rec_7(:)) / norm(x7(:));
fprintf('Original size: %dx%d\n', size(x7, 1), size(x7, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_7, 1), size(x_rec_7, 2));
fprintf('Relative error: %.6e\n', err7);

% Test 8: Using 9-7 filters at level 1
fprintf('\nTest 8: 9-7 filters at level 1 (64x64)\n');
x8 = rand(64, 64);
lev8 = 1;
[y0_8, y1_8] = nsfbdec(x8, h0_97, h1_97, lev8);
x_rec_8 = nsfbrec(y0_8, y1_8, g0_97, g1_97, lev8);
err8 = norm(x8(:) - x_rec_8(:)) / norm(x8(:));
fprintf('Original size: %dx%d\n', size(x8, 1), size(x8, 2));
fprintf('Reconstructed size: %dx%d\n', size(x_rec_8, 1), size(x_rec_8, 2));
fprintf('Relative error: %.6e\n', err8);

% Test 9: Direct reconstruction test (without decomposition first)
fprintf('\nTest 9: Direct reconstruction test (32x32)\n');
x9_orig = rand(32, 32);
[y0_9, y1_9] = nsfbdec(x9_orig, h0, h1, 0);
% Store decomposition outputs for Python to reconstruct
x9 = x9_orig;  % Keep original for comparison
lev9 = 0;
x_rec_9 = nsfbrec(y0_9, y1_9, g0, g1, lev9);
err9 = norm(x9(:) - x_rec_9(:)) / norm(x9(:));
fprintf('Relative error: %.6e\n', err9);

% Test 10: Multi-level cascade (decompose at level 0, reconstruct at level 0)
fprintf('\nTest 10: Multi-level consistency check (32x32)\n');
x10 = rand(32, 32);
lev10 = 0;
[y0_10, y1_10] = nsfbdec(x10, h0, h1, lev10);
x_rec_10 = nsfbrec(y0_10, y1_10, g0, g1, lev10);
err10 = norm(x10(:) - x_rec_10(:)) / norm(x10(:));
fprintf('Relative error: %.6e\n', err10);

% Save all results
fprintf('\nSaving results to test_nsfbrec_results.mat...\n');
save('data/test_nsfbrec_results.mat', ...
    'h0', 'h1', 'g0', 'g1', ...
    'h0_97', 'h1_97', 'g0_97', 'g1_97', ...
    'x1', 'y0_1', 'y1_1', 'lev1', 'x_rec_1', 'err1', ...
    'x2', 'y0_2', 'y1_2', 'lev2', 'x_rec_2', 'err2', ...
    'x3', 'y0_3', 'y1_3', 'lev3', 'x_rec_3', 'err3', ...
    'x4', 'y0_4', 'y1_4', 'lev4', 'x_rec_4', 'err4', ...
    'x5', 'y0_5', 'y1_5', 'lev5', 'x_rec_5', 'err5', ...
    'x6', 'y0_6', 'y1_6', 'lev6', 'x_rec_6', 'err6', ...
    'x7', 'y0_7', 'y1_7', 'lev7', 'x_rec_7', 'err7', ...
    'x8', 'y0_8', 'y1_8', 'lev8', 'x_rec_8', 'err8', ...
    'x9', 'y0_9', 'y1_9', 'lev9', 'x_rec_9', 'err9', ...
    'x10', 'y0_10', 'y1_10', 'lev10', 'x_rec_10', 'err10');

fprintf('Done! All tests completed.\n');
