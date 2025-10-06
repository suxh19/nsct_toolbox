%% Test nsctrec() - NSCT Reconstruction
% Tests complete decomposition-reconstruction cycle for perfect reconstruction

% Add MATLAB functions to path
addpath('../nsct_matlab');

% Initialize results structure
results = struct();
test_count = 0;

%% Test 1: Single pyramid level [2] with pkva/maxflat
test_count = test_count + 1;
fprintf('Test %d: Single pyramid level [2], pkva/maxflat\n', test_count);
x_orig = rand(64, 64);
levels = [2];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test1.x_orig = x_orig;
results.test1.levels = levels;
results.test1.dfilt = 'pkva';
results.test1.pfilt = 'maxflat';
results.test1.y = y;
results.test1.x_rec = x_rec;
results.test1.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test1.error);

%% Test 2: Two pyramid levels [2,3] with pkva/maxflat
test_count = test_count + 1;
fprintf('Test %d: Two pyramid levels [2,3], pkva/maxflat\n', test_count);
x_orig = rand(64, 64);
levels = [2, 3];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test2.x_orig = x_orig;
results.test2.levels = levels;
results.test2.dfilt = 'pkva';
results.test2.pfilt = 'maxflat';
results.test2.y = y;
results.test2.x_rec = x_rec;
results.test2.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test2.error);

%% Test 3: Three pyramid levels [2,3,4] with pkva/maxflat
test_count = test_count + 1;
fprintf('Test %d: Three pyramid levels [2,3,4], pkva/maxflat\n', test_count);
x_orig = rand(128, 128);
levels = [2, 3, 4];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test3.x_orig = x_orig;
results.test3.levels = levels;
results.test3.dfilt = 'pkva';
results.test3.pfilt = 'maxflat';
results.test3.y = y;
results.test3.x_rec = x_rec;
results.test3.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test3.error);

%% Test 4: Level 0 (no directional decomposition)
test_count = test_count + 1;
fprintf('Test %d: Level 0 (no directional), pkva/maxflat\n', test_count);
x_orig = rand(64, 64);
levels = [0];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test4.x_orig = x_orig;
results.test4.levels = levels;
results.test4.dfilt = 'pkva';
results.test4.pfilt = 'maxflat';
results.test4.y = y;
results.test4.x_rec = x_rec;
results.test4.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test4.error);

%% Test 5: Mixed levels [0,2]
test_count = test_count + 1;
fprintf('Test %d: Mixed levels [0,2], pkva/maxflat\n', test_count);
x_orig = rand(64, 64);
levels = [0, 2];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test5.x_orig = x_orig;
results.test5.levels = levels;
results.test5.dfilt = 'pkva';
results.test5.pfilt = 'maxflat';
results.test5.y = y;
results.test5.x_rec = x_rec;
results.test5.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test5.error);

%% Test 6: dmaxflat7 filter
test_count = test_count + 1;
fprintf('Test %d: dmaxflat7 filter with [2,3]\n', test_count);
x_orig = rand(64, 64);
levels = [2, 3];
y = nsctdec(x_orig, levels, 'dmaxflat7', 'maxflat');
x_rec = nsctrec(y, 'dmaxflat7', 'maxflat');

results.test6.x_orig = x_orig;
results.test6.levels = levels;
results.test6.dfilt = 'dmaxflat7';
results.test6.pfilt = 'maxflat';
results.test6.y = y;
results.test6.x_rec = x_rec;
results.test6.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test6.error);

%% Test 7: Non-square image
test_count = test_count + 1;
fprintf('Test %d: Non-square image (64x96), [2,3]\n', test_count);
x_orig = rand(64, 96);
levels = [2, 3];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test7.x_orig = x_orig;
results.test7.levels = levels;
results.test7.dfilt = 'pkva';
results.test7.pfilt = 'maxflat';
results.test7.y = y;
results.test7.x_rec = x_rec;
results.test7.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test7.error);

%% Test 8: Constant image (special case)
test_count = test_count + 1;
fprintf('Test %d: Constant image (value=5.0), [2]\n', test_count);
x_orig = 5.0 * ones(64, 64);
levels = [2];
y = nsctdec(x_orig, levels, 'pkva', 'maxflat');
x_rec = nsctrec(y, 'pkva', 'maxflat');

results.test8.x_orig = x_orig;
results.test8.levels = levels;
results.test8.dfilt = 'pkva';
results.test8.pfilt = 'maxflat';
results.test8.y = y;
results.test8.x_rec = x_rec;
results.test8.error = max(abs(x_orig(:) - x_rec(:)));
fprintf('  Reconstruction error: %.15e\n', results.test8.error);

%% Save all results
save('../data/test_nsctrec_results.mat', 'results');
fprintf('\nAll tests completed. Results saved to test_nsctrec_results.mat\n');
fprintf('Total tests: %d\n', test_count);
