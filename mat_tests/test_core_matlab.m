% test_core_matlab.m
% MATLAB test script for core.py functions
% Tests: nssfbdec, nssfbrec
% Generates reference data in test_core_results.mat

clear all;
close all;
rng(42);  % Set random seed for reproducibility

fprintf('Starting MATLAB core tests...\n');

% Initialize test results structure
results = struct();

%% ============================================================
%% Test 1: nssfbdec - basic test without mup (using efilter2)
%% ============================================================
fprintf('Test 1: nssfbdec basic (no mup)...\n');

x1 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');

[y1_1, y2_1] = nssfbdec(x1, h0, h1);

results.test1_input = x1;
results.test1_f1 = h0;
results.test1_f2 = h1;
results.test1_y1 = y1_1;
results.test1_y2 = y2_1;

%% ============================================================
%% Test 2: nssfbdec - with identity mup (mup = 1)
%% ============================================================
fprintf('Test 2: nssfbdec with mup=1...\n');

x2 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
mup2 = 1;

[y1_2, y2_2] = nssfbdec(x2, h0, h1, mup2);

results.test2_input = x2;
results.test2_f1 = h0;
results.test2_f2 = h1;
results.test2_mup = mup2;
results.test2_y1 = y1_2;
results.test2_y2 = y2_2;

%% ============================================================
%% Test 3: nssfbdec - with separable mup (scalar)
%% ============================================================
fprintf('Test 3: nssfbdec with separable mup=2...\n');

x3 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
mup3 = 2;

[y1_3, y2_3] = nssfbdec(x3, h0, h1, mup3);

results.test3_input = x3;
results.test3_f1 = h0;
results.test3_f2 = h1;
results.test3_mup = mup3;
results.test3_y1 = y1_3;
results.test3_y2 = y2_3;

%% ============================================================
%% Test 4: nssfbdec - with quincunx mup
%% ============================================================
fprintf('Test 4: nssfbdec with quincunx mup...\n');

x4 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
mup4 = [1, 1; -1, 1];

[y1_4, y2_4] = nssfbdec(x4, h0, h1, mup4);

results.test4_input = x4;
results.test4_f1 = h0;
results.test4_f2 = h1;
results.test4_mup = mup4;
results.test4_y1 = y1_4;
results.test4_y2 = y2_4;

%% ============================================================
%% Test 5: nssfbdec - with different filter (pyr)
%% ============================================================
fprintf('Test 5: nssfbdec with pyr filters and quincunx mup...\n');

x5 = rand(10, 10);
[h0, h1] = atrousfilters('pyr');
mup5 = [1, 1; -1, 1];

[y1_5, y2_5] = nssfbdec(x5, h0, h1, mup5);

results.test5_input = x5;
results.test5_f1 = h0;
results.test5_f2 = h1;
results.test5_mup = mup5;
results.test5_y1 = y1_5;
results.test5_y2 = y2_5;

%% ============================================================
%% Test 6: nssfbrec - basic test without mup
%% ============================================================
fprintf('Test 6: nssfbrec basic (no mup)...\n');

x1_6 = rand(8, 8);
x2_6 = rand(8, 8);
[g0, g1] = dfilters('pkva', 'r');

y6 = nssfbrec(x1_6, x2_6, g0, g1);

results.test6_input1 = x1_6;
results.test6_input2 = x2_6;
results.test6_f1 = g0;
results.test6_f2 = g1;
results.test6_output = y6;

%% ============================================================
%% Test 7: nssfbrec - with identity mup (mup = 1)
%% ============================================================
fprintf('Test 7: nssfbrec with mup=1...\n');

x1_7 = rand(8, 8);
x2_7 = rand(8, 8);
[g0, g1] = dfilters('pkva', 'r');
mup7 = 1;

y7 = nssfbrec(x1_7, x2_7, g0, g1, mup7);

results.test7_input1 = x1_7;
results.test7_input2 = x2_7;
results.test7_f1 = g0;
results.test7_f2 = g1;
results.test7_mup = mup7;
results.test7_output = y7;

%% ============================================================
%% Test 8: nssfbrec - with separable mup
%% ============================================================
fprintf('Test 8: nssfbrec with separable mup=2...\n');

x1_8 = rand(8, 8);
x2_8 = rand(8, 8);
[g0, g1] = dfilters('pkva', 'r');
mup8 = 2;

y8 = nssfbrec(x1_8, x2_8, g0, g1, mup8);

results.test8_input1 = x1_8;
results.test8_input2 = x2_8;
results.test8_f1 = g0;
results.test8_f2 = g1;
results.test8_mup = mup8;
results.test8_output = y8;

%% ============================================================
%% Test 9: nssfbrec - with quincunx mup
%% ============================================================
fprintf('Test 9: nssfbrec with quincunx mup...\n');

x1_9 = rand(8, 8);
x2_9 = rand(8, 8);
[g0, g1] = dfilters('pkva', 'r');
mup9 = [1, 1; -1, 1];

y9 = nssfbrec(x1_9, x2_9, g0, g1, mup9);

results.test9_input1 = x1_9;
results.test9_input2 = x2_9;
results.test9_f1 = g0;
results.test9_f2 = g1;
results.test9_mup = mup9;
results.test9_output = y9;

%% ============================================================
%% Test 10: Perfect reconstruction test without mup
%% ============================================================
fprintf('Test 10: Perfect reconstruction (no mup)...\n');

x10 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
[g0, g1] = dfilters('pkva', 'r');

[y1_10, y2_10] = nssfbdec(x10, h0, h1);
recon10 = nssfbrec(y1_10, y2_10, g0, g1);

results.test10_input = x10;
results.test10_h0 = h0;
results.test10_h1 = h1;
results.test10_g0 = g0;
results.test10_g1 = g1;
results.test10_y1 = y1_10;
results.test10_y2 = y2_10;
results.test10_recon = recon10;
results.test10_mse = mean((x10(:) - recon10(:)).^2);

fprintf('  MSE: %.15e\n', results.test10_mse);

%% ============================================================
%% Test 11: Perfect reconstruction test with separable mup
%% ============================================================
fprintf('Test 11: Perfect reconstruction with separable mup=2...\n');

x11 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
[g0, g1] = dfilters('pkva', 'r');
mup11 = 2;

[y1_11, y2_11] = nssfbdec(x11, h0, h1, mup11);
recon11 = nssfbrec(y1_11, y2_11, g0, g1, mup11);

results.test11_input = x11;
results.test11_h0 = h0;
results.test11_h1 = h1;
results.test11_g0 = g0;
results.test11_g1 = g1;
results.test11_mup = mup11;
results.test11_y1 = y1_11;
results.test11_y2 = y2_11;
results.test11_recon = recon11;
results.test11_mse = mean((x11(:) - recon11(:)).^2);

fprintf('  MSE: %.15e\n', results.test11_mse);

%% ============================================================
%% Test 12: Perfect reconstruction test with quincunx mup
%% ============================================================
fprintf('Test 12: Perfect reconstruction with quincunx mup...\n');

x12 = rand(8, 8);
[h0, h1] = dfilters('pkva', 'd');
[g0, g1] = dfilters('pkva', 'r');
mup12 = [1, 1; -1, 1];

[y1_12, y2_12] = nssfbdec(x12, h0, h1, mup12);
recon12 = nssfbrec(y1_12, y2_12, g0, g1, mup12);

results.test12_input = x12;
results.test12_h0 = h0;
results.test12_h1 = h1;
results.test12_g0 = g0;
results.test12_g1 = g1;
results.test12_mup = mup12;
results.test12_y1 = y1_12;
results.test12_y2 = y2_12;
results.test12_recon = recon12;
results.test12_mse = mean((x12(:) - recon12(:)).^2);

fprintf('  MSE: %.15e\n', results.test12_mse);

%% Save results
save('test_core_results.mat', '-struct', 'results');
fprintf('MATLAB core tests completed. Results saved to test_core_results.mat\n');
