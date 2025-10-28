% test_edge_cases_matlab.m
% Extended MATLAB test script for edge cases and boundary conditions
% Tests edge cases for all core functions
% Generates reference data in test_edge_cases_results.mat

clear all;
close all;
rng(42);  % Set random seed for reproducibility

fprintf('=== Starting MATLAB Edge Cases Tests ===\n\n');

% Initialize test results structure
results = struct();

%% ============================================================
%% Section 1: extend2 Edge Cases
%% ============================================================
fprintf('Section 1: extend2 Edge Cases\n');
fprintf('-------------------------------\n');

% Test 1.1: Small matrix (2x2) with periodic extension
fprintf('Test 1.1: extend2 - Small 2x2 matrix\n');
test1_1_input = [1, 2; 3, 4];
test1_1_output = extend2(test1_1_input, 1, 1, 1, 1, 'per');
results.edge_extend2_small.input = test1_1_input;
results.edge_extend2_small.output = test1_1_output;
fprintf('  Input: 2x2, Output: %dx%d\n', size(test1_1_output));

% Test 1.2: Single element matrix
fprintf('Test 1.2: extend2 - Single element matrix\n');
test1_2_input = 5;
test1_2_output = extend2(test1_2_input, 2, 2, 2, 2, 'per');
results.edge_extend2_single.input = test1_2_input;
results.edge_extend2_single.output = test1_2_output;
fprintf('  Input: 1x1, Output: %dx%d\n', size(test1_2_output));

% Test 1.3: Non-square matrix
fprintf('Test 1.3: extend2 - Non-square matrix (3x5)\n');
test1_3_input = reshape(1:15, 3, 5);
test1_3_output = extend2(test1_3_input, 2, 2, 3, 3, 'per');
results.edge_extend2_nonsquare.input = test1_3_input;
results.edge_extend2_nonsquare.output = test1_3_output;
fprintf('  Input: 3x5, Output: %dx%d\n', size(test1_3_output));

% Test 1.4: Large extension relative to input
fprintf('Test 1.4: extend2 - Large extension (4x4 input, 10 pixel extension)\n');
test1_4_input = reshape(1:16, 4, 4);
test1_4_output = extend2(test1_4_input, 10, 10, 10, 10, 'per');
results.edge_extend2_large_ext.input = test1_4_input;
results.edge_extend2_large_ext.output = test1_4_output;
fprintf('  Input: 4x4, Output: %dx%d\n', size(test1_4_output));

% Test 1.5: Zero extension
fprintf('Test 1.5: extend2 - Zero extension\n');
test1_5_input = reshape(1:9, 3, 3);
test1_5_output = extend2(test1_5_input, 0, 0, 0, 0, 'per');
results.edge_extend2_zero_ext.input = test1_5_input;
results.edge_extend2_zero_ext.output = test1_5_output;
fprintf('  Input: 3x3, Output: %dx%d\n', size(test1_5_output));

%% ============================================================
%% Section 2: upsample2df Edge Cases
%% ============================================================
fprintf('\nSection 2: upsample2df Edge Cases\n');
fprintf('-----------------------------------\n');

% Test 2.1: Zero matrix
fprintf('Test 2.1: upsample2df - Zero matrix\n');
test2_1_input = zeros(3, 3);
test2_1_output = upsample2df(test2_1_input, 1);
results.edge_upsample2df_zero.input = test2_1_input;
results.edge_upsample2df_zero.output = test2_1_output;
fprintf('  Input: 3x3 zeros, Output: %dx%d\n', size(test2_1_output));

% Test 2.2: Single element
fprintf('Test 2.2: upsample2df - Single element\n');
test2_2_input = 3.14;
test2_2_output = upsample2df(test2_2_input, 2);
results.edge_upsample2df_single.input = test2_2_input;
results.edge_upsample2df_single.output = test2_2_output;
fprintf('  Input: 1x1, Output: %dx%d\n', size(test2_2_output));

% Test 2.3: High power
fprintf('Test 2.3: upsample2df - High power (power=3)\n');
test2_3_input = [1, 2; 3, 4];
test2_3_output = upsample2df(test2_3_input, 3);
results.edge_upsample2df_high_power.input = test2_3_input;
results.edge_upsample2df_high_power.output = test2_3_output;
fprintf('  Input: 2x2, Power: 3, Output: %dx%d\n', size(test2_3_output));

%% ============================================================
%% Section 3: modulate2 Edge Cases
%% ============================================================
fprintf('\nSection 3: modulate2 Edge Cases\n');
fprintf('--------------------------------\n');

% Test 3.1: Single row
fprintf('Test 3.1: modulate2 - Single row matrix\n');
test3_1_input = [1, 2, 3, 4, 5];
test3_1_output = modulate2(test3_1_input, 'c', [0, 0]);
results.edge_modulate2_single_row.input = test3_1_input;
results.edge_modulate2_single_row.output = test3_1_output;
fprintf('  Input: 1x5, Output: %dx%d\n', size(test3_1_output));

% Test 3.2: Single column
fprintf('Test 3.2: modulate2 - Single column matrix\n');
test3_2_input = [1; 2; 3; 4];
test3_2_output = modulate2(test3_2_input, 'r', [0, 0]);
results.edge_modulate2_single_col.input = test3_2_input;
results.edge_modulate2_single_col.output = test3_2_output;
fprintf('  Input: 4x1, Output: %dx%d\n', size(test3_2_output));

% Test 3.3: Negative center offset
fprintf('Test 3.3: modulate2 - Negative center offset\n');
test3_3_input = ones(4, 4);
test3_3_output = modulate2(test3_3_input, 'b', [-2, -2]);
results.edge_modulate2_neg_center.input = test3_3_input;
results.edge_modulate2_neg_center.output = test3_3_output;
results.edge_modulate2_neg_center.center = [-2, -2];
fprintf('  Input: 4x4, Center: [-2, -2], Output: %dx%d\n', size(test3_3_output));

% Test 3.4: Large center offset
fprintf('Test 3.4: modulate2 - Large center offset\n');
test3_4_input = ones(3, 3);
test3_4_output = modulate2(test3_4_input, 'b', [10, 10]);
results.edge_modulate2_large_center.input = test3_4_input;
results.edge_modulate2_large_center.output = test3_4_output;
results.edge_modulate2_large_center.center = [10, 10];
fprintf('  Input: 3x3, Center: [10, 10], Output: %dx%d\n', size(test3_4_output));

%% ============================================================
%% Section 4: qupz Edge Cases
%% ============================================================
fprintf('\nSection 4: qupz Edge Cases\n');
fprintf('---------------------------\n');

% Test 4.1: Single element
fprintf('Test 4.1: qupz - Single element, type 1\n');
test4_1_input = 7;
test4_1_output = qupz(test4_1_input, 1);
results.edge_qupz_single_t1.input = test4_1_input;
results.edge_qupz_single_t1.output = test4_1_output;
fprintf('  Input: 1x1, Output: %dx%d\n', size(test4_1_output));

% Test 4.2: Single element, type 2
fprintf('Test 4.2: qupz - Single element, type 2\n');
test4_2_input = 7;
test4_2_output = qupz(test4_2_input, 2);
results.edge_qupz_single_t2.input = test4_2_input;
results.edge_qupz_single_t2.output = test4_2_output;
fprintf('  Input: 1x1, Output: %dx%d\n', size(test4_2_output));

% Test 4.3: Non-square matrix, type 1
fprintf('Test 4.3: qupz - Non-square (2x4), type 1\n');
test4_3_input = [1, 2, 3, 4; 5, 6, 7, 8];
test4_3_output = qupz(test4_3_input, 1);
results.edge_qupz_nonsquare_t1.input = test4_3_input;
results.edge_qupz_nonsquare_t1.output = test4_3_output;
fprintf('  Input: 2x4, Output: %dx%d\n', size(test4_3_output));

% Test 4.4: Small non-zero matrix (avoiding zero matrix due to MATLAB resampz bug)
fprintf('Test 4.4: qupz - Small non-zero matrix 2x2\n');
test4_4_input = [1, 0; 0, 1];  % Changed from zeros(2,2) to avoid MATLAB resampz index bug
test4_4_output = qupz(test4_4_input, 1);
results.edge_qupz_small_nonzero.input = test4_4_input;
results.edge_qupz_small_nonzero.output = test4_4_output;
fprintf('  Input: 2x2 identity-like, Output: %dx%d\n', size(test4_4_output));

%% ============================================================
%% Section 5: resampz Edge Cases
%% ============================================================
fprintf('\nSection 5: resampz Edge Cases\n');
fprintf('------------------------------\n');

% Test 5.1: Single row, type 1
fprintf('Test 5.1: resampz - Single row, type 1\n');
test5_1_input = [1, 2, 3, 4];
test5_1_output = resampz(test5_1_input, 1, 1);
results.edge_resampz_row_t1.input = test5_1_input;
results.edge_resampz_row_t1.output = test5_1_output;
fprintf('  Input: 1x4, Output: %dx%d\n', size(test5_1_output));

% Test 5.2: Single column, type 3
fprintf('Test 5.2: resampz - Single column, type 3\n');
test5_2_input = [1; 2; 3];
test5_2_output = resampz(test5_2_input, 3, 1);
results.edge_resampz_col_t3.input = test5_2_input;
results.edge_resampz_col_t3.output = test5_2_output;
fprintf('  Input: 3x1, Output: %dx%d\n', size(test5_2_output));

% Test 5.3: Large shift
fprintf('Test 5.3: resampz - Large shift (shift=5)\n');
test5_3_input = reshape(1:6, 2, 3);
test5_3_output = resampz(test5_3_input, 1, 5);
results.edge_resampz_large_shift.input = test5_3_input;
results.edge_resampz_large_shift.output = test5_3_output;
fprintf('  Input: 2x3, Shift: 5, Output: %dx%d\n', size(test5_3_output));

% Test 5.4: Zero shift
fprintf('Test 5.4: resampz - Zero shift\n');
test5_4_input = reshape(1:6, 2, 3);
test5_4_output = resampz(test5_4_input, 1, 0);
results.edge_resampz_zero_shift.input = test5_4_input;
results.edge_resampz_zero_shift.output = test5_4_output;
fprintf('  Input: 2x3, Shift: 0, Output: %dx%d\n', size(test5_4_output));

%% ============================================================
%% Section 6: nssfbdec/nssfbrec Edge Cases
%% ============================================================
fprintf('\nSection 6: nssfbdec/nssfbrec Edge Cases\n');
fprintf('----------------------------------------\n');

% Test 6.1: Very small input (2x2)
fprintf('Test 6.1: nssfbdec - Very small input (2x2)\n');
test6_1_input = [1, 2; 3, 4];
[test6_1_h0, test6_1_h1] = dfilters('pkva', 'd');
[test6_1_y1, test6_1_y2] = nssfbdec(test6_1_input, test6_1_h0, test6_1_h1);
results.edge_nssfbdec_small.input = test6_1_input;
results.edge_nssfbdec_small.h0 = test6_1_h0;
results.edge_nssfbdec_small.h1 = test6_1_h1;
results.edge_nssfbdec_small.y1 = test6_1_y1;
results.edge_nssfbdec_small.y2 = test6_1_y2;
fprintf('  Input: 2x2, Output: %dx%d\n', size(test6_1_y1));

% Test 6.2: Zero input matrix
fprintf('Test 6.2: nssfbdec - Zero input matrix\n');
test6_2_input = zeros(4, 4);
[test6_2_h0, test6_2_h1] = dfilters('pkva', 'd');
[test6_2_y1, test6_2_y2] = nssfbdec(test6_2_input, test6_2_h0, test6_2_h1);
results.edge_nssfbdec_zero.input = test6_2_input;
results.edge_nssfbdec_zero.y1 = test6_2_y1;
results.edge_nssfbdec_zero.y2 = test6_2_y2;
fprintf('  Input: 4x4 zeros, Output: %dx%d\n', size(test6_2_y1));

% Test 6.3: Constant input matrix
fprintf('Test 6.3: nssfbdec - Constant input matrix (all 5s)\n');
test6_3_input = ones(4, 4) * 5;
[test6_3_h0, test6_3_h1] = dfilters('pkva', 'd');
[test6_3_y1, test6_3_y2] = nssfbdec(test6_3_input, test6_3_h0, test6_3_h1);
results.edge_nssfbdec_const.input = test6_3_input;
results.edge_nssfbdec_const.y1 = test6_3_y1;
results.edge_nssfbdec_const.y2 = test6_3_y2;
fprintf('  Input: 4x4 constant, Output: %dx%d\n', size(test6_3_y1));

% Test 6.4: Non-square input
fprintf('Test 6.4: nssfbdec - Non-square input (6x10)\n');
test6_4_input = rand(6, 10);
[test6_4_h0, test6_4_h1] = dfilters('pkva', 'd');
[test6_4_y1, test6_4_y2] = nssfbdec(test6_4_input, test6_4_h0, test6_4_h1);
results.edge_nssfbdec_nonsquare.input = test6_4_input;
results.edge_nssfbdec_nonsquare.y1 = test6_4_y1;
results.edge_nssfbdec_nonsquare.y2 = test6_4_y2;
fprintf('  Input: 6x10, Output: %dx%d\n', size(test6_4_y1));

% Test 6.5: Perfect reconstruction with small input
fprintf('Test 6.5: Perfect reconstruction - Small input (4x4)\n');
test6_5_input = rand(4, 4);
[test6_5_h0, test6_5_h1] = dfilters('pkva', 'd');
[test6_5_g0, test6_5_g1] = dfilters('pkva', 'r');
[test6_5_y1, test6_5_y2] = nssfbdec(test6_5_input, test6_5_h0, test6_5_h1);
test6_5_recon = nssfbrec(test6_5_y1, test6_5_y2, test6_5_g0, test6_5_g1);
test6_5_mse = mean((test6_5_input(:) - test6_5_recon(:)).^2);
results.edge_perfect_recon_small.input = test6_5_input;
results.edge_perfect_recon_small.recon = test6_5_recon;
results.edge_perfect_recon_small.mse = test6_5_mse;
fprintf('  Input: 4x4, MSE: %.15e\n', test6_5_mse);

%% ============================================================
%% Section 7: efilter2 Edge Cases
%% ============================================================
fprintf('\nSection 7: efilter2 Edge Cases\n');
fprintf('-------------------------------\n');

% Test 7.1: Small filter on small image
fprintf('Test 7.1: efilter2 - Small filter (1x1) on small image\n');
test7_1_input = [1, 2; 3, 4];
test7_1_filter = 0.5;
test7_1_output = efilter2(test7_1_input, test7_1_filter, 'per');
results.edge_efilter2_small.input = test7_1_input;
results.edge_efilter2_small.filter = test7_1_filter;
results.edge_efilter2_small.output = test7_1_output;
fprintf('  Input: 2x2, Filter: 1x1, Output: %dx%d\n', size(test7_1_output));

% Test 7.2: Large filter on small image
fprintf('Test 7.2: efilter2 - Large filter on small image\n');
test7_2_input = [1, 2, 3; 4, 5, 6; 7, 8, 9];
test7_2_filter = ones(5, 5) / 25;
test7_2_output = efilter2(test7_2_input, test7_2_filter, 'per');
results.edge_efilter2_large_filter.input = test7_2_input;
results.edge_efilter2_large_filter.filter = test7_2_filter;
results.edge_efilter2_large_filter.output = test7_2_output;
fprintf('  Input: 3x3, Filter: 5x5, Output: %dx%d\n', size(test7_2_output));

% Test 7.3: Maximum shift
fprintf('Test 7.3: efilter2 - Maximum shift\n');
test7_3_input = ones(5, 5);
test7_3_filter = ones(5, 5);
test7_3_shift = [2; 2];  % Maximum shift for a 5x5 filter
test7_3_output = efilter2(test7_3_input, test7_3_filter, 'per', test7_3_shift);
results.edge_efilter2_max_shift.input = test7_3_input;
results.edge_efilter2_max_shift.filter = test7_3_filter;
results.edge_efilter2_max_shift.shift = test7_3_shift;
results.edge_efilter2_max_shift.output = test7_3_output;
fprintf('  Input: 5x5, Filter: 5x5, Shift: [2,2], Output: %dx%d\n', size(test7_3_output));

%% ============================================================
%% Section 8: ldfilter and ld2quin Edge Cases
%% ============================================================
fprintf('\nSection 8: ldfilter and ld2quin Edge Cases\n');
fprintf('-------------------------------------------\n');

% Test 8.1: Smallest filter (pkva6)
fprintf('Test 8.1: ldfilter - pkva6 (smallest)\n');
test8_1_output = ldfilter('pkva6');
results.edge_ldfilter_pkva6.output = test8_1_output;
fprintf('  Output length: %d\n', length(test8_1_output));

% Test 8.2: ld2quin with pkva6
fprintf('Test 8.2: ld2quin - pkva6\n');
test8_2_beta = ldfilter('pkva6');
[test8_2_h0, test8_2_h1] = ld2quin(test8_2_beta);
results.edge_ld2quin_pkva6.beta = test8_2_beta;
results.edge_ld2quin_pkva6.h0 = test8_2_h0;
results.edge_ld2quin_pkva6.h1 = test8_2_h1;
fprintf('  Beta length: %d, h0: %dx%d, h1: %dx%d\n', ...
    length(test8_2_beta), size(test8_2_h0), size(test8_2_h1));

%% ============================================================
%% Section 9: Special Values Tests
%% ============================================================
fprintf('\nSection 9: Special Values Tests\n');
fprintf('--------------------------------\n');

% Test 9.1: Very small values
fprintf('Test 9.1: nssfbdec - Very small values (1e-10)\n');
test9_1_input = ones(4, 4) * 1e-10;
[test9_1_h0, test9_1_h1] = dfilters('pkva', 'd');
[test9_1_y1, test9_1_y2] = nssfbdec(test9_1_input, test9_1_h0, test9_1_h1);
results.edge_small_values.input = test9_1_input;
results.edge_small_values.y1 = test9_1_y1;
results.edge_small_values.y2 = test9_1_y2;
fprintf('  Input max: %.2e, Output max: %.2e\n', max(test9_1_input(:)), max(test9_1_y1(:)));

% Test 9.2: Very large values
fprintf('Test 9.2: nssfbdec - Very large values (1e10)\n');
test9_2_input = ones(4, 4) * 1e10;
[test9_2_h0, test9_2_h1] = dfilters('pkva', 'd');
[test9_2_y1, test9_2_y2] = nssfbdec(test9_2_input, test9_2_h0, test9_2_h1);
results.edge_large_values.input = test9_2_input;
results.edge_large_values.y1 = test9_2_y1;
results.edge_large_values.y2 = test9_2_y2;
fprintf('  Input max: %.2e, Output max: %.2e\n', max(test9_2_input(:)), max(test9_2_y1(:)));

% Test 9.3: Negative values
fprintf('Test 9.3: nssfbdec - Negative values\n');
test9_3_input = -rand(4, 4);
[test9_3_h0, test9_3_h1] = dfilters('pkva', 'd');
[test9_3_y1, test9_3_y2] = nssfbdec(test9_3_input, test9_3_h0, test9_3_h1);
results.edge_negative_values.input = test9_3_input;
results.edge_negative_values.y1 = test9_3_y1;
results.edge_negative_values.y2 = test9_3_y2;
fprintf('  Input min: %.4f, Output min: %.4f\n', min(test9_3_input(:)), min(test9_3_y1(:)));

% Test 9.4: Mixed positive and negative
fprintf('Test 9.4: nssfbdec - Mixed positive and negative\n');
test9_4_input = randn(4, 4);
[test9_4_h0, test9_4_h1] = dfilters('pkva', 'd');
[test9_4_y1, test9_4_y2] = nssfbdec(test9_4_input, test9_4_h0, test9_4_h1);
results.edge_mixed_values.input = test9_4_input;
results.edge_mixed_values.y1 = test9_4_y1;
results.edge_mixed_values.y2 = test9_4_y2;
fprintf('  Input range: [%.4f, %.4f], Output range: [%.4f, %.4f]\n', ...
    min(test9_4_input(:)), max(test9_4_input(:)), min(test9_4_y1(:)), max(test9_4_y1(:)));

%% ============================================================
%% Section 10: Different Filter Types
%% ============================================================
fprintf('\nSection 10: Different Filter Types\n');
fprintf('-----------------------------------\n');

% Test 10.1: pyrexc filters
fprintf('Test 10.1: nssfbdec - pyrexc filters\n');
test10_1_input = rand(8, 8);
[test10_1_h0, test10_1_h1] = atrousfilters('pyrexc');
[test10_1_y1, test10_1_y2] = nssfbdec(test10_1_input, test10_1_h0, test10_1_h1);
results.edge_filter_pyrexc.input = test10_1_input;
results.edge_filter_pyrexc.h0 = test10_1_h0;
results.edge_filter_pyrexc.h1 = test10_1_h1;
results.edge_filter_pyrexc.y1 = test10_1_y1;
results.edge_filter_pyrexc.y2 = test10_1_y2;
fprintf('  Filter: pyrexc, Input: 8x8, Output: %dx%d\n', size(test10_1_y1));

% Test 10.2: dmaxflat filters
fprintf('Test 10.2: dmaxflat filters - N=1\n');
test10_2_f1 = dmaxflat(1, 0);
test10_2_f2 = dmaxflat(1, 1);
results.edge_filter_dmaxflat_1.f1 = test10_2_f1;
results.edge_filter_dmaxflat_1.f2 = test10_2_f2;
fprintf('  N=1, f1: %dx%d, f2: %dx%d\n', size(test10_2_f1), size(test10_2_f2));

% Test 10.3: dmaxflat filters - N=2
fprintf('Test 10.3: dmaxflat filters - N=2\n');
test10_3_f1 = dmaxflat(2, 0);
test10_3_f2 = dmaxflat(2, 1);
results.edge_filter_dmaxflat_2.f1 = test10_3_f1;
results.edge_filter_dmaxflat_2.f2 = test10_3_f2;
fprintf('  N=2, f1: %dx%d, f2: %dx%d\n', size(test10_3_f1), size(test10_3_f2));

%% ============================================================
%% Save results
%% ============================================================
save('test_edge_cases_results.mat', '-struct', 'results');
fprintf('\n=== MATLAB edge cases tests completed ===\n');
fprintf('Results saved to test_edge_cases_results.mat\n');
fprintf('Total test sections: 10\n');
fprintf('Total individual tests: ~40\n');
