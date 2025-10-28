% test_utils_matlab.m
% Generate test cases for utils.py functions
% This script will generate MATLAB reference outputs and save them to .mat files

%% Setup
clear all;
rng(42); % Set random seed for reproducibility
test_results = struct();

fprintf('=== MATLAB Utils Test Suite ===\n\n');

%% Test 1: extend2 - Periodic extension (basic)
fprintf('Test 1: extend2 - Periodic extension (basic)\n');
test1_input = reshape(0:15, 4, 4)';
test1_ru = 2; test1_rd = 2; test1_cl = 2; test1_cr = 2;
test1_output = extend2(test1_input, test1_ru, test1_rd, test1_cl, test1_cr, 'per');

test_results.test1.input = test1_input;
test_results.test1.ru = test1_ru;
test_results.test1.rd = test1_rd;
test_results.test1.cl = test1_cl;
test_results.test1.cr = test1_cr;
test_results.test1.extmod = 'per';
test_results.test1.output = test1_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test1_input), size(test1_output));

%% Test 2: extend2 - Periodic extension (small)
fprintf('\nTest 2: extend2 - Periodic extension\n');
test2_input = reshape(1:12, 3, 4)';
test2_ru = 1; test2_rd = 1; test2_cl = 1; test2_cr = 1;
test2_output = extend2(test2_input, test2_ru, test2_rd, test2_cl, test2_cr, 'per');

test_results.test2.input = test2_input;
test_results.test2.ru = test2_ru;
test_results.test2.rd = test2_rd;
test_results.test2.cl = test2_cl;
test_results.test2.cr = test2_cr;
test_results.test2.extmod = 'per';
test_results.test2.output = test2_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test2_input), size(test2_output));

%% Test 3: extend2 - Quincunx periodic extension (row)
fprintf('\nTest 3: extend2 - Quincunx periodic extension (row)\n');
test3_input = reshape(1:24, 4, 6)';
test3_ru = 2; test3_rd = 2; test3_cl = 2; test3_cr = 2;
test3_output = extend2(test3_input, test3_ru, test3_rd, test3_cl, test3_cr, 'qper_row');

test_results.test3.input = test3_input;
test_results.test3.ru = test3_ru;
test_results.test3.rd = test3_rd;
test_results.test3.cl = test3_cl;
test_results.test3.cr = test3_cr;
test_results.test3.extmod = 'qper_row';
test_results.test3.output = test3_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test3_input), size(test3_output));

%% Test 4: extend2 - Quincunx periodic extension (col)
fprintf('\nTest 4: extend2 - Quincunx periodic extension (col)\n');
test4_input = reshape(1:24, 6, 4)';
test4_ru = 2; test4_rd = 2; test4_cl = 2; test4_cr = 2;
test4_output = extend2(test4_input, test4_ru, test4_rd, test4_cl, test4_cr, 'qper_col');

test_results.test4.input = test4_input;
test_results.test4.ru = test4_ru;
test_results.test4.rd = test4_rd;
test_results.test4.cl = test4_cl;
test_results.test4.cr = test4_cr;
test_results.test4.extmod = 'qper_col';
test_results.test4.output = test4_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test4_input), size(test4_output));

%% Test 5: upsample2df - Power 1
fprintf('\nTest 5: upsample2df - Power 1\n');
test5_input = [1, 2, 3; 4, 5, 6; 7, 8, 9];
test5_power = 1;
test5_output = upsample2df(test5_input, test5_power);

test_results.test5.input = test5_input;
test_results.test5.power = test5_power;
test_results.test5.output = test5_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test5_input), size(test5_output));

%% Test 6: upsample2df - Power 2
fprintf('\nTest 6: upsample2df - Power 2\n');
test6_input = [1, 2; 3, 4];
test6_power = 2;
test6_output = upsample2df(test6_input, test6_power);

test_results.test6.input = test6_input;
test_results.test6.power = test6_power;
test_results.test6.output = test6_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test6_input), size(test6_output));

%% Test 7: modulate2 - Row modulation
fprintf('\nTest 7: modulate2 - Row modulation\n');
test7_input = ones(3, 4);
test7_type = 'r';
test7_center = [0, 0];
test7_output = modulate2(test7_input, test7_type, test7_center);

test_results.test7.input = test7_input;
test_results.test7.type = test7_type;
test_results.test7.center = test7_center;
test_results.test7.output = test7_output;
fprintf('  Input size: %dx%d, Type: %s\n', size(test7_input), test7_type);

%% Test 8: modulate2 - Column modulation
fprintf('\nTest 8: modulate2 - Column modulation\n');
test8_input = ones(4, 5);
test8_type = 'c';
test8_center = [0, 0];
test8_output = modulate2(test8_input, test8_type, test8_center);

test_results.test8.input = test8_input;
test_results.test8.type = test8_type;
test_results.test8.center = test8_center;
test_results.test8.output = test8_output;
fprintf('  Input size: %dx%d, Type: %s\n', size(test8_input), test8_type);

%% Test 9: modulate2 - Both directions
fprintf('\nTest 9: modulate2 - Both directions\n');
test9_input = ones(3, 4);
test9_type = 'b';
test9_center = [0, 0];
test9_output = modulate2(test9_input, test9_type, test9_center);

test_results.test9.input = test9_input;
test_results.test9.type = test9_type;
test_results.test9.center = test9_center;
test_results.test9.output = test9_output;
fprintf('  Input size: %dx%d, Type: %s\n', size(test9_input), test9_type);

%% Test 10: modulate2 - Both directions with center offset
fprintf('\nTest 10: modulate2 - Both directions with center offset\n');
test10_input = ones(4, 4);
test10_type = 'b';
test10_center = [1, -1];
test10_output = modulate2(test10_input, test10_type, test10_center);

test_results.test10.input = test10_input;
test_results.test10.type = test10_type;
test_results.test10.center = test10_center;
test_results.test10.output = test10_output;
fprintf('  Input size: %dx%d, Type: %s, Center: [%d, %d]\n', ...
    size(test10_input), test10_type, test10_center);

%% Test 11: resampz - Type 1
fprintf('\nTest 11: resampz - Type 1 (R1 = [1,1;0,1])\n');
test11_input = reshape(1:6, 2, 3);
test11_type = 1;
test11_shift = 1;
test11_output = resampz(test11_input, test11_type, test11_shift);

test_results.test11.input = test11_input;
test_results.test11.type = test11_type;
test_results.test11.shift = test11_shift;
test_results.test11.output = test11_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test11_input), size(test11_output));

%% Test 12: resampz - Type 2
fprintf('\nTest 12: resampz - Type 2 (R2 = [1,-1;0,1])\n');
test12_input = reshape(1:6, 2, 3);
test12_type = 2;
test12_shift = 1;
test12_output = resampz(test12_input, test12_type, test12_shift);

test_results.test12.input = test12_input;
test_results.test12.type = test12_type;
test_results.test12.shift = test12_shift;
test_results.test12.output = test12_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test12_input), size(test12_output));

%% Test 13: resampz - Type 3
fprintf('\nTest 13: resampz - Type 3 (R3 = [1,0;1,1])\n');
test13_input = reshape(1:6, 2, 3);
test13_type = 3;
test13_shift = 1;
test13_output = resampz(test13_input, test13_type, test13_shift);

test_results.test13.input = test13_input;
test_results.test13.type = test13_type;
test_results.test13.shift = test13_shift;
test_results.test13.output = test13_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test13_input), size(test13_output));

%% Test 14: resampz - Type 4
fprintf('\nTest 14: resampz - Type 4 (R4 = [1,0;-1,1])\n');
test14_input = reshape(1:6, 2, 3);
test14_type = 4;
test14_shift = 1;
test14_output = resampz(test14_input, test14_type, test14_shift);

test_results.test14.input = test14_input;
test_results.test14.type = test14_type;
test_results.test14.shift = test14_shift;
test_results.test14.output = test14_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test14_input), size(test14_output));

%% Test 15: resampz - Type 1 with shift 2
fprintf('\nTest 15: resampz - Type 1 with shift 2\n');
test15_input = reshape(1:12, 3, 4);
test15_type = 1;
test15_shift = 2;
test15_output = resampz(test15_input, test15_type, test15_shift);

test_results.test15.input = test15_input;
test_results.test15.type = test15_type;
test_results.test15.shift = test15_shift;
test_results.test15.output = test15_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test15_input), size(test15_output));

%% Test 16: qupz - Type 1
fprintf('\nTest 16: qupz - Type 1 (Q1 = [1,-1;1,1])\n');
test16_input = [1, 2; 3, 4];
test16_type = 1;
test16_output = qupz(test16_input, test16_type);

test_results.test16.input = test16_input;
test_results.test16.type = test16_type;
test_results.test16.output = test16_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test16_input), size(test16_output));

%% Test 17: qupz - Type 2
fprintf('\nTest 17: qupz - Type 2 (Q2 = [1,1;-1,1])\n');
test17_input = [1, 2; 3, 4];
test17_type = 2;
test17_output = qupz(test17_input, test17_type);

test_results.test17.input = test17_input;
test_results.test17.type = test17_type;
test_results.test17.output = test17_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test17_input), size(test17_output));

%% Test 18: qupz - Type 1 with larger matrix
fprintf('\nTest 18: qupz - Type 1 with 3x3 matrix\n');
test18_input = [1, 2, 3; 4, 5, 6; 7, 8, 9];
test18_type = 1;
test18_output = qupz(test18_input, test18_type);

test_results.test18.input = test18_input;
test_results.test18.type = test18_type;
test_results.test18.output = test18_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test18_input), size(test18_output));

%% Test 19: qupz - Type 2 with larger matrix
fprintf('\nTest 19: qupz - Type 2 with 3x3 matrix\n');
test19_input = [1, 2, 3; 4, 5, 6; 7, 8, 9];
test19_type = 2;
test19_output = qupz(test19_input, test19_type);

test_results.test19.input = test19_input;
test_results.test19.type = test19_type;
test_results.test19.output = test19_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', size(test19_input), size(test19_output));

%% Save all test results
save('test_utils_results.mat', 'test_results');
fprintf('\n=== Test data saved to test_utils_results.mat ===\n');
fprintf('Total tests: 19\n');
