% test_filters_matlab.m
% Generate test cases for filters.py functions
% This script will generate MATLAB reference outputs and save them to .mat files

%% Setup
clear all;
rng(42); % Set random seed for reproducibility
test_results = struct();

fprintf('=== MATLAB Filters Test Suite ===\n\n');

%% Test 1: ldfilter - pkva12
fprintf('Test 1: ldfilter - pkva12\n');
test1_fname = 'pkva12';
test1_output = ldfilter(test1_fname);

test_results.test1.fname = test1_fname;
test_results.test1.output = test1_output;
fprintf('  Filter: %s, Length: %d\n', test1_fname, length(test1_output));

%% Test 2: ldfilter - pkva8
fprintf('\nTest 2: ldfilter - pkva8\n');
test2_fname = 'pkva8';
test2_output = ldfilter(test2_fname);

test_results.test2.fname = test2_fname;
test_results.test2.output = test2_output;
fprintf('  Filter: %s, Length: %d\n', test2_fname, length(test2_output));

%% Test 3: ldfilter - pkva6
fprintf('\nTest 3: ldfilter - pkva6\n');
test3_fname = 'pkva6';
test3_output = ldfilter(test3_fname);

test_results.test3.fname = test3_fname;
test_results.test3.output = test3_output;
fprintf('  Filter: %s, Length: %d\n', test3_fname, length(test3_output));

%% Test 4: ld2quin - pkva6
fprintf('\nTest 4: ld2quin - pkva6\n');
test4_beta = ldfilter('pkva6');
[test4_h0, test4_h1] = ld2quin(test4_beta);

test_results.test4.beta = test4_beta;
test_results.test4.h0 = test4_h0;
test_results.test4.h1 = test4_h1;
fprintf('  Input length: %d, h0 size: %dx%d, h1 size: %dx%d\n', ...
    length(test4_beta), size(test4_h0), size(test4_h1));

%% Test 5: ld2quin - pkva12
fprintf('\nTest 5: ld2quin - pkva12\n');
test5_beta = ldfilter('pkva12');
[test5_h0, test5_h1] = ld2quin(test5_beta);

test_results.test5.beta = test5_beta;
test_results.test5.h0 = test5_h0;
test_results.test5.h1 = test5_h1;
fprintf('  Input length: %d, h0 size: %dx%d, h1 size: %dx%d\n', ...
    length(test5_beta), size(test5_h0), size(test5_h1));

%% Test 6: dmaxflat - N=1, d=0
fprintf('\nTest 6: dmaxflat - N=1, d=0\n');
test6_N = 1;
test6_d = 0;
test6_output = dmaxflat(test6_N, test6_d);

test_results.test6.N = test6_N;
test_results.test6.d = test6_d;
test_results.test6.output = test6_output;
fprintf('  N=%d, d=%d, Output size: %dx%d\n', test6_N, test6_d, size(test6_output));

%% Test 7: dmaxflat - N=2, d=1
fprintf('\nTest 7: dmaxflat - N=2, d=1\n');
test7_N = 2;
test7_d = 1;
test7_output = dmaxflat(test7_N, test7_d);

test_results.test7.N = test7_N;
test_results.test7.d = test7_d;
test_results.test7.output = test7_output;
fprintf('  N=%d, d=%d, Output size: %dx%d\n', test7_N, test7_d, size(test7_output));

%% Test 8: dmaxflat - N=3, d=0
fprintf('\nTest 8: dmaxflat - N=3, d=0\n');
test8_N = 3;
test8_d = 0;
test8_output = dmaxflat(test8_N, test8_d);

test_results.test8.N = test8_N;
test_results.test8.d = test8_d;
test_results.test8.output = test8_output;
fprintf('  N=%d, d=%d, Output size: %dx%d\n', test8_N, test8_d, size(test8_output));

%% Test 9: atrousfilters - pyr
fprintf('\nTest 9: atrousfilters - pyr\n');
test9_fname = 'pyr';
[test9_h0, test9_h1, test9_g0, test9_g1] = atrousfilters(test9_fname);

test_results.test9.fname = test9_fname;
test_results.test9.h0 = test9_h0;
test_results.test9.h1 = test9_h1;
test_results.test9.g0 = test9_g0;
test_results.test9.g1 = test9_g1;
fprintf('  Filter: %s, h0: %dx%d, h1: %dx%d, g0: %dx%d, g1: %dx%d\n', ...
    test9_fname, size(test9_h0), size(test9_h1), size(test9_g0), size(test9_g1));

%% Test 10: atrousfilters - pyrexc
fprintf('\nTest 10: atrousfilters - pyrexc\n');
test10_fname = 'pyrexc';
[test10_h0, test10_h1, test10_g0, test10_g1] = atrousfilters(test10_fname);

test_results.test10.fname = test10_fname;
test_results.test10.h0 = test10_h0;
test_results.test10.h1 = test10_h1;
test_results.test10.g0 = test10_g0;
test_results.test10.g1 = test10_g1;
fprintf('  Filter: %s, h0: %dx%d, h1: %dx%d, g0: %dx%d, g1: %dx%d\n', ...
    test10_fname, size(test10_h0), size(test10_h1), size(test10_g0), size(test10_g1));

%% Test 11: mctrans - simple case
fprintf('\nTest 11: mctrans - simple case\n');
test11_b = [1, 2, 1] / 4;
test11_t = [0, 1, 0; 1, 0, 1; 0, 1, 0] / 4;
test11_output = mctrans(test11_b, test11_t);

test_results.test11.b = test11_b;
test_results.test11.t = test11_t;
test_results.test11.output = test11_output;
fprintf('  b length: %d, t size: %dx%d, Output size: %dx%d\n', ...
    length(test11_b), size(test11_t), size(test11_output));

%% Test 12: mctrans - larger filter
fprintf('\nTest 12: mctrans - larger filter\n');
test12_b = [1, 3, 3, 1] / 8;
test12_t = [0, 1, 0; 1, 0, 1; 0, 1, 0] / 4;
test12_output = mctrans(test12_b, test12_t);

test_results.test12.b = test12_b;
test_results.test12.t = test12_t;
test_results.test12.output = test12_output;
fprintf('  b length: %d, t size: %dx%d, Output size: %dx%d\n', ...
    length(test12_b), size(test12_t), size(test12_output));

%% Test 13: efilter2 - basic filtering
fprintf('\nTest 13: efilter2 - basic filtering\n');
test13_x = reshape(1:9, 3, 3);
test13_f = [0, 1, 0; 1, -4, 1; 0, 1, 0];
test13_extmod = 'per';
test13_shift = [0; 0];
test13_output = efilter2(test13_x, test13_f, test13_extmod, test13_shift);

test_results.test13.x = test13_x;
test_results.test13.f = test13_f;
test_results.test13.extmod = test13_extmod;
test_results.test13.shift = test13_shift;
test_results.test13.output = test13_output;
fprintf('  Input size: %dx%d, Filter size: %dx%d, Output size: %dx%d\n', ...
    size(test13_x), size(test13_f), size(test13_output));

%% Test 14: efilter2 - with shift
fprintf('\nTest 14: efilter2 - with shift\n');
test14_x = ones(4, 4);
test14_f = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16;
test14_extmod = 'per';
test14_shift = [1; 0];
test14_output = efilter2(test14_x, test14_f, test14_extmod, test14_shift);

test_results.test14.x = test14_x;
test_results.test14.f = test14_f;
test_results.test14.extmod = test14_extmod;
test_results.test14.shift = test14_shift;
test_results.test14.output = test14_output;
fprintf('  Input size: %dx%d, Filter size: %dx%d, Shift: [%d, %d], Output size: %dx%d\n', ...
    size(test14_x), size(test14_f), test14_shift, size(test14_output));

%% Test 15: parafilters - basic case
fprintf('\nTest 15: parafilters - basic case\n');
test15_f1 = ones(3, 3);
test15_f2 = ones(3, 3) * 2;
[test15_y1, test15_y2] = parafilters(test15_f1, test15_f2);

test_results.test15.f1 = test15_f1;
test_results.test15.f2 = test15_f2;
test_results.test15.y1_1 = test15_y1{1};
test_results.test15.y1_2 = test15_y1{2};
test_results.test15.y1_3 = test15_y1{3};
test_results.test15.y1_4 = test15_y1{4};
test_results.test15.y2_1 = test15_y2{1};
test_results.test15.y2_2 = test15_y2{2};
test_results.test15.y2_3 = test15_y2{3};
test_results.test15.y2_4 = test15_y2{4};
fprintf('  f1 size: %dx%d, f2 size: %dx%d\n', size(test15_f1), size(test15_f2));
fprintf('  y1{1}: %dx%d, y1{2}: %dx%d, y1{3}: %dx%d, y1{4}: %dx%d\n', ...
    size(test15_y1{1}), size(test15_y1{2}), size(test15_y1{3}), size(test15_y1{4}));

%% Test 16: parafilters - with dmaxflat filters
fprintf('\nTest 16: parafilters - with dmaxflat filters\n');
test16_f1 = dmaxflat(2, 0);
test16_f2 = dmaxflat(2, 1);
[test16_y1, test16_y2] = parafilters(test16_f1, test16_f2);

test_results.test16.f1 = test16_f1;
test_results.test16.f2 = test16_f2;
test_results.test16.y1_1 = test16_y1{1};
test_results.test16.y1_2 = test16_y1{2};
test_results.test16.y1_3 = test16_y1{3};
test_results.test16.y1_4 = test16_y1{4};
test_results.test16.y2_1 = test16_y2{1};
test_results.test16.y2_2 = test16_y2{2};
test_results.test16.y2_3 = test16_y2{3};
test_results.test16.y2_4 = test16_y2{4};
fprintf('  f1 size: %dx%d, f2 size: %dx%d\n', size(test16_f1), size(test16_f2));
fprintf('  y1{1}: %dx%d, y1{2}: %dx%d, y1{3}: %dx%d, y1{4}: %dx%d\n', ...
    size(test16_y1{1}), size(test16_y1{2}), size(test16_y1{3}), size(test16_y1{4}));

%% Save all test results
save('test_filters_results.mat', 'test_results');
fprintf('\n=== Test data saved to test_filters_results.mat ===\n');
fprintf('Total tests: 16\n');
