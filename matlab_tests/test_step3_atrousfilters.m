% 测试 atrousfilters.m - à trous 金字塔滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 atrousfilters.m ===\n');

%% 测试案例 1: '9-7' 滤波器
fname1 = '9-7';
[h0_1, h1_1, g0_1, g1_1] = atrousfilters(fname1);

%% 测试案例 2: 'maxflat' 滤波器
fname2 = 'maxflat';
[h0_2, h1_2, g0_2, g1_2] = atrousfilters(fname2);

%% 测试案例 3: 'pyr' 滤波器
fname3 = 'pyr';
[h0_3, h1_3, g0_3, g1_3] = atrousfilters(fname3);

%% 测试案例 4: 'pyrexc' 滤波器
fname4 = 'pyrexc';
[h0_4, h1_4, g0_4, g1_4] = atrousfilters(fname4);

%% 保存测试数据
save('../test_data/step3_atrousfilters.mat', ...
    'fname1', 'h0_1', 'h1_1', 'g0_1', 'g1_1', ...
    'fname2', 'h0_2', 'h1_2', 'g0_2', 'g1_2', ...
    'fname3', 'h0_3', 'h1_3', 'g0_3', 'g1_3', ...
    'fname4', 'h0_4', 'h1_4', 'g0_4', 'g1_4');

fprintf('测试数据已保存到 test_data/step3_atrousfilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 9-7:     h0=%dx%d, h1=%dx%d, g0=%dx%d, g1=%dx%d\n', ...
    size(h0_1,1), size(h0_1,2), size(h1_1,1), size(h1_1,2), size(g0_1,1), size(g0_1,2), size(g1_1,1), size(g1_1,2));
fprintf('  2. maxflat: h0=%dx%d, h1=%dx%d, g0=%dx%d, g1=%dx%d\n', ...
    size(h0_2,1), size(h0_2,2), size(h1_2,1), size(h1_2,2), size(g0_2,1), size(g0_2,2), size(g1_2,1), size(g1_2,2));
fprintf('  3. pyr:     h0=%dx%d, h1=%dx%d, g0=%dx%d, g1=%dx%d\n', ...
    size(h0_3,1), size(h0_3,2), size(h1_3,1), size(h1_3,2), size(g0_3,1), size(g0_3,2), size(g1_3,1), size(g1_3,2));
fprintf('  4. pyrexc:  h0=%dx%d, h1=%dx%d, g0=%dx%d, g1=%dx%d\n', ...
    size(h0_4,1), size(h0_4,2), size(h1_4,1), size(h1_4,2), size(g0_4,1), size(g0_4,2), size(g1_4,1), size(g1_4,2));
