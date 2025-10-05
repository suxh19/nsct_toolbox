% 测试 atrousfilters.m - à trous 金字塔滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 atrousfilters.m ===\n');

%% 测试案例 1: 'maxflat' 滤波器
fname1 = 'maxflat';
[h_1, g_1] = atrousfilters(fname1);

%% 测试案例 2: 'haar' 滤波器
fname2 = 'haar';
[h_2, g_2] = atrousfilters(fname2);

%% 测试案例 3: 'binom' 滤波器
fname3 = 'binom';
[h_3, g_3] = atrousfilters(fname3);

%% 测试案例 4: 'c3' 滤波器
fname4 = 'c3';
[h_4, g_4] = atrousfilters(fname4);

%% 测试案例 5: 'c5' 滤波器
fname5 = 'c5';
[h_5, g_5] = atrousfilters(fname5);

%% 保存测试数据
save('../test_data/step3_atrousfilters.mat', ...
    'fname1', 'h_1', 'g_1', ...
    'fname2', 'h_2', 'g_2', ...
    'fname3', 'h_3', 'g_3', ...
    'fname4', 'h_4', 'g_4', ...
    'fname5', 'h_5', 'g_5');

fprintf('测试数据已保存到 test_data/step3_atrousfilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. maxflat: h=%dx%d, g=%dx%d\n', size(h_1,1), size(h_1,2), size(g_1,1), size(g_1,2));
fprintf('  2. haar:    h=%dx%d, g=%dx%d\n', size(h_2,1), size(h_2,2), size(g_2,1), size(g_2,2));
fprintf('  3. binom:   h=%dx%d, g=%dx%d\n', size(h_3,1), size(h_3,2), size(g_3,1), size(g_3,2));
fprintf('  4. c3:      h=%dx%d, g=%dx%d\n', size(h_4,1), size(h_4,2), size(g_4,1), size(g_4,2));
fprintf('  5. c5:      h=%dx%d, g=%dx%d\n', size(h_5,1), size(h_5,2), size(g_5,1), size(g_5,2));
