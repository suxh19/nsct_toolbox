% 测试 dfilters.m - 方向滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 dfilters.m ===\n');

%% 测试案例 1: 'pkva' 滤波器 (分解)
fname1 = 'pkva';
type1 = 'd';  % decomposition
[h0_1, h1_1] = dfilters(fname1, type1);

%% 测试案例 2: 'cd' 滤波器 (分解)
fname2 = 'cd';
type2 = 'd';
[h0_2, h1_2] = dfilters(fname2, type2);

%% 测试案例 3: 'dmaxflat7' 滤波器 (分解)
fname3 = 'dmaxflat7';
type3 = 'd';
[h0_3, h1_3] = dfilters(fname3, type3);

%% 测试案例 4: '9-7' 滤波器 (分解)
fname4 = '9-7';
type4 = 'd';
[h0_4, h1_4] = dfilters(fname4, type4);

%% 测试案例 5: 'pkva6' 滤波器 (分解)
fname5 = 'pkva6';
type5 = 'd';
[h0_5, h1_5] = dfilters(fname5, type5);

%% 测试案例 6: 'pkva8' 滤波器 (分解)
fname6 = 'pkva8';
type6 = 'd';
[h0_6, h1_6] = dfilters(fname6, type6);

%% 测试案例 7: 'pkva12' 滤波器 (分解)
fname7 = 'pkva12';
type7 = 'd';
[h0_7, h1_7] = dfilters(fname7, type7);

%% 保存测试数据
save('../test_data/step3_dfilters.mat', ...
    'fname1', 'type1', 'h0_1', 'h1_1', ...
    'fname2', 'type2', 'h0_2', 'h1_2', ...
    'fname3', 'type3', 'h0_3', 'h1_3', ...
    'fname4', 'type4', 'h0_4', 'h1_4', ...
    'fname5', 'type5', 'h0_5', 'h1_5', ...
    'fname6', 'type6', 'h0_6', 'h1_6', ...
    'fname7', 'type7', 'h0_7', 'h1_7');

fprintf('测试数据已保存到 test_data/step3_dfilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. pkva:      h0=%dx%d, h1=%dx%d\n', size(h0_1,1), size(h0_1,2), size(h1_1,1), size(h1_1,2));
fprintf('  2. cd:        h0=%dx%d, h1=%dx%d\n', size(h0_2,1), size(h0_2,2), size(h1_2,1), size(h1_2,2));
fprintf('  3. dmaxflat7: h0=%dx%d, h1=%dx%d\n', size(h0_3,1), size(h0_3,2), size(h1_3,1), size(h1_3,2));
fprintf('  4. 9-7:       h0=%dx%d, h1=%dx%d\n', size(h0_4,1), size(h0_4,2), size(h1_4,1), size(h1_4,2));
fprintf('  5. pkva6:     h0=%dx%d, h1=%dx%d\n', size(h0_5,1), size(h0_5,2), size(h1_5,1), size(h1_5,2));
fprintf('  6. pkva8:     h0=%dx%d, h1=%dx%d\n', size(h0_6,1), size(h0_6,2), size(h1_6,1), size(h1_6,2));
fprintf('  7. pkva12:    h0=%dx%d, h1=%dx%d\n', size(h0_7,1), size(h0_7,2), size(h1_7,1), size(h1_7,2));
