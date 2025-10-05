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

%% 测试案例 4: '7-9' 滤波器 (分解) - Cohen-Daubechies 滤波器对
fname4 = '7-9';  % 使用 '7-9' 而不是 '9-7'，这是 dfilters.m 中支持的名称
type4 = 'd';
[h0_4, h1_4] = dfilters(fname4, type4);

%% 测试案例 5-7: 跳过 pkva-half 系列滤波器
% 注意: pkva-half4/6/8 需要 ldfilterhalf() 函数，该函数在原工具箱中缺失
% 这些测试被跳过以避免 "函数或变量 'ldfilterhalf' 无法识别" 错误
fprintf('跳过测试: pkva-half4, pkva-half6, pkva-half8 (需要缺失的 ldfilterhalf 函数)\n');

%% 保存测试数据
save('../test_data/step3_dfilters.mat', ...
    'fname1', 'type1', 'h0_1', 'h1_1', ...
    'fname2', 'type2', 'h0_2', 'h1_2', ...
    'fname3', 'type3', 'h0_3', 'h1_3', ...
    'fname4', 'type4', 'h0_4', 'h1_4');

fprintf('测试数据已保存到 test_data/step3_dfilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. pkva:       h0=%dx%d, h1=%dx%d\n', size(h0_1,1), size(h0_1,2), size(h1_1,1), size(h1_1,2));
fprintf('  2. cd:         h0=%dx%d, h1=%dx%d\n', size(h0_2,1), size(h0_2,2), size(h1_2,1), size(h1_2,2));
fprintf('  3. dmaxflat7:  h0=%dx%d, h1=%dx%d\n', size(h0_3,1), size(h0_3,2), size(h1_3,1), size(h1_3,2));
fprintf('  4. 7-9:        h0=%dx%d, h1=%dx%d\n', size(h0_4,1), size(h0_4,2), size(h1_4,1), size(h1_4,2));
