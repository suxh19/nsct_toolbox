% 测试 upsample2df.m - 滤波器上采样
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 upsample2df.m ===\n');

%% 测试数据准备
% 小滤波器
small_filter = [1 2; 3 4];

%% 测试案例 1: 上采样因子 = 2
m = 2;
result1 = upsample2df(small_filter, m);

%% 测试案例 2: 上采样因子 = 3
m = 3;
result2 = upsample2df(small_filter, m);

%% 测试案例 3: 上采样因子 = 4
m = 4;
result3 = upsample2df(small_filter, m);

%% 测试案例 4: 较大滤波器，上采样因子 = 2
large_filter = [1 2 3; 4 5 6; 7 8 9];
m = 2;
result4 = upsample2df(large_filter, m);

%% 测试案例 5: 非方阵滤波器
rect_filter = [1 2 3 4; 5 6 7 8];
m = 2;
result5 = upsample2df(rect_filter, m);

%% 保存测试数据
save('../test_data/step1_upsample2df.mat', ...
    'small_filter', 'result1', 'result2', 'result3', ...
    'large_filter', 'result4', 'rect_filter', 'result5');

fprintf('测试数据已保存到 test_data/step1_upsample2df.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 2x2滤波器, 上采样因子=2\n');
fprintf('  2. 2x2滤波器, 上采样因子=3\n');
fprintf('  3. 2x2滤波器, 上采样因子=4\n');
fprintf('  4. 3x3滤波器, 上采样因子=2\n');
fprintf('  5. 2x4滤波器, 上采样因子=2\n');
