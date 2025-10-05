% 测试 symext.m - 对称边界扩展
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 symext.m ===\n');

%% 测试数据准备
test_matrix = [1 2 3; 4 5 6; 7 8 9];
shift = [1, 1]; % 默认延迟补偿

%% 测试案例 1: 对称扩展 (3x3滤波器)
h1 = ones(3, 3); % 3x3滤波器
result1 = symext(test_matrix, h1, shift);

%% 测试案例 2: 对称扩展 (5x5滤波器)
h2 = ones(5, 5); % 5x5滤波器
result2 = symext(test_matrix, h2, shift);

%% 测试案例 3: 对称扩展 (1x1滤波器)
h3 = ones(1, 1); % 1x1滤波器
result3 = symext(test_matrix, h3, shift);

%% 测试案例 4: 较大矩阵与7x7滤波器
large_matrix = magic(6);
h4 = ones(7, 7); % 7x7滤波器
result4 = symext(large_matrix, h4, shift);

%% 保存测试数据
save('../test_data/step1_symext.mat', ...
    'test_matrix', 'h1', 'result1', 'h2', 'result2', 'h3', 'result3', ...
    'large_matrix', 'h4', 'result4', 'shift');

fprintf('测试数据已保存到 test_data/step1_symext.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 3x3矩阵, 3x3滤波器\n');
fprintf('  2. 3x3矩阵, 5x5滤波器\n');
fprintf('  3. 3x3矩阵, 1x1滤波器\n');
fprintf('  4. 6x6矩阵, 7x7滤波器\n');
