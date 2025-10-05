% 测试 symext.m - 对称边界扩展
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 symext.m ===\n');

%% 测试数据准备
test_matrix = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25]; % 5x5矩阵
shift = [1, 1]; % 默认延迟补偿

%% 测试案例 1: 对称扩展 (3x3滤波器)
h1 = ones(3, 3); % 3x3滤波器
result1 = symext(test_matrix, h1, shift);

%% 测试案例 2: 对称扩展 (1x1滤波器)
h2 = ones(1, 1); % 1x1滤波器
result2 = symext(test_matrix, h2, shift);

%% 测试案例 3: 较大矩阵与3x3滤波器
large_matrix = magic(8); % 8x8矩阵
h3 = ones(3, 3); % 3x3滤波器
result3 = symext(large_matrix, h3, shift);

%% 测试案例 4: 使用实际的9-7滤波器
try
    [h0, h1, g0, g1] = atrousfilters('9-7');
    % 使用更大的测试矩阵以适应9x9滤波器
    big_matrix = magic(15); % 15x15矩阵
    result4 = symext(big_matrix, h0, shift); % 使用9x9的h0滤波器
    h4 = h0;
    test_matrix_4 = big_matrix;
catch
    % 如果atrousfilters不可用，使用5x5滤波器
    test_matrix_4 = magic(10); % 10x10矩阵
    h4 = ones(5, 5); % 5x5滤波器
    result4 = symext(test_matrix_4, h4, shift);
end

%% 显示测试结果
fprintf('测试结果:\n');
fprintf('  1. 5x5矩阵 + 3x3滤波器 → %dx%d输出\n', size(result1));
fprintf('  2. 5x5矩阵 + 1x1滤波器 → %dx%d输出\n', size(result2));
fprintf('  3. 8x8矩阵 + 3x3滤波器 → %dx%d输出\n', size(result3));
fprintf('  4. 大矩阵 + 滤波器 → %dx%d输出\n', size(result4));

%% 保存测试数据
save('../test_data/step1_symext.mat', ...
    'test_matrix', 'h1', 'result1', 'h2', 'result2', ...
    'large_matrix', 'h3', 'result3', 'test_matrix_4', 'h4', 'result4', 'shift');

fprintf('测试数据已保存到 test_data/step1_symext.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 5x5矩阵, 3x3滤波器\n');
fprintf('  2. 5x5矩阵, 1x1滤波器\n');
fprintf('  3. 8x8矩阵, 3x3滤波器\n');
fprintf('  4. 大矩阵, 实际滤波器\n');
