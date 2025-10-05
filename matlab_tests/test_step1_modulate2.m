% 测试 modulate2.m - 矩阵调制
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 modulate2.m ===\n');

%% 测试数据准备
test_matrix = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];

%% 测试案例 1: 沿 'c' 方向调制 (列)
type = 'c';
result1 = modulate2(test_matrix, type);

%% 测试案例 2: 沿 'r' 方向调制 (行)
type = 'r';
result2 = modulate2(test_matrix, type);

%% 测试案例 3: 'b' 方向调制 (双向)
type = 'b';
result3 = modulate2(test_matrix, type);

%% 测试案例 4: 非方阵矩阵 - 'c' 调制
rect_matrix = [1 2 3; 4 5 6; 7 8 9; 10 11 12];
type = 'c';
result4 = modulate2(rect_matrix, type);

%% 测试案例 5: 非方阵矩阵 - 'r' 调制
type = 'r';
result5 = modulate2(rect_matrix, type);

%% 测试案例 6: 小矩阵测试
small_matrix = [1 2; 3 4];
result6_c = modulate2(small_matrix, 'c');
result6_r = modulate2(small_matrix, 'r');
result6_b = modulate2(small_matrix, 'b');

%% 保存测试数据
save('../test_data/step1_modulate2.mat', ...
    'test_matrix', 'result1', 'result2', 'result3', ...
    'rect_matrix', 'result4', 'result5', ...
    'small_matrix', 'result6_c', 'result6_r', 'result6_b');

fprintf('测试数据已保存到 test_data/step1_modulate2.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 4x4矩阵, 列调制\n');
fprintf('  2. 4x4矩阵, 行调制\n');
fprintf('  3. 4x4矩阵, 双向调制\n');
fprintf('  4. 4x3矩阵, 列调制\n');
fprintf('  5. 4x3矩阵, 行调制\n');
fprintf('  6. 2x2矩阵, 全部调制方式\n');
