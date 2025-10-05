% 测试 resampz.m - 矩阵剪切重采样
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 resampz.m ===\n');

%% 测试数据准备
test_matrix = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16];

%% 测试案例 1: type=1 (行方向，R1矩阵), shift = 1
type = 1;
shift = 1;
result1 = resampz(test_matrix, type, shift);

%% 测试案例 2: type=3 (列方向，R3矩阵), shift = 1
type = 3;
shift = 1;
result2 = resampz(test_matrix, type, shift);

%% 测试案例 3: type=2 (行方向，R2矩阵), shift = 2
type = 2;
shift = 2;
result3 = resampz(test_matrix, type, shift);

%% 测试案例 4: type=4 (列方向，R4矩阵), shift = -1
type = 4;
shift = -1;
result4 = resampz(test_matrix, type, shift);

%% 测试案例 5: type=1 (行方向，R1矩阵), shift = 2
type = 1;
shift = 2;
result5 = resampz(test_matrix, type, shift);

%% 测试案例 6: 较大矩阵
large_matrix = magic(8);
type = 1;
shift = 1;
result6 = resampz(large_matrix, type, shift);

%% 测试案例 7: 小矩阵
small_matrix = [1 2 3; 4 5 6; 7 8 9];
type = 3;
shift = 1;
result7 = resampz(small_matrix, type, shift);

%% 保存测试数据
save('../test_data/step1_resampz.mat', ...
    'test_matrix', 'result1', 'result2', 'result3', ...
    'result4', 'result5', 'large_matrix', 'result6', ...
    'small_matrix', 'result7');

fprintf('测试数据已保存到 test_data/step1_resampz.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 4x4矩阵, type=1 (R1矩阵), shift=1\n');
fprintf('  2. 4x4矩阵, type=3 (R3矩阵), shift=1\n');
fprintf('  3. 4x4矩阵, type=2 (R2矩阵), shift=2\n');
fprintf('  4. 4x4矩阵, type=4 (R4矩阵), shift=-1\n');
fprintf('  5. 4x4矩阵, type=1 (R1矩阵), shift=2\n');
fprintf('  6. 8x8矩阵, type=1 (R1矩阵), shift=1\n');
fprintf('  7. 3x3矩阵, type=3 (R3矩阵), shift=1\n');
