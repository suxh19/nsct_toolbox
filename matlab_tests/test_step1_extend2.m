% 测试 extend2.m - 图像边界扩展
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 extend2.m ===\n');

%% 测试数据准备
% 创建一个简单的测试矩阵
test_matrix = [1 2 3; 4 5 6; 7 8 9];

%% 测试案例 1: 周期性扩展 (ru=1, rd=1, cl=1, cr=1)
ru = 1; rd = 1; cl = 1; cr = 1;
result1 = extend2(test_matrix, ru, rd, cl, cr, 'per');

%% 测试案例 2: 周期性扩展的另一种参数 (ru=2, rd=2, cl=2, cr=2)
ru = 2; rd = 2; cl = 2; cr = 2;
result2 = extend2(test_matrix, ru, rd, cl, cr, 'per');

%% 测试案例 3: 非对称扩展
ru = 1; rd = 2; cl = 3; cr = 1;
result3 = extend2(test_matrix, ru, rd, cl, cr, 'per');

%% 测试案例 4: 较大的矩阵
large_matrix = magic(8);
result4 = extend2(large_matrix, 3, 3, 3, 3, 'per');

%% 保存测试数据
save('../test_data/step1_extend2.mat', ...
    'test_matrix', 'result1', 'result2', 'result3', ...
    'large_matrix', 'result4');

fprintf('测试数据已保存到 test_data/step1_extend2.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 周期扩展 (1,1,1,1)\n');
fprintf('  2. 周期扩展 (2,2,2,2)\n');
fprintf('  3. 非对称周期扩展 (1,2,3,1)\n');
fprintf('  4. 大矩阵周期扩展 (3,3,3,3)\n');
