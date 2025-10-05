% 测试 efilter2.m - 带边界扩展的卷积
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 efilter2.m ===\n');

%% 测试数据准备
test_image = magic(8);
small_filter = [1 2 1; 2 4 2; 1 2 1] / 16;  % 简单的高斯滤波器

%% 测试案例 1: 使用对称扩展
result1 = efilter2(test_image, small_filter, 'sym');

%% 测试案例 2: 使用周期扩展
result2 = efilter2(test_image, small_filter, 'per');

%% 测试案例 3: 使用单个字符指定扩展类型
result3 = efilter2(test_image, small_filter, 's');

%% 测试案例 4: 较大的滤波器
large_filter = fspecial('gaussian', [5 5], 1.5);
result4 = efilter2(test_image, large_filter, 'sym');

%% 测试案例 5: 非方阵图像
rect_image = rand(6, 10);
result5 = efilter2(rect_image, small_filter, 'sym');

%% 测试案例 6: 非对称滤波器
asym_filter = [1 2 3; 4 5 6];
result6 = efilter2(test_image, asym_filter, 'per');

%% 保存测试数据
save('../test_data/step2_efilter2.mat', ...
    'test_image', 'small_filter', 'result1', 'result2', 'result3', ...
    'large_filter', 'result4', 'rect_image', 'result5', ...
    'asym_filter', 'result6');

fprintf('测试数据已保存到 test_data/step2_efilter2.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 8x8图像, 3x3滤波器, 对称扩展\n');
fprintf('  2. 8x8图像, 3x3滤波器, 周期扩展\n');
fprintf('  3. 8x8图像, 3x3滤波器, 单字符扩展类型\n');
fprintf('  4. 8x8图像, 5x5滤波器, 对称扩展\n');
fprintf('  5. 6x10图像, 3x3滤波器, 对称扩展\n');
fprintf('  6. 8x8图像, 2x3滤波器, 周期扩展\n');
