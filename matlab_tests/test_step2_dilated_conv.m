% 测试带膨胀的卷积功能 (模拟 MEX 文件功能)
% 这些测试用于验证 Python 中使用 scipy.signal.convolve2d 的 dilation 参数
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试带膨胀的卷积 (MEX 文件功能) ===\n');

%% 测试数据准备
test_image = magic(16);
small_filter = [1 2 1; 2 4 2; 1 2 1] / 16;

%% 测试案例 1: 无膨胀 (m = [1 0; 0 1])
m = [1 0; 0 1];
% 使用标准 conv2
result1_standard = conv2(test_image, small_filter, 'valid');

% 如果 atrousc MEX 文件已编译，可以使用它
% 否则使用等效的方法
try
    result1_mex = atrousc(test_image, small_filter, m);
    has_mex = true;
catch
    fprintf('警告: atrousc MEX 文件未编译，使用标准 conv2\n');
    result1_mex = result1_standard;
    has_mex = false;
end

%% 测试案例 2: 膨胀因子 = 2 (m = [2 0; 0 2])
m = [2 0; 0 2];
% 手动上采样滤波器
upsampled_filter2 = upsample2df(small_filter, 2);
result2_manual = conv2(test_image, upsampled_filter2, 'valid');

if has_mex
    result2_mex = atrousc(test_image, small_filter, m);
else
    result2_mex = result2_manual;
end

%% 测试案例 3: 膨胀因子 = 4 (m = [4 0; 0 4])
m = [4 0; 0 4];
upsampled_filter4 = upsample2df(small_filter, 4);
result3_manual = conv2(test_image, upsampled_filter4, 'valid');

if has_mex
    result3_mex = atrousc(test_image, small_filter, m);
else
    result3_mex = result3_manual;
end

%% 测试案例 4: 不同的滤波器
filter_laplacian = [0 1 0; 1 -4 1; 0 1 0];
m = [2 0; 0 2];
upsampled_laplacian = upsample2df(filter_laplacian, 2);
result4_manual = conv2(test_image, upsampled_laplacian, 'valid');

if has_mex
    result4_mex = atrousc(test_image, filter_laplacian, m);
else
    result4_mex = result4_manual;
end

%% 测试案例 5: 较小的图像和膨胀
small_image = magic(8);
m = [3 0; 0 3];
upsampled_filter3 = upsample2df(small_filter, 3);
result5_manual = conv2(small_image, upsampled_filter3, 'valid');

if has_mex
    result5_mex = atrousc(small_image, small_filter, m);
else
    result5_mex = result5_manual;
end

%% 测试案例 6: 非对称膨胀 (不同的行列膨胀因子，如果支持)
% 注意：标准的 atrousc 可能只支持对角矩阵
m_asym = [2 0; 0 3];
% 这个案例可能需要特殊处理

%% 保存测试数据
save('../test_data/step2_dilated_conv.mat', ...
    'test_image', 'small_filter', ...
    'result1_standard', 'result1_mex', ...
    'upsampled_filter2', 'result2_manual', 'result2_mex', ...
    'upsampled_filter4', 'result3_manual', 'result3_mex', ...
    'filter_laplacian', 'upsampled_laplacian', 'result4_manual', 'result4_mex', ...
    'small_image', 'upsampled_filter3', 'result5_manual', 'result5_mex', ...
    'has_mex');

fprintf('测试数据已保存到 test_data/step2_dilated_conv.mat\n');
fprintf('测试用例:\n');
fprintf('  1. 无膨胀 (m=1)\n');
fprintf('  2. 膨胀因子=2\n');
fprintf('  3. 膨胀因子=4\n');
fprintf('  4. 拉普拉斯滤波器, 膨胀因子=2\n');
fprintf('  5. 小图像, 膨胀因子=3\n');
fprintf('  注意: 保存了手动上采样的结果和 MEX 结果（如果可用）\n');
