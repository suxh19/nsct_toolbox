% 测试完整的 NSCT 分解和重构
% 这是最重要的端到端测试
% 用于验证 Python 实现的正确性

clear all;
close all;

% 添加父目录到 MATLAB 路径
addpath('..');

fprintf('=== 测试 NSCT 完整分解和重构 ===\n');

%% 测试数据准备
% 使用较大的测试图像以避免边界扩展问题
test_image_small = magic(128);

% 使用一个中等大小的测试图像
test_image_medium = magic(256);

% 创建一个真实的测试图像（如果有）
try
    test_image_real = double(imread('../zoneplate.png'));
    has_real_image = true;
catch
    try
        test_image_real = double(rgb2gray(imread('cameraman.tif')));
        has_real_image = true;
    catch
        fprintf('警告: 无法加载真实图像，使用合成图像\n');
        test_image_real = peaks(256);
        has_real_image = false;
    end
end

%% 测试案例 1: 基本 NSCT 分解 - 3 级，每级 [2 3 4] 方向
fprintf('\n测试案例 1: 3 级分解，方向数 [2 3 4]\n');
nlevels1 = [2 3 4];
c1 = nsctdec(test_image_small, nlevels1);
fprintf('  系数单元数: %d\n', length(c1));

% 重构
rec1 = nsctrec(c1);
error1 = max(abs(test_image_small(:) - rec1(:)));
fprintf('  重构误差: %e\n', error1);

%% 测试案例 2: 2 级分解，方向数 [1 2]
fprintf('\n测试案例 2: 2 级分解，方向数 [1 2]\n');
nlevels2 = [1 2];
c2 = nsctdec(test_image_small, nlevels2);
fprintf('  系数单元数: %d\n', length(c2));

rec2 = nsctrec(c2);
error2 = max(abs(test_image_small(:) - rec2(:)));
fprintf('  重构误差: %e\n', error2);

%% 测试案例 3: 4 级分解，方向数 [2 3 3 4]
fprintf('\n测试案例 3: 4 级分解，方向数 [2 3 3 4]\n');
nlevels3 = [2 3 3 4];
c3 = nsctdec(test_image_medium, nlevels3);
fprintf('  系数单元数: %d\n', length(c3));

rec3 = nsctrec(c3);
error3 = max(abs(test_image_medium(:) - rec3(:)));
fprintf('  重构误差: %e\n', error3);

%% 测试案例 4: 使用不同的滤波器 - 'pkva' 和 '9-7'
fprintf('\n测试案例 4: 自定义滤波器\n');
nlevels4 = [2 3];
dfilt = 'pkva';  % 方向滤波器
afilt = '9-7';   % 金字塔滤波器
try
    c4 = nsctdec(test_image_small, nlevels4, dfilt, afilt);
    rec4 = nsctrec(c4, dfilt, afilt);
    error4 = max(abs(test_image_small(:) - rec4(:)));
    fprintf('  重构误差: %e\n', error4);
    has_custom_filters = true;
catch ME
    fprintf('  警告: 自定义滤波器测试失败: %s\n', ME.message);
    c4 = {};
    rec4 = [];
    error4 = inf;
    has_custom_filters = false;
end

%% 测试案例 5: 真实/复杂图像
fprintf('\n测试案例 5: 真实/复杂图像\n');
nlevels5 = [2 3 4];
c5 = nsctdec(test_image_real, nlevels5);
fprintf('  系数单元数: %d\n', length(c5));

rec5 = nsctrec(c5);
error5 = max(abs(test_image_real(:) - rec5(:)));
fprintf('  重构误差: %e\n', error5);

%% 保存所有测试数据
save('../test_data/step5_nsct_full.mat', ...
    'test_image_small', 'nlevels1', 'c1', 'rec1', 'error1', ...
    'nlevels2', 'c2', 'rec2', 'error2', ...
    'test_image_medium', 'nlevels3', 'c3', 'rec3', 'error3', ...
    'nlevels4', 'dfilt', 'afilt', 'c4', 'rec4', 'error4', 'has_custom_filters', ...
    'test_image_real', 'nlevels5', 'c5', 'rec5', 'error5', 'has_real_image', ...
    '-v7.3');  % 使用 v7.3 格式以支持大文件

fprintf('\n测试数据已保存到 test_data/step5_nsct_full.mat\n');
fprintf('\n========================================\n');
fprintf('完整 NSCT 测试摘要:\n');
fprintf('  案例 1 (128x128, [2 3 4]):     误差 = %e\n', error1);
fprintf('  案例 2 (128x128, [1 2]):       误差 = %e\n', error2);
fprintf('  案例 3 (256x256, [2 3 3 4]):   误差 = %e\n', error3);
fprintf('  案例 4 (自定义滤波器):          误差 = %e\n', error4);
fprintf('  案例 5 (真实图像, [2 3 4]):     误差 = %e\n', error5);
fprintf('========================================\n');
