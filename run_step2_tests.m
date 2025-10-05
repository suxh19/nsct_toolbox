% 运行所有第 2 步测试 - 核心卷积和滤波
% 这个脚本会依次运行所有第 2 步的测试并生成测试数据

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  第 2 步：核心卷积和滤波测试\n');
fprintf('========================================\n\n');

% 确保在正确的目录
original_dir = pwd;
cd('matlab_tests');

try
    % 测试 1: efilter2
    fprintf('\n[1/2] 运行 test_step2_efilter2.m ...\n');
    run('test_step2_efilter2.m');
    
    % 测试 2: dilated convolution (MEX 功能)
    fprintf('\n[2/2] 运行 test_step2_dilated_conv.m ...\n');
    run('test_step2_dilated_conv.m');
    
    fprintf('\n========================================\n');
    fprintf('  所有第 2 步测试完成!\n');
    fprintf('  测试数据已保存到 test_data/ 目录\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
end

% 返回原目录
cd(original_dir);
