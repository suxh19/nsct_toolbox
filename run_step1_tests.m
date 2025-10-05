% 运行所有第 1 步测试 - 底层图像和矩阵操作
% 这个脚本会依次运行所有第 1 步的测试并生成测试数据

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  第 1 步：底层图像和矩阵操作测试\n');
fprintf('========================================\n\n');

% 确保在正确的目录
original_dir = pwd;
cd('matlab_tests');

try
    % 测试 1: extend2
    fprintf('\n[1/5] 运行 test_step1_extend2.m ...\n');
    run('test_step1_extend2.m');
    
    % 测试 2: symext
    fprintf('\n[2/5] 运行 test_step1_symext.m ...\n');
    run('test_step1_symext.m');
    
    % 测试 3: upsample2df
    fprintf('\n[3/5] 运行 test_step1_upsample2df.m ...\n');
    run('test_step1_upsample2df.m');
    
    % 测试 4: modulate2
    fprintf('\n[4/5] 运行 test_step1_modulate2.m ...\n');
    run('test_step1_modulate2.m');
    
    % 测试 5: resampz
    fprintf('\n[5/5] 运行 test_step1_resampz.m ...\n');
    run('test_step1_resampz.m');
    
    fprintf('\n========================================\n');
    fprintf('  所有第 1 步测试完成!\n');
    fprintf('  测试数据已保存到 test_data/ 目录\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
end

% 返回原目录
cd(original_dir);
