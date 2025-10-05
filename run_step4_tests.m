% 运行第 4 步测试 - 核心分解与重构模块
% 这个脚本会运行核心分解与重构的测试

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  第 4 步：核心分解与重构测试\n');
fprintf('========================================\n\n');

% 确保在正确的目录
original_dir = pwd;
cd('matlab_tests');

try
    fprintf('运行 test_step4_core_decomposition.m ...\n');
    run('test_step4_core_decomposition.m');
    
    fprintf('\n========================================\n');
    fprintf('  第 4 步测试完成!\n');
    fprintf('  测试数据已保存到 test_data/ 目录\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
end

% 返回原目录
cd(original_dir);
