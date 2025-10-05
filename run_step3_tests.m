% 运行所有第 3 步测试 - 滤波器生成
% 这个脚本会依次运行所有第 3 步的测试并生成测试数据

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  第 3 步：滤波器生成测试\n');
fprintf('========================================\n\n');

% 确保在正确的目录
original_dir = pwd;
cd('matlab_tests');

try
    % 测试 1: dmaxflat
    fprintf('\n[1/5] 运行 test_step3_dmaxflat.m ...\n');
    run('test_step3_dmaxflat.m');
    
    % 测试 2: dfilters
    fprintf('\n[2/5] 运行 test_step3_dfilters.m ...\n');
    run('test_step3_dfilters.m');
    
    % 测试 3: atrousfilters
    fprintf('\n[3/5] 运行 test_step3_atrousfilters.m ...\n');
    run('test_step3_atrousfilters.m');
    
    % 测试 4: parafilters
    fprintf('\n[4/5] 运行 test_step3_parafilters.m ...\n');
    run('test_step3_parafilters.m');
    
    % 测试 5: 辅助函数
    fprintf('\n[5/5] 运行 test_step3_auxiliary.m ...\n');
    run('test_step3_auxiliary.m');
    
    fprintf('\n========================================\n');
    fprintf('  所有第 3 步测试完成!\n');
    fprintf('  测试数据已保存到 test_data/ 目录\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
end

% 返回原目录
cd(original_dir);
