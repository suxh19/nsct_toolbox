% 运行所有测试 - 主测试脚本
% 这个脚本会依次运行所有 5 个步骤的测试并生成所有测试数据
% 
% 用途：为 MATLAB 到 Python 的翻译生成参考数据

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  NSCT 工具箱测试套件\n');
fprintf('  生成 Python 翻译的参考数据\n');
fprintf('========================================\n\n');

start_time = tic;

%% 确保测试目录存在
if ~exist('matlab_tests', 'dir')
    error('matlab_tests 目录不存在!');
end

if ~exist('test_data', 'dir')
    mkdir('test_data');
end

%% 运行各步骤测试
try
    fprintf('\n>>> 第 1 步：底层图像和矩阵操作\n');
    fprintf('================================================\n');
    run('run_step1_tests.m');
    
    fprintf('\n>>> 第 2 步：核心卷积和滤波\n');
    fprintf('================================================\n');
    run('run_step2_tests.m');
    
    fprintf('\n>>> 第 3 步：滤波器生成\n');
    fprintf('================================================\n');
    run('run_step3_tests.m');
    
    fprintf('\n>>> 第 4 步：核心分解与重构模块\n');
    fprintf('================================================\n');
    run('run_step4_tests.m');
    
    fprintf('\n>>> 第 5 步：顶层 NSCT 接口\n');
    fprintf('================================================\n');
    run('run_step5_tests.m');
    
    elapsed_time = toc(start_time);
    
    fprintf('\n\n========================================\n');
    fprintf('  所有测试完成!\n');
    fprintf('  总耗时: %.2f 秒\n', elapsed_time);
    fprintf('========================================\n');
    
    % 列出生成的测试数据文件
    fprintf('\n生成的测试数据文件:\n');
    test_files = dir('test_data/*.mat');
    for i = 1:length(test_files)
        file_info = dir(fullfile('test_data', test_files(i).name));
        size_mb = file_info.bytes / 1024 / 1024;
        fprintf('  [%2d] %-40s (%.2f MB)\n', i, test_files(i).name, size_mb);
    end
    
    fprintf('\n下一步:\n');
    fprintf('  1. 查看 test_data/ 目录中的 .mat 文件\n');
    fprintf('  2. 使用这些数据验证 Python 实现的正确性\n');
    fprintf('  3. 按照文档顺序从第 1 步开始翻译 Python 代码\n');
    fprintf('  4. 每完成一步，用对应的 .mat 文件验证结果\n');
    
catch ME
    fprintf('\n测试失败!\n');
    fprintf('错误: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
        fprintf('\n完整堆栈:\n');
        for i = 1:length(ME.stack)
            fprintf('  [%d] %s (行 %d)\n', i, ME.stack(i).file, ME.stack(i).line);
        end
    end
end

fprintf('\n========================================\n');
