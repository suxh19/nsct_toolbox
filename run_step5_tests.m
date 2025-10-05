% 运行第 5 步测试 - 顶层接口
% 这是最重要的端到端测试

clear all;
close all;
clc;

fprintf('========================================\n');
fprintf('  第 5 步：顶层 NSCT 接口测试\n');
fprintf('========================================\n\n');

% 确保在正确的目录
original_dir = pwd;
cd('matlab_tests');

try
    fprintf('运行 test_step5_nsct_full.m ...\n');
    run('test_step5_nsct_full.m');
    
    fprintf('\n========================================\n');
    fprintf('  第 5 步测试完成!\n');
    fprintf('  测试数据已保存到 test_data/ 目录\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('\n错误: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('在文件: %s, 行: %d\n', ME.stack(1).file, ME.stack(1).line);
    end
end

% 返回原目录
cd(original_dir);
