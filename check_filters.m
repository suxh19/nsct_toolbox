% 检查实际滤波器的大小
clear all;

% 检查不同类型滤波器的大小
try
    [h0,h1,g0,g1] = atrousfilters('9-7');
    fprintf('9-7 滤波器:\n');
    fprintf('  h0 大小: [%d, %d]\n', size(h0));
    fprintf('  h1 大小: [%d, %d]\n', size(h1));
catch
    fprintf('无法生成 9-7 滤波器\n');
end

try
    [h0,h1,g0,g1] = atrousfilters('maxflat');
    fprintf('maxflat 滤波器:\n');
    fprintf('  h0 大小: [%d, %d]\n', size(h0));
    fprintf('  h1 大小: [%d, %d]\n', size(h1));
catch
    fprintf('无法生成 maxflat 滤波器\n');
end

% 测试小尺寸滤波器
fprintf('\n测试小尺寸滤波器与 symext:\n');
x = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20; 21 22 23 24 25];
shift = [1, 1];

% 3x3 滤波器（应该可以工作，因为输入是5x5）
h_small = ones(3, 3);
fprintf('输入矩阵大小: [%d, %d]\n', size(x));
fprintf('滤波器大小: [%d, %d]\n', size(h_small));

try
    result = symext(x, h_small, shift);
    fprintf('symext 成功! 输出大小: [%d, %d]\n', size(result));
catch ME
    fprintf('symext 失败: %s\n', ME.message);
end