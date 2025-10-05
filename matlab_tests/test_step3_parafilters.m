% 测试 parafilters.m - 平行四边形滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 parafilters.m ===\n');

%% 测试数据准备 - 使用 dfilters 生成菱形滤波器对
% parafilters 需要两个输入: h1 (lowpass) 和 h2 (highpass)
[h1, h2] = dfilters('dmaxflat4', 'd');
% 对于非子采样情况需要缩放
h1 = h1 / sqrt(2);
h2 = h2 / sqrt(2);

%% 测试案例 1: 使用 dmaxflat4 生成平行四边形滤波器
[p0, p1] = parafilters(h1, h2);

%% 测试案例 2: 使用不同的菱形滤波器 (dmaxflat5)
[h1_2, h2_2] = dfilters('dmaxflat5', 'd');
h1_2 = h1_2 / sqrt(2);
h2_2 = h2_2 / sqrt(2);
[p0_2, p1_2] = parafilters(h1_2, h2_2);

%% 测试案例 3: 使用 dmaxflat7
[h1_3, h2_3] = dfilters('dmaxflat7', 'd');
h1_3 = h1_3 / sqrt(2);
h2_3 = h2_3 / sqrt(2);
[p0_3, p1_3] = parafilters(h1_3, h2_3);

%% 保存测试数据
save('../test_data/step3_parafilters.mat', ...
    'h1', 'h2', 'p0', 'p1', ...
    'h1_2', 'h2_2', 'p0_2', 'p1_2', ...
    'h1_3', 'h2_3', 'p0_3', 'p1_3');

fprintf('测试数据已保存到 test_data/step3_parafilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. dmaxflat4: 输入 h1=%dx%d, h2=%dx%d\n', size(h1,1), size(h1,2), size(h2,1), size(h2,2));
fprintf('     p0: ');
for i = 1:length(p0)
    fprintf('%dx%d ', size(p0{i},1), size(p0{i},2));
end
fprintf('\n     p1: ');
for i = 1:length(p1)
    fprintf('%dx%d ', size(p1{i},1), size(p1{i},2));
end
fprintf('\n');

fprintf('  2. dmaxflat5: 输入 h1=%dx%d, h2=%dx%d\n', size(h1_2,1), size(h1_2,2), size(h2_2,1), size(h2_2,2));
fprintf('  3. dmaxflat7: 输入 h1=%dx%d, h2=%dx%d\n', size(h1_3,1), size(h1_3,2), size(h2_3,1), size(h2_3,2));
