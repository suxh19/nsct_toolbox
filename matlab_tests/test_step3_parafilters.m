% 测试 parafilters.m - 平行四边形滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 parafilters.m ===\n');

%% 测试数据准备 - 使用 dmaxflat 生成基础滤波器
h = dmaxflat(3);

%% 测试案例 1: 使用 dmaxflat3 生成平行四边形滤波器
[p0, p1] = parafilters(h);

%% 测试案例 2: 使用不同的菱形滤波器
h2 = dmaxflat(4);
[p0_2, p1_2] = parafilters(h2);

%% 测试案例 3: 使用 dmaxflat2
h3 = dmaxflat(2);
[p0_3, p1_3] = parafilters(h3);

%% 保存测试数据
save('../test_data/step3_parafilters.mat', ...
    'h', 'p0', 'p1', ...
    'h2', 'p0_2', 'p1_2', ...
    'h3', 'p0_3', 'p1_3');

fprintf('测试数据已保存到 test_data/step3_parafilters.mat\n');
fprintf('测试用例:\n');
fprintf('  1. dmaxflat(3): 输入=%dx%d\n', size(h,1), size(h,2));
fprintf('     p0: ');
for i = 1:length(p0)
    fprintf('%dx%d ', size(p0{i},1), size(p0{i},2));
end
fprintf('\n     p1: ');
for i = 1:length(p1)
    fprintf('%dx%d ', size(p1{i},1), size(p1{i},2));
end
fprintf('\n');

fprintf('  2. dmaxflat(4): 输入=%dx%d\n', size(h2,1), size(h2,2));
fprintf('  3. dmaxflat(2): 输入=%dx%d\n', size(h3,1), size(h3,2));
