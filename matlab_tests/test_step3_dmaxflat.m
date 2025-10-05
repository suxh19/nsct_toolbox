% 测试 dmaxflat.m - 菱形最大平坦滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 dmaxflat.m ===\n');

%% 测试案例 1: d = 1
d = 1;
h1 = dmaxflat(d);

%% 测试案例 2: d = 2
d = 2;
h2 = dmaxflat(d);

%% 测试案例 3: d = 3
d = 3;
h3 = dmaxflat(d);

%% 测试案例 4: d = 4
d = 4;
h4 = dmaxflat(d);

%% 测试案例 5: d = 5
d = 5;
h5 = dmaxflat(d);

%% 保存测试数据
save('../test_data/step3_dmaxflat.mat', ...
    'h1', 'h2', 'h3', 'h4', 'h5');

fprintf('测试数据已保存到 test_data/step3_dmaxflat.mat\n');
fprintf('测试用例:\n');
fprintf('  1. d=1, 滤波器大小: %dx%d\n', size(h1,1), size(h1,2));
fprintf('  2. d=2, 滤波器大小: %dx%d\n', size(h2,1), size(h2,2));
fprintf('  3. d=3, 滤波器大小: %dx%d\n', size(h3,1), size(h3,2));
fprintf('  4. d=4, 滤波器大小: %dx%d\n', size(h4,1), size(h4,2));
fprintf('  5. d=5, 滤波器大小: %dx%d\n', size(h5,1), size(h5,2));
