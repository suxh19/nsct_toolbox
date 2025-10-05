% 测试 dmaxflat.m - 菱形最大平坦滤波器生成
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试 dmaxflat.m ===\n');

%% 测试案例 1: N = 1 (阶数1, d = 1)
N = 1;
d = 1;
h1 = dmaxflat(N, d);

%% 测试案例 2: N = 2 (阶数2, d = 1)
N = 2;
d = 1;
h2 = dmaxflat(N, d);

%% 测试案例 3: N = 3 (阶数3, d = 1)
N = 3;
d = 1;
h3 = dmaxflat(N, d);

%% 测试案例 4: N = 4 (阶数4, d = 1)
N = 4;
d = 1;
h4 = dmaxflat(N, d);

%% 测试案例 5: N = 5 (阶数5, d = 1)
N = 5;
d = 1;
h5 = dmaxflat(N, d);

%% 测试案例 6: N = 1, d = 0 (测试不同的d值)
N = 1;
d = 0;
h1_d0 = dmaxflat(N, d);

%% 保存测试数据
save('../test_data/step3_dmaxflat.mat', ...
    'h1', 'h2', 'h3', 'h4', 'h5', 'h1_d0');

fprintf('测试数据已保存到 test_data/step3_dmaxflat.mat\n');
fprintf('测试用例:\n');
fprintf('  1. N=1, d=1, 滤波器大小: %dx%d\n', size(h1,1), size(h1,2));
fprintf('  2. N=2, d=1, 滤波器大小: %dx%d\n', size(h2,1), size(h2,2));
fprintf('  3. N=3, d=1, 滤波器大小: %dx%d\n', size(h3,1), size(h3,2));
fprintf('  4. N=4, d=1, 滤波器大小: %dx%d\n', size(h4,1), size(h4,2));
fprintf('  5. N=5, d=1, 滤波器大小: %dx%d\n', size(h5,1), size(h5,2));
fprintf('  6. N=1, d=0, 滤波器大小: %dx%d\n', size(h1_d0,1), size(h1_d0,2));
