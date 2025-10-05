% 测试辅助函数: mctrans, ld2quin, qupz, wfilters, ldfilter
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试辅助滤波器函数 ===\n');

%% 测试 wfilters - 小波滤波器
fprintf('\n测试 wfilters...\n');
try
    [h_9_7, g_9_7] = wfilters('9-7');
    wfilters_97_exists = true;
catch
    fprintf('警告: wfilters 函数不可用\n');
    wfilters_97_exists = false;
    h_9_7 = [];
    g_9_7 = [];
end

%% 测试 ldfilter - Ladder 滤波器
fprintf('\n测试 ldfilter...\n');
try
    [h_ld, g_ld] = ldfilter(8);
    ldfilter_exists = true;
catch
    fprintf('警告: ldfilter 函数不可用\n');
    ldfilter_exists = false;
    h_ld = [];
    g_ld = [];
end

%% 测试 ld2quin - Ladder to Quincunx
fprintf('\n测试 ld2quin...\n');
try
    beta = [0.3 0.4 0.5 0.6];  % 示例 beta 参数
    [h0_quin, h1_quin] = ld2quin(beta);
    ld2quin_exists = true;
catch
    fprintf('警告: ld2quin 函数不可用\n');
    ld2quin_exists = false;
    h0_quin = [];
    h1_quin = [];
    beta = [];
end

%% 测试 mctrans - McClellan 变换
fprintf('\n测试 mctrans...\n');
try
    % 1-D 滤波器
    b = [1 2 3 2 1] / 9;
    % 变换矩阵
    t = [0 1 0; 1 0 1; 0 1 0] / 4;
    h_mc = mctrans(b, t);
    mctrans_exists = true;
catch
    fprintf('警告: mctrans 函数不可用\n');
    mctrans_exists = false;
    h_mc = [];
    b = [];
    t = [];
end

%% 测试 qupz - Quincunx 上采样
fprintf('\n测试 qupz...\n');
try
    x_test = magic(5);
    y_qupz1 = qupz(x_test, 1);
    y_qupz2 = qupz(x_test, 2);
    qupz_exists = true;
catch
    fprintf('警告: qupz 函数不可用\n');
    qupz_exists = false;
    x_test = [];
    y_qupz1 = [];
    y_qupz2 = [];
end

%% 保存测试数据
save('../test_data/step3_auxiliary_filters.mat', ...
    'wfilters_97_exists', 'h_9_7', 'g_9_7', ...
    'ldfilter_exists', 'h_ld', 'g_ld', ...
    'ld2quin_exists', 'beta', 'h0_quin', 'h1_quin', ...
    'mctrans_exists', 'b', 't', 'h_mc', ...
    'qupz_exists', 'x_test', 'y_qupz1', 'y_qupz2');

fprintf('\n测试数据已保存到 test_data/step3_auxiliary_filters.mat\n');
fprintf('\n测试结果摘要:\n');
fprintf('  wfilters:  %s\n', mat2str(wfilters_97_exists));
fprintf('  ldfilter:  %s\n', mat2str(ldfilter_exists));
fprintf('  ld2quin:   %s\n', mat2str(ld2quin_exists));
fprintf('  mctrans:   %s\n', mat2str(mctrans_exists));
fprintf('  qupz:      %s\n', mat2str(qupz_exists));
