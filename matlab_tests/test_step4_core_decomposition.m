% 测试核心分解和重构模块
% 测试 nssfbdec/rec, nsdfbdec/rec, atrousdec/rec
% 用于验证 Python 实现的正确性

clear all;
close all;

fprintf('=== 测试核心分解与重构模块 ===\n');

%% 准备测试数据
test_image = magic(64);

%% 测试案例 1: nssfbdec 和 nssfbrec (双通道滤波器组)
fprintf('\n[1] 测试 nssfbdec/nssfbrec ...\n');
try
    % 获取滤波器
    [h0, h1] = dfilters('pkva');
    
    % 分解
    [y0, y1] = nssfbdec(test_image, h0, h1);
    
    % 重构
    rec_nssfb = nssfbrec(y0, y1, h0, h1);
    
    % 计算误差
    error_nssfb = max(abs(test_image(:) - rec_nssfb(:)));
    fprintf('  重构误差: %e\n', error_nssfb);
    
    nssfb_success = true;
catch ME
    fprintf('  错误: %s\n', ME.message);
    y0 = []; y1 = []; rec_nssfb = []; error_nssfb = inf;
    nssfb_success = false;
end

%% 测试案例 2: nsdfbdec 和 nsdfbrec (方向滤波器组)
fprintf('\n[2] 测试 nsdfbdec/nsdfbrec ...\n');
try
    % 使用 3 级方向分解
    nlevel = 3;  % 意味着 2^3 = 8 个方向
    
    % 方向分解
    y_nsdfb = nsdfbdec(test_image, 'pkva', nlevel);
    fprintf('  方向子带数: %d\n', length(y_nsdfb));
    
    % 方向重构
    rec_nsdfb = nsdfbrec(y_nsdfb, 'pkva');
    
    % 计算误差
    error_nsdfb = max(abs(test_image(:) - rec_nsdfb(:)));
    fprintf('  重构误差: %e\n', error_nsdfb);
    
    nsdfb_success = true;
catch ME
    fprintf('  错误: %s\n', ME.message);
    nlevel = 0; y_nsdfb = {}; rec_nsdfb = []; error_nsdfb = inf;
    nsdfb_success = false;
end

%% 测试案例 3: nsdfbdec 和 nsdfbrec - 不同级别
fprintf('\n[3] 测试 nsdfbdec/nsdfbrec (不同级别) ...\n');
try
    nlevel2 = 2;  % 2^2 = 4 个方向
    y_nsdfb2 = nsdfbdec(test_image, 'pkva', nlevel2);
    rec_nsdfb2 = nsdfbrec(y_nsdfb2, 'pkva');
    error_nsdfb2 = max(abs(test_image(:) - rec_nsdfb2(:)));
    fprintf('  方向子带数: %d, 重构误差: %e\n', length(y_nsdfb2), error_nsdfb2);
    nsdfb2_success = true;
catch ME
    fprintf('  错误: %s\n', ME.message);
    nlevel2 = 0; y_nsdfb2 = {}; rec_nsdfb2 = []; error_nsdfb2 = inf;
    nsdfb2_success = false;
end

%% 测试案例 4: atrousdec 和 atrousrec (金字塔分解)
fprintf('\n[4] 测试 atrousdec/atrousrec ...\n');
try
    % 获取金字塔滤波器
    [h, g] = atrousfilters('maxflat');
    
    % 3 级金字塔分解
    nlev_pyramid = 3;
    y_atrous = atrousdec(test_image, h, nlev_pyramid);
    fprintf('  金字塔子带数: %d\n', length(y_atrous));
    
    % 金字塔重构
    rec_atrous = atrousrec(y_atrous, g);
    
    % 计算误差
    error_atrous = max(abs(test_image(:) - rec_atrous(:)));
    fprintf('  重构误差: %e\n', error_atrous);
    
    atrous_success = true;
catch ME
    fprintf('  错误: %s\n', ME.message);
    h = []; g = []; nlev_pyramid = 0; y_atrous = {}; rec_atrous = []; error_atrous = inf;
    atrous_success = false;
end

%% 测试案例 5: atrousdec - 不同级别和滤波器
fprintf('\n[5] 测试 atrousdec/atrousrec (不同参数) ...\n');
try
    [h2, g2] = atrousfilters('binom');
    nlev_pyramid2 = 2;
    y_atrous2 = atrousdec(test_image, h2, nlev_pyramid2);
    rec_atrous2 = atrousrec(y_atrous2, g2);
    error_atrous2 = max(abs(test_image(:) - rec_atrous2(:)));
    fprintf('  金字塔子带数: %d, 重构误差: %e\n', length(y_atrous2), error_atrous2);
    atrous2_success = true;
catch ME
    fprintf('  错误: %s\n', ME.message);
    h2 = []; g2 = []; nlev_pyramid2 = 0; y_atrous2 = {}; rec_atrous2 = []; error_atrous2 = inf;
    atrous2_success = false;
end

%% 保存测试数据
save('../test_data/step4_core_decomposition.mat', ...
    'test_image', ...
    'h0', 'h1', 'y0', 'y1', 'rec_nssfb', 'error_nssfb', 'nssfb_success', ...
    'nlevel', 'y_nsdfb', 'rec_nsdfb', 'error_nsdfb', 'nsdfb_success', ...
    'nlevel2', 'y_nsdfb2', 'rec_nsdfb2', 'error_nsdfb2', 'nsdfb2_success', ...
    'h', 'g', 'nlev_pyramid', 'y_atrous', 'rec_atrous', 'error_atrous', 'atrous_success', ...
    'h2', 'g2', 'nlev_pyramid2', 'y_atrous2', 'rec_atrous2', 'error_atrous2', 'atrous2_success');

fprintf('\n测试数据已保存到 test_data/step4_core_decomposition.mat\n');
fprintf('\n========================================\n');
fprintf('核心分解与重构测试摘要:\n');
fprintf('  nssfbdec/rec:       %s (误差: %e)\n', mat2str(nssfb_success), error_nssfb);
fprintf('  nsdfbdec/rec (L=3): %s (误差: %e)\n', mat2str(nsdfb_success), error_nsdfb);
fprintf('  nsdfbdec/rec (L=2): %s (误差: %e)\n', mat2str(nsdfb2_success), error_nsdfb2);
fprintf('  atrousdec/rec (L=3):%s (误差: %e)\n', mat2str(atrous_success), error_atrous);
fprintf('  atrousdec/rec (L=2):%s (误差: %e)\n', mat2str(atrous2_success), error_atrous2);
fprintf('========================================\n');
