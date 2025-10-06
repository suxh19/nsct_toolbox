%% MATLAB 版本的 NSCT 分解和重建
% 独立运行并保存结果供后续对比

clear; clc;

% 添加 nsct_matlab 目录到路径
addpath('nsct_matlab');

fprintf('======================================================================\n');
fprintf('MATLAB 版本 - NSCT 分解和重建\n');
fprintf('======================================================================\n\n');

%% 1. 加载图像
fprintf('1. 加载图像...\n');
img = imread('test_image.jpg');
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = double(img);
fprintf('   图像尺寸: %dx%d\n', size(img, 1), size(img, 2));
fprintf('   像素值范围: [%.2f, %.2f]\n', min(img(:)), max(img(:)));

%% 2. 设置参数
levels = [2, 3];  % 2个金字塔层级，每层分别进行2和3级方向分解
dfilt = 'dmaxflat7';  % 方向滤波器
pfilt = 'maxflat';  % 金字塔滤波器

% 创建输出文件夹结构: output/levels_X_Y_dfilt_pfilt/matlab/
levels_str = strjoin(arrayfun(@num2str, levels, 'UniformOutput', false), '_');
output_dir = fullfile('output', sprintf('levels_%s_%s_%s', levels_str, dfilt, pfilt), 'matlab');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
fprintf('\n输出目录: %s\n', output_dir);

fprintf('\n2. NSCT 分解参数:\n');
fprintf('   金字塔层级: %d\n', length(levels));
fprintf('   方向分解层级: [%s]\n', num2str(levels));
fprintf('   方向滤波器: %s\n', dfilt);
fprintf('   金字塔滤波器: %s\n', pfilt);
fprintf('   预计子带数: 1个低频 + %d个方向(尺度1) + %d个方向(尺度2)\n', ...
    2^levels(1), 2^levels(2));

%% 3. NSCT 分解
fprintf('\n3. 执行 NSCT 分解...\n');
tic;
y = nsctdec(img, levels, dfilt, pfilt);
dec_time = toc;
fprintf('   分解完成，耗时: %.3f 秒\n', dec_time);

%% 4. 显示分解结果信息
fprintf('\n4. 分解结果:\n');
fprintf('   总子带数: %d\n', length(y));
fprintf('   - y{1} (低频): %dx%d\n', size(y{1}, 1), size(y{1}, 2));
for i = 2:length(y)
    if iscell(y{i})
        fprintf('   - y{%d} (尺度%d方向子带): %d个子带，每个尺寸 %dx%d\n', ...
            i, i-1, length(y{i}), size(y{i}{1}, 1), size(y{i}{1}, 2));
    else
        fprintf('   - y{%d} (尺度%d): %dx%d\n', ...
            i, i-1, size(y{i}, 1), size(y{i}, 2));
    end
end

%% 5. NSCT 重建
fprintf('\n5. 执行 NSCT 重建...\n');
tic;
img_rec = nsctrec(y, dfilt, pfilt);
rec_time = toc;
fprintf('   重建完成，耗时: %.3f 秒\n', rec_time);
fprintf('   重建图像尺寸: %dx%d\n', size(img_rec, 1), size(img_rec, 2));

%% 6. 重建质量评估
fprintf('\n6. 重建质量评估:\n');
mse = mean((img(:) - img_rec(:)).^2);
if mse > 0
    psnr = 10 * log10(255^2 / mse);
else
    psnr = Inf;
end
max_error = max(abs(img(:) - img_rec(:)));
relative_error = norm(img(:) - img_rec(:)) / norm(img(:)) * 100;

fprintf('   均方误差 (MSE): %.6e\n', mse);
fprintf('   峰值信噪比 (PSNR): %.2f dB\n', psnr);
fprintf('   最大绝对误差: %.6e\n', max_error);
fprintf('   相对误差: %.6f%%\n', relative_error);

%% 7. 保存结果到 .mat 文件
fprintf('\n7. 保存结果到文件...\n');

% 准备保存的数据结构
results = struct();
results.original_image = img;
results.reconstructed_image = img_rec;
results.decomposition = y;
results.parameters = struct('levels', levels, 'dfilt', dfilt, 'pfilt', pfilt);
results.timing = struct('decomposition_time', dec_time, 'reconstruction_time', rec_time);
results.metrics = struct('mse', mse, 'psnr', psnr, 'max_error', max_error, 'relative_error', relative_error);

results_file = fullfile(output_dir, 'matlab_nsct_results.mat');
save(results_file, 'results', '-v7.3');
fprintf('   结果已保存到: %s\n', results_file);

%% 8. 保存详细的子带信息
fprintf('\n8. 保存详细的子带信息...\n');

subband_info = struct();

% 低频子带
subband_info.lowpass = struct();
subband_info.lowpass.shape = size(y{1});
subband_info.lowpass.data = y{1};
subband_info.lowpass.mean = mean(y{1}(:));
subband_info.lowpass.std = std(y{1}(:));
subband_info.lowpass.min = min(y{1}(:));
subband_info.lowpass.max = max(y{1}(:));

% 方向子带
subband_info.bandpass = cell(length(y)-1, 1);

for i = 2:length(y)
    scale_idx = i - 1;
    
    if iscell(y{i})
        % 有方向分解
        scale_info = struct();
        scale_info.scale = scale_idx;
        scale_info.num_directions = length(y{i});
        scale_info.directions = cell(length(y{i}), 1);
        
        for j = 1:length(y{i})
            dir_info = struct();
            dir_info.direction = j - 1;  % 0-based for Python compatibility
            dir_info.shape = size(y{i}{j});
            dir_info.data = y{i}{j};
            dir_info.mean = mean(y{i}{j}(:));
            dir_info.std = std(y{i}{j}(:));
            dir_info.min = min(y{i}{j}(:));
            dir_info.max = max(y{i}{j}(:));
            scale_info.directions{j} = dir_info;
        end
        
        subband_info.bandpass{scale_idx} = scale_info;
    else
        % 无方向分解
        scale_info = struct();
        scale_info.scale = scale_idx;
        scale_info.num_directions = 0;
        scale_info.shape = size(y{i});
        scale_info.data = y{i};
        scale_info.mean = mean(y{i}(:));
        scale_info.std = std(y{i}(:));
        scale_info.min = min(y{i}(:));
        scale_info.max = max(y{i}(:));
        subband_info.bandpass{scale_idx} = scale_info;
    end
end

subband_file = fullfile(output_dir, 'matlab_subband_details.mat');
save(subband_file, 'subband_info', '-v7.3');
fprintf('   子带详细信息已保存到: %s\n', subband_file);

fprintf('\n======================================================================\n');
fprintf('MATLAB 版本完成！\n');
fprintf('======================================================================\n');
