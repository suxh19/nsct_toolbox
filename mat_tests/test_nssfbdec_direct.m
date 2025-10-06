% Test nssfbdec directly with upsampling matrix
clear; clc;

% Test image
x = rand(64, 64);

% Get filters
[h1, h2] = dfilters('pkva', 'd');
h1 = h1 / sqrt(2);
h2 = h2 / sqrt(2);
k1 = modulate2(h1, 'c');
k2 = modulate2(h2, 'c');

% Quincunx upsampling matrix
q1 = [1, -1; 1, 1];

% No upsampling case
fprintf('=== No upsampling ===\n');
[y1_no, y2_no] = nssfbdec(x, k1, k2);
fprintf('y1 size: %d x %d, mean: %.6f\n', size(y1_no, 1), size(y1_no, 2), mean(y1_no(:)));
fprintf('y2 size: %d x %d, mean: %.6f\n', size(y2_no, 1), size(y2_no, 2), mean(y2_no(:)));

% With upsampling (quincunx)
fprintf('\n=== With quincunx upsampling ===\n');
[y1_q, y2_q] = nssfbdec(x, k1, k2, q1);
fprintf('y1 size: %d x %d, mean: %.6f\n', size(y1_q, 1), size(y1_q, 2), mean(y1_q(:)));
fprintf('y2 size: %d x %d, mean: %.6f\n', size(y2_q, 1), size(y2_q, 2), mean(y2_q(:)));

% Save for Python comparison
save('../data/test_nssfbdec_direct.mat', 'x', 'k1', 'k2', 'q1', ...
     'y1_no', 'y2_no', 'y1_q', 'y2_q');

fprintf('\nData saved to test_nssfbdec_direct.mat\n');
