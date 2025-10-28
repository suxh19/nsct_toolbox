% Test script for nsdfbdec function
% Tests nonsubsampled directional filter bank decomposition
% Results are saved to test_nsdfbdec_results.mat for Python comparison

fprintf('=== Testing nsdfbdec function ===\n\n');

% Test 1: Level 0 (no decomposition) - 32x32 image
fprintf('Test 1: Level 0 decomposition (32x32 image)\n');
rng(42);
x1 = rand(32, 32);
clevels1 = 0;
y1 = nsdfbdec(x1, 'pkva', clevels1);
fprintf('Input size: %dx%d\n', size(x1, 1), size(x1, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y1), 2^clevels1);
fprintf('Subband 1 size: %dx%d\n', size(y1{1}, 1), size(y1{1}, 2));

% Test 2: Level 1 decomposition - 64x64 image
fprintf('\nTest 2: Level 1 decomposition (64x64 image)\n');
x2 = rand(64, 64);
clevels2 = 1;
y2 = nsdfbdec(x2, 'pkva', clevels2);
fprintf('Input size: %dx%d\n', size(x2, 1), size(x2, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y2), 2^clevels2);
for i = 1:length(y2)
    fprintf('  Subband %d size: %dx%d\n', i, size(y2{i}, 1), size(y2{i}, 2));
end

% Test 3: Level 2 decomposition - 64x64 image
fprintf('\nTest 3: Level 2 decomposition (64x64 image)\n');
x3 = rand(64, 64);
clevels3 = 2;
y3 = nsdfbdec(x3, 'pkva', clevels3);
fprintf('Input size: %dx%d\n', size(x3, 1), size(x3, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y3), 2^clevels3);
for i = 1:length(y3)
    fprintf('  Subband %d size: %dx%d\n', i, size(y3{i}, 1), size(y3{i}, 2));
end

% Test 4: Level 3 decomposition - 128x128 image
fprintf('\nTest 4: Level 3 decomposition (128x128 image)\n');
x4 = rand(128, 128);
clevels4 = 3;
y4 = nsdfbdec(x4, 'pkva', clevels4);
fprintf('Input size: %dx%d\n', size(x4, 1), size(x4, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y4), 2^clevels4);
for i = 1:length(y4)
    fprintf('  Subband %d size: %dx%d\n', i, size(y4{i}, 1), size(y4{i}, 2));
end

% Test 5: Level 2 with dmaxflat7 filter - 64x64 image
fprintf('\nTest 5: Level 2 with dmaxflat7 filter (64x64 image)\n');
x5 = rand(64, 64);
clevels5 = 2;
y5 = nsdfbdec(x5, 'dmaxflat7', clevels5);
fprintf('Input size: %dx%d\n', size(x5, 1), size(x5, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y5), 2^clevels5);

% Test 6: Non-square image - 64x96, level 2
fprintf('\nTest 6: Non-square image (64x96), level 2\n');
x6 = rand(64, 96);
clevels6 = 2;
y6 = nsdfbdec(x6, 'pkva', clevels6);
fprintf('Input size: %dx%d\n', size(x6, 1), size(x6, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y6), 2^clevels6);

% Test 7: Energy conservation check - level 2
fprintf('\nTest 7: Energy conservation check (64x64, level 2)\n');
x7 = rand(64, 64);
clevels7 = 2;
y7 = nsdfbdec(x7, 'pkva', clevels7);
energy_in = sum(x7(:).^2);
energy_out = 0;
for i = 1:length(y7)
    energy_out = energy_out + sum(y7{i}(:).^2);
end
fprintf('Input energy: %.6f\n', energy_in);
fprintf('Output energy (sum of all subbands): %.6f\n', energy_out);
fprintf('Energy ratio: %.6f\n', energy_out / energy_in);

% Test 8: Level 4 decomposition - 256x256 image
fprintf('\nTest 8: Level 4 decomposition (256x256 image)\n');
x8 = rand(256, 256);
clevels8 = 4;
y8 = nsdfbdec(x8, 'pkva', clevels8);
fprintf('Input size: %dx%d\n', size(x8, 1), size(x8, 2));
fprintf('Number of subbands: %d (expected: %d)\n', length(y8), 2^clevels8);

% Test 9: Small image - 32x32, level 1
fprintf('\nTest 9: Small image (32x32), level 1\n');
x9 = rand(32, 32);
clevels9 = 1;
y9 = nsdfbdec(x9, 'pkva', clevels9);
fprintf('Input size: %dx%d\n', size(x9, 1), size(x9, 2));
fprintf('Number of subbands: %d\n', length(y9));

% Test 10: Constant image - 64x64, level 2
fprintf('\nTest 10: Constant image (64x64), level 2\n');
x10 = ones(64, 64) * 3.14159;
clevels10 = 2;
y10 = nsdfbdec(x10, 'pkva', clevels10);
fprintf('Input: constant value %.5f\n', x10(1,1));
fprintf('Number of subbands: %d\n', length(y10));
for i = 1:length(y10)
    fprintf('  Subband %d mean: %.6f, std: %.6f\n', i, mean(y10{i}(:)), std(y10{i}(:)));
end

% Save all results
fprintf('\nSaving results to test_nsdfbdec_results.mat...\n');

% Convert cell arrays to structure for easier Python loading
results = struct();
results.x1 = x1; results.clevels1 = clevels1;
results.x2 = x2; results.clevels2 = clevels2;
results.x3 = x3; results.clevels3 = clevels3;
results.x4 = x4; results.clevels4 = clevels4;
results.x5 = x5; results.clevels5 = clevels5;
results.x6 = x6; results.clevels6 = clevels6;
results.x7 = x7; results.clevels7 = clevels7;
results.x8 = x8; results.clevels8 = clevels8;
results.x9 = x9; results.clevels9 = clevels9;
results.x10 = x10; results.clevels10 = clevels10;

% Save y1 (single element)
results.y1_1 = y1{1};

% Save y2 (2 elements)
results.y2_1 = y2{1}; results.y2_2 = y2{2};

% Save y3 (4 elements)
results.y3_1 = y3{1}; results.y3_2 = y3{2};
results.y3_3 = y3{3}; results.y3_4 = y3{4};

% Save y4 (8 elements)
for i = 1:8
    results.(sprintf('y4_%d', i)) = y4{i};
end

% Save y5 (4 elements)
for i = 1:4
    results.(sprintf('y5_%d', i)) = y5{i};
end

% Save y6 (4 elements)
for i = 1:4
    results.(sprintf('y6_%d', i)) = y6{i};
end

% Save y7 (4 elements)
for i = 1:4
    results.(sprintf('y7_%d', i)) = y7{i};
end
results.energy_in = energy_in;
results.energy_out = energy_out;

% Save y8 (16 elements)
for i = 1:16
    results.(sprintf('y8_%d', i)) = y8{i};
end

% Save y9 (2 elements)
results.y9_1 = y9{1}; results.y9_2 = y9{2};

% Save y10 (4 elements)
for i = 1:4
    results.(sprintf('y10_%d', i)) = y10{i};
end

save('data/test_nsdfbdec_results.mat', '-struct', 'results');

fprintf('Done! All tests completed.\n');
