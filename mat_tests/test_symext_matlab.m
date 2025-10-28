% Test script for symext function
% Tests symmetric extension with various inputs
% Results are saved to test_symext_results.mat for Python comparison

fprintf('=== Testing symext function ===\n\n');

% Test 1: Basic 4x4 image with 3x3 filter
fprintf('Test 1: Basic 4x4 image with 3x3 filter\n');
x1 = reshape(0:15, 4, 4)';  % 4x4 matrix
h1 = ones(3, 3);
shift1 = [1, 1];
y1 = symext(x1, h1, shift1);
fprintf('Input size: %dx%d\n', size(x1, 1), size(x1, 2));
fprintf('Filter size: %dx%d\n', size(h1, 1), size(h1, 2));
fprintf('Output size: %dx%d\n', size(y1, 1), size(y1, 2));
fprintf('Expected size: %dx%d\n', size(x1, 1) + size(h1, 1) - 1, size(x1, 2) + size(h1, 2) - 1);

% Test 2: 5x5 image with 5x5 filter (use smaller shift to avoid negative indices)
fprintf('\nTest 2: 5x5 image with 5x5 filter\n');
x2 = magic(5);
h2 = ones(5, 5);
shift2 = [0, 0];  % Changed from [2,2] to avoid negative indices
y2 = symext(x2, h2, shift2);
fprintf('Output size: %dx%d\n', size(y2, 1), size(y2, 2));

% Test 3: Non-square image
fprintf('\nTest 3: Non-square image (6x4)\n');
x3 = reshape(1:24, 6, 4)';
h3 = ones(3, 3);
shift3 = [1, 1];
y3 = symext(x3, h3, shift3);
fprintf('Input size: %dx%d\n', size(x3, 1), size(x3, 2));
fprintf('Output size: %dx%d\n', size(y3, 1), size(y3, 2));

% Test 4: Different shift values
fprintf('\nTest 4: Different shift values [0, 1]\n');
x4 = reshape(1:16, 4, 4)';
h4 = ones(3, 3);
shift4 = [0, 1];
y4 = symext(x4, h4, shift4);
fprintf('Output size: %dx%d\n', size(y4, 1), size(y4, 2));

% Test 5: Negative shift
fprintf('\nTest 5: Negative shift [-1, -1]\n');
x5 = reshape(1:16, 4, 4)';
h5 = ones(3, 3);
shift5 = [-1, -1];
y5 = symext(x5, h5, shift5);
fprintf('Output size: %dx%d\n', size(y5, 1), size(y5, 2));

% Test 6: Large filter (7x7) with larger image
fprintf('\nTest 6: Large filter (7x7)\n');
x6 = reshape(1:64, 8, 8)';  % Use 8x8 image instead of 4x4
h6 = ones(7, 7);
shift6 = [1, 1];  % Conservative shift
y6 = symext(x6, h6, shift6);
fprintf('Output size: %dx%d\n', size(y6, 1), size(y6, 2));

% Test 7: Reasonably-sized 6x6 image with 3x3 filter
fprintf('\nTest 7: 6x6 image with 3x3 filter\n');
x7 = reshape(1:36, 6, 6)';
h7 = ones(3, 3);
shift7 = [1, 1];
y7 = symext(x7, h7, shift7);
fprintf('Output size: %dx%d\n', size(y7, 1), size(y7, 2));

% Test 8: Non-uniform filter (3x5) with safe shift
fprintf('\nTest 8: Non-uniform filter (3x5)\n');
x8 = reshape(1:36, 6, 6)';  % Larger image
h8 = ones(3, 5);
shift8 = [1, 1];  % More conservative shift
y8 = symext(x8, h8, shift8);
fprintf('Output size: %dx%d\n', size(y8, 1), size(y8, 2));

% Test 9: Non-uniform filter (5x3) with safe shift
fprintf('\nTest 9: Non-uniform filter (5x3)\n');
x9 = reshape(1:36, 6, 6)';  % Larger image
h9 = ones(5, 3);
shift9 = [1, 1];  % More conservative shift
y9 = symext(x9, h9, shift9);
fprintf('Output size: %dx%d\n', size(y9, 1), size(y9, 2));

% Test 10: Random values
fprintf('\nTest 10: Random values\n');
rng(42);  % Set seed for reproducibility
x10 = rand(8, 8);
h10 = ones(3, 3);
shift10 = [1, 1];
y10 = symext(x10, h10, shift10);
fprintf('Output size: %dx%d\n', size(y10, 1), size(y10, 2));

% Test 11: Verify symmetry property
fprintf('\nTest 11: Verify symmetry - check boundaries\n');
x11 = reshape(1:16, 4, 4)';
h11 = ones(3, 3);
shift11 = [1, 1];
y11 = symext(x11, h11, shift11);
fprintf('Top-left corner of original: %.2f\n', x11(1, 1));
fprintf('Extended left edge (should reflect): %.2f\n', y11(2, 1));
fprintf('Extended top edge (should reflect): %.2f\n', y11(1, 2));

% Test 12: Edge case - minimum filter size (1x1)
fprintf('\nTest 12: Minimum filter size (1x1)\n');
x12 = reshape(1:16, 4, 4)';
h12 = ones(1, 1);
shift12 = [0, 0];
y12 = symext(x12, h12, shift12);
fprintf('Output size: %dx%d (should be same as input)\n', size(y12, 1), size(y12, 2));

% Save all results
fprintf('\nSaving results to test_symext_results.mat...\n');
save('../data/test_symext_results.mat', ...
    'x1', 'h1', 'shift1', 'y1', ...
    'x2', 'h2', 'shift2', 'y2', ...
    'x3', 'h3', 'shift3', 'y3', ...
    'x4', 'h4', 'shift4', 'y4', ...
    'x5', 'h5', 'shift5', 'y5', ...
    'x6', 'h6', 'shift6', 'y6', ...
    'x7', 'h7', 'shift7', 'y7', ...
    'x8', 'h8', 'shift8', 'y8', ...
    'x9', 'h9', 'shift9', 'y9', ...
    'x10', 'h10', 'shift10', 'y10', ...
    'x11', 'h11', 'shift11', 'y11', ...
    'x12', 'h12', 'shift12', 'y12');

fprintf('Done! All tests completed.\n');
