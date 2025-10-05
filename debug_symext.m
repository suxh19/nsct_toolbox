% 调试 symext 函数
clear all;

x = [1 2 3; 4 5 6; 7 8 9];  % 3x3 矩阵
h = ones(3, 3);  % 3x3 滤波器
shift = [1, 1];

[m,n] = size(x);
[p,q] = size(h);

fprintf('x 尺寸: [%d, %d]\n', m, n);
fprintf('h 尺寸: [%d, %d]\n', p, q);

p2=floor(p/2);q2=floor(q/2);
s1=shift(1);s2=shift(2);

fprintf('p2=%d, q2=%d\n', p2, q2);
fprintf('s1=%d, s2=%d\n', s1, s2);

ss = p2 - s1 + 1;
rr = q2 - s2 + 1;

fprintf('ss=%d, rr=%d\n', ss, rr);
fprintf('n-p-s1+1 = %d-%d-%d+1 = %d\n', n, p, s1, n-p-s1+1);
fprintf('m-q-s2+1 = %d-%d-%d+1 = %d\n', m, q, s2, m-q-s2+1);

% 检查索引范围
if ss <= n && ss > 0
    fprintf('第一个水平扩展索引 1:ss = 1:%d 有效\n', ss);
else
    fprintf('第一个水平扩展索引 1:ss = 1:%d 无效！\n', ss);
end

end_idx = n-p-s1+1;
if end_idx >= 1 && end_idx <= n
    fprintf('第二个水平扩展索引 n:-1:%d = %d:-1:%d 有效\n', end_idx, n, end_idx);
else
    fprintf('第二个水平扩展索引 n:-1:%d = %d:-1:%d 无效！\n', end_idx, n, end_idx);
end
