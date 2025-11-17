clc
clear
% 原始符号时间点和对应的信道增益
time_points_symbols = [0, 1, 2, 3, 4.5];
h_sequence_per_symbol = [1, 0.8, 0.6, 0.4, 0.3];

% 需要插值的时间点（例如过采样时间点）
time_points_stream = 0:0.2:4.5;

% 使用线性插值进行计算
h_magnitude_upsampled = interp1(time_points_symbols, abs(h_sequence_per_symbol), time_points_stream, 'linear');

% 显示插值结果
disp(h_magnitude_upsampled);
