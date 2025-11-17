close all; 
clc; 
clear;

% --- 基本参数 ---
qpsk_code = [1+1j, 1-1j, -1+1j, -1-1j] ./ sqrt(2); % 归一化使每个符号的功率为1
SNR_dB = -5:1:10;  % 信噪比范围
SNR = 10.^(SNR_dB / 10);  % 将SNR从dB转换为线性值
osr = 16;  % 过采样率
filter_taps = 129;  % 根升余弦滤波器的抽头数
rolloff = 0.25;  % 滚降因子

% --- 预分配误码率矩阵 ---
ber_Mu = zeros(length(SNR), 1);

% --- 根升余弦滤波器设计 ---
rrc_filter = rcosdesign(rolloff, filter_taps, osr, 'sqrt');  % 根升余弦滤波器
num_symbols = 2.5e4;  % 设置符号数

time_points_symbols = 1:1:num_symbols;  % 时间序列
time_points_stream = 1:1/16:num_symbols+15/16;

for snr_N = 1:length(SNR)
    % --- 随机比特生成 ---
    qpsk_bit = randi([0, 1], 2*num_symbols, 1);  % 随机生成比特流
    qpsk_bit2 = reshape(qpsk_bit, 2, []);  % 每两个比特一组

    % --- 比特到符号映射 ---
    data_signal = qpsk_code([2, 1] * qpsk_bit2 + 1);  % 比特到符号的映射

    % --- 上采样 ---
    upsampled_signal = upsample(data_signal, osr);  % 上采样（每个符号插入 osr-1 个零）

    % --- 脉冲成型 ---
    shaped_signal = conv(upsampled_signal, rrc_filter, 'same');  % 通过根升余弦滤波器进行脉冲成型

    %% 瑞利信道
    % --- 生成瑞利衰落信道系数 --- 
    X = randn(num_symbols, 1);  % I分量
    Y = randn(num_symbols, 1);  % Q分量

    % 计算瑞利信道幅度
    h_magnitude = sqrt(X.^2 + Y.^2);

    % 生成均匀分布的随机相位
    h_phase = 2 * pi * rand(num_symbols, 1);
    
    % 生成复数瑞利信道系数
    h_channel = h_magnitude .* exp(1i * h_phase);

    h_channel = h_channel';
    %线性插值将hc扩展到样点维度
    h_channel_upsampled = interp1(time_points_symbols, h_channel, time_points_stream, 'linear');
    h_channel_upsampled(end-14:end) = 0;
    % --- 将信号通过瑞利衰落信道 ---
    faded_signal = shaped_signal .* h_channel_upsampled;  % 应用信道衰落
    
    % --- 添加噪声并计算接收信号 ---
    noise_var = 10^(-SNR_dB(snr_N)/10);  % 计算噪声方差
    R_sig = awgn(faded_signal, (SNR_dB(snr_N)-10*log10(16)), 'measured');  % 使用SNR_dB与shaped_signal作为参考计算噪声

    % --- 匹配滤波 ---
    matched_signal = conv(R_sig, rrc_filter, 'same');  % 匹配滤波，使用与发射端相同的滤波器

    % --- 下采样 ---
    received_signal = downsample(matched_signal, osr);  % 下采样（还原到原始符号率）

    

    % --- 判决过程: 使用矩阵操作进行符号判决 ---
    % 计算接收信号与QPSK符号的距离
    [~, position] = min(abs(received_signal.' - qpsk_code), [], 2);  % 向量化计算，减少内层循环

    % --- 比特解调 ---
    bit_dec = zeros(length(qpsk_bit), 1);
    bit_dec(1:2:end) = floor((position - 1) / 2);  % 取符号的第一个比特
    bit_dec(2:2:end) = mod((position - 1), 2);  % 取符号的第二个比特

    % --- 计算误码率 ---
    err_bit = sum(bit_dec ~= qpsk_bit);  % 计算错误比特数
    ber_Mu(snr_N, :) = err_bit / length(qpsk_bit);  % 计算每次仿真的误码率
end

% --- 理论BER计算 ---
BER = 1 / 2 * erfc(sqrt(SNR / 2));  % 理论误码率

% --- 绘制仿真和理论BER ---
semilogy(SNR_dB, mean(ber_Mu, 2), 'ro-');  % 绘制仿真误码率
hold on;
grid on; 
semilogy(SNR_dB, BER, 'bo-');  % 绘制理论BER
legend('QPSK仿真', '理论误码率');
xlabel('E_b/N_0(dB)');
ylabel('BER');
