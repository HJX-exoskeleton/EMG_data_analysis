% 《肌电信号处理入门》
%% 认识肌肉电信号
% 3类肌肉: 骨骼肌\心肌\平滑肌
% 肌肉电信号是随机的
% 时域：有强有弱 ； 频域：60-80Hz


%% 肌电信号的采集
% 肌电设备，传感器
% 工作电极（肌腹），参考电极（肘关节），地电极（设备接地）


%% 
% 肌电信号预处理
% 干扰：工频噪声，运动伪迹，心电噪声
load('Wear_bend.mat'); %导入数据%%%%%%%%%%%%%%%%%%%%%%
figure(1);
plot(Wear_bend,'DisplayName','Wear_bend');
Fs = 1000; %设置采样频率

fdatool; % 打开设计滤波器的工具箱
% 首先点击Bandpass, 再电机IIR, 然后在Options中选择passband, Fs设置为1000, 
% Fstop1设置为20, Fpass1设置为30,  Fpass2设置为350, Fstop2设置为400, 设置完成
% 在File中点击Export, 在Export As 中改为Objects，在Variable Name中改为filt_bp, 并勾选下方


%% 下面开始滤波

Wear_bend_bp =  filter(filt_bp,Wear_bend); % 带通滤波 30-350
figure(2);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp');


%% 设置50Hz陷波滤波器(可有可不有)
% 在fdatool工具箱中Response Type中选最后一行中的Notching, 在Design Method中的IIR中选择Single Notch
% Fs设置为1000, Fnotch设置为50, 再设置下方的Q为60, 设置完成
% 在File中点击Export, 在Export As 中改为Objects，在Variable Name中改为filt_50, 并勾选下方
Wear_bend_50 =  filter(filt_50,Wear_bend_bp); % 50Hz陷波
figure(3);
subplot(121);
plot(Wear_bend_bp);
subplot(122);
plot(Wear_bend_50);

%50Hz陷波与带通滤波比对,看50Hz陷波滤波器有无发挥作用
figure(4);
plot(Wear_bend_bp(:,1));
hold on
plot(Wear_bend_50(:,1));
hold off
%若发现50Hz陷波与带通滤波差不多,没有带来提升,则可以忽略50Hz陷波这一操作


%% 画纯净肌电信号
figure(5);
plot(Wear_bend_bp(:,1));
hold on
plot(Wear_bend_bp(:,2)); 
hold off


%% 下面使用APP中的Signal Analyzer
% 界面化滤波（新手友善，无需写代码）
% 详情见B站UP: https://www.bilibili.com/video/BV1Eg411S7LC/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93


%% 肌电信号时域特征
figure(6);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp'); % 选取数据点后导入工作区, 找8个点
% 最大值，最小值，均方根值RMS，绝对值平均值，包络
emg1 = Wear_bend_bp(:,1);
emg2 = Wear_bend_bp(:,2);
inde = [cursor_info.DataIndex];

% 时域特征——最大值
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.Max(i,j) = max(emg(star:term));
    end
end

% 时域特征——最小值
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.Min(i,j) = min(emg(star:term));
    end
end


% 时域特征——均方根值RMS
% 均方根值稳定性强
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.RMS(i,j) = rms(emg(star:term));
    end
end


% 时域特征——绝对值平均值
for i = 1:2
    emg = abs(Wear_bend_bp(:,i));
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.ABS_mean(i,j) = mean(emg(star:term));
    end
end


% 时域特征——包络
figure(7); % 绘制整段包络线
subplot(211);
[upper1,lower1] = envelope(emg1,150,'rms');
plot(emg1); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); hold off
subplot(212);
[upper2,lower2] = envelope(emg2,150,'rms');
plot(emg2); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); hold off

% 单独抽取一段信号
figure(8); % 绘制抽取的一段信号的包络线
emg1_seg = emg1(inde(8):inde(1)); % 指针 注意是有8个数据点
subplot(211);
[upper1,lower1] = envelope(emg1_seg,150,'rms');
plot(emg1_seg); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); hold off
emg2_seg = emg2(inde(8):inde(1)); % 指针
subplot(212);
[upper2,lower2] = envelope(emg2_seg,150,'rms');
plot(emg2_seg); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); hold off

% 将2个通道包络线绘制在一起
figure(9); % 仅仅绘制包络线
plot(upper1,'LineWidth',1.5); hold on; plot(upper2,'LineWidth',1.5); hold on; 
plot(lower1,'LineWidth',1.5); hold on; plot(lower2,'LineWidth',1.5); hold off; 


%% 肌电信号频域特征
inde = [cursor_info.DataIndex];
seg = 1; % seg = 1表示的是第一段信号, seg可以修改 %%%%%%%%%%%%%%%%%%%
init = inde(2*seg);
term = inde(2*seg-1);
ch1 = Wear_bend_bp(init:term,1);
ch2 = Wear_bend_bp(init:term,2);
L = length(ch1);
Fs = 1000;

% fft  快速傅里叶变换 , 频谱信息 , 幅频图
% ch1与ch2的幅频图比对
% ch1的fft    通道1
ch1_fft = fft(ch1);
ch1_P2 = abs(ch1_fft/L);
ch1_P1 = ch1_P2(1:L/2+1);
ch1_P1(2:end-1) = 2*ch1_P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure(10);plot(f,ch1_P1);hold on;

% ch2的fft    通道2
ch2_fft = fft(ch2);
ch2_P2 = abs(ch2_fft/L);
ch2_P1 = ch2_P2(1:L/2+1);
ch2_P1(2:end-1) = 2*ch2_P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,ch2_P1);hold off;

% 频域幅值平均值
FDF.AF_AM(1,seg) = mean(ch1_P1);
FDF.AF_AM(2,seg) = mean(ch2_P1);

% 重心频率
FDF.AF_CF(1,seg) = sum(f'.*ch1_P1)/sum(ch1_P1);
FDF.AF_CF(2,seg) = sum(f'.*ch2_P1)/sum(ch2_P1);

% 频率方差
FDF.AF_FVAR(1,seg) = sum(((f-FDF.AF_CF(1,seg)).^2)'.*ch1_P1)/sum(ch1_P1);
FDF.AF_FVAR(2,seg) = sum(((f-FDF.AF_CF(2,seg)).^2)'.*ch2_P1)/sum(ch2_P1);

% 中值频率 , 基于功率谱图
% 计算ch1的中值频率
temp_power = ch1_P1.*ch1_P1;
TP = sum(temp_power);
TP_half_l = 0;
TP_half_r = 0;

for i = 1:L-1
    TP_half_l =TP_half_l + temp_power(i);
    TP_half_r =TP_half_l + temp_power(i+1);
    if TP_half_l == TP/2
        FDF.mdf(1,seg) = f(i);
        break
    elseif TP_half_l < TP/2 && TP_half_r > TP/2
        FDF.mdf(1,seg) = (f(i)+f(i+1))/2;
        break;
    else
        continue
    end
end

% 计算ch2的中值频率
temp_power = ch2_P1.*ch2_P1;
TP = sum(temp_power);
TP_half_l = 0;
TP_half_r = 0;

for i = 1:L-1
    TP_half_l =TP_half_l + temp_power(i);
    TP_half_r =TP_half_l + temp_power(i+1);
    if TP_half_l == TP/2
        FDF.mdf(2,seg) = f(i);
        break
    elseif TP_half_l < TP/2 && TP_half_r > TP/2
        FDF.mdf(2,seg) = (f(i)+f(i+1))/2;
        break;
    else
        continue
    end
end


%% 肌电信号时频阈特征
% 方法：（1）APP工具箱 ——信号分析器(Signal Analyzer)； （2）spectrogram函数
% 方法（1）
% B站链接: https://www.bilibili.com/video/BV1ye4y1q7v3/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93

% 方法（2）
figure(11);
spectrogram(Wear_bend_1,256,250,1000,1000,'yaxis'); % Wear_bend_1为处理后的第1列数据，可以先用Signal Analyzer处理再导出Wear_bend_1
figure(12);
spectrogram(Wear_bend_2,256,250,1000,1000,'yaxis'); % Wear_bend_2为处理后的第2列数据，可以先用Signal Analyzer处理再导出Wear_bend_2

        