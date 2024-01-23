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
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);
Fs = 1000; %设置采样频率

fdatool; % 打开设计滤波器的工具箱
% 首先点击Bandpass, 再点击IIR, 然后在Options中选择passband, Fs设置为1000, 
% Fstop1设置为20, Fpass1设置为30,  Fpass2设置为350, Fstop2设置为400, 设置完成
% 在File中点击Export, 在Export As 中改为Objects，在Variable Name中改为filt_bp, 并勾选下方


%% 下面开始滤波

Wear_bend_bp =  filter(filt_bp,Wear_bend); % 带通滤波 30-350
figure(2);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp');
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);

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
%plot(Wear_bend_bp(:,2)+400);  % 错位展示2个采集信号的波形
hold off
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);

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
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); legend('Collected value','Upper envelope of the collected value','Lower envelope of the acquired value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off
subplot(212);
[upper2,lower2] = envelope(emg2,150,'rms');
plot(emg2); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); legend('Power value','Upper envelope of the power value','Lower envelope of power value','FontSize',8);xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);hold off

% 单独抽取一段信号
figure(8); % 绘制抽取的一段信号的包络线
emg1_seg = emg1(inde(8):inde(1)); % 指针 注意是有8个数据点
subplot(211);
[upper1,lower1] = envelope(emg1_seg,150,'rms');
plot(emg1_seg); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); legend('Collected value','The upper envelope of the collected value','Lower envelope of the acquired value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off
emg2_seg = emg2(inde(8):inde(1)); % 指针
subplot(212);
[upper2,lower2] = envelope(emg2_seg,150,'rms');
plot(emg2_seg); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); legend('Power value','Upper envelope of the power value','Lower envelope of power value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off

% 将2个通道包络线绘制在一起
figure(9); % 仅仅绘制包络线
plot(upper1,'LineWidth',1.5); hold on; plot(upper2,'LineWidth',1.5); hold on; 
plot(lower1,'LineWidth',1.5); hold on; plot(lower2,'LineWidth',1.5); 
legend('Upper envelope of the collected value','Upper envelope of the power value','Lower envelope of the acquired value','Lower envelope of power value','FontSize',8); 
xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off; 

%% 肌电信号频域特征
inde = [cursor_info.DataIndex];
seg = 1; % seg = 1表示的是第一段信号, seg可以修改 %%%%%%%%%%%%%%%%%%%  Notice: 8个点提取4段，seg = 1, 2, 3, 4
% 频域特征这里seg = 1, 2, 3, 4依次赋值执行整段代码行即可, 变量在工作区中可见——FDF       %%%%%%%%%%%%%%%%%%% 
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
figure(10);plot(f,ch1_P1);
legend('Acquisition value (first paragraph)','Power value (first paragraph)','FontSize',12); % 第一段表示 seg = 1，第二段表示 seg = 2，。。。。。
xlabel('Frequency (Hz)','FontSize',12); ylabel('Maginitude (dB)','FontSize',12);hold on;

% ch2的fft    通道2
ch2_fft = fft(ch2);
ch2_P2 = abs(ch2_fft/L);
ch2_P1 = ch2_P2(1:L/2+1);
ch2_P1(2:end-1) = 2*ch2_P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,ch2_P1);
legend('Acquisition value (first paragraph)','Power value (first paragraph)','FontSize',12); % 第一段表示 seg = 1，第二段表示 seg = 2，。。。。。
xlabel('Frequency (Hz)','FontSize',12); ylabel('Maginitude (dB)','FontSize',12);hold off;



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

% 方法（1）原始信号导入
% B站链接: https://www.bilibili.com/video/BV1ye4y1q7v3/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93

% 方法（2）数据变量需要用Signal Analyzer工具箱导出
figure(11);
spectrogram(Wear_bend_1,256,250,1000,1000,'yaxis'); % Wear_bend_1为处理后的第1列数据，可以先用Signal Analyzer处理再导出Wear_bend_1
figure(12);
spectrogram(Wear_bend_2,256,250,1000,1000,'yaxis'); % Wear_bend_2为处理后的第2列数据，可以先用Signal Analyzer处理再导出Wear_bend_2


%% 全面剖析spectrogram函数
% 视频链接：https://www.bilibili.com/video/BV14F41157Gz/?spm_id_from=333.999.0.0&vd_source=76446368a03ef22caf531595d665cb93

% Part1------spectrogram函数是啥?
% spectrogram函数其实就是--------短时傅里叶变换（STFT）

% Part2------为什么要用到spectrogram函数?
% 因为傅里叶变换（FFT）无法告诉我们一段信号中的瞬时频率是多少，需要由短时傅里叶变换（STFT）来解决，STFT能够应对时变非平稳的信号

% Part3------spectrogram函数的过程是怎样的?
% 要素1：窗长（数据点的个数），要素2：重叠（两个相邻窗重叠数据点的个数）

% Part4------界面化spectrogram
% signal analyzer (APP工具箱) --------简单分析操作

% Part5------如何准确无误地使用代码spectrogram
% 重点
% ------------------------------------------------------------------------------
% doc spectrogram  % 查看spectrogram函数的帮助文档
x = Wear_bend_1;     % 变量可以更换

% 语法1：s = spectrogram(x)
s = spectrogram(x);   % 默认将数据分成8段
spectrogram(x)          % 不给输出, 将绘制图像

% 语法2：s = spectrogram(x,window)
spectrogram(x,hann(128)); % 第二个参数默认使用hamming, 第二个参数默认窗长的一半
% 傅里叶变换的次数为128/2 + 1

% 语法3：s = spectrogram(x,window,noverlap)
spectrogram(x,128 ,120); % 第三个参数反映了：重叠的越多,那么进行的傅里叶变换就越多

% 语法4：s = spectrogram(x,window,noverlap,nfft)
spectrogram(x,128,120,128); % 第四个参数默认为256

% 语法5：[s,w,t] = spectrogram(___)
[s,w,t] = spectrogram(x,128,120,128); % s为输出结果，w为标准化频率，t为时间 
waterfall(t,w,abs(s)); % 绘制瀑布图

% 语法6：[s,f,t] = spectrogram(___,fs)
[s,f,t] = spectrogram(x,128,120,128,1000); % s为输出结果，w为标准化频率，t为时间 , Fs = 1000
waterfall(t,f,abs(s)); % 绘制瀑布图

% 语法7：[s,w,t] = spectrogram(x,window,noverlap,w)
[s,w,t] = spectrogram(x,128,120, 1:0.1:3.14); % 指定标准化的频率向量
waterfall(t,w,abs(s)); % 绘制瀑布图

% 语法8：[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % 给定采样率, 频率向量
waterfall(t,f,abs(s)); % 绘制瀑布图

% 语法9：[___,ps] = spectrogram(___)
[s,f,t,ps] = spectrogram(x,128,120,1:500,Fs,'power'); % 功率谱
[s,f,t,psd] = spectrogram(x,128,120,1:500,Fs,'PSD'); % 功率密度谱

% 语法10：[___] = spectrogram(___,'reassigned')
[s,f,t] = spectrogram(x,128,120,1000,'reassigned');
waterfall(t,f,abs(s)); % 绘制瀑布图
% ------------------------------------------------------
subplot(211); % 与语法6进行比较，用时频图
spectrogram(x,128,120,128,1000);
subplot(212); 
spectrogram(x,128,120,1000,'reassigned'); % 每一个频率bin中取最大值进行显示

% 语法11：[___,ps,fc,tc] = spectrogram(___)
[s,f,t,ps,fc,tc] = spectrogram(x,128,120,128,1000);

% 语法12：[___] = spectrogram(___,freqrange)
subplot(311);
spectrogram(x,128,120,128,1000,"onesided"); % 频率范围: 单边
subplot(312);
spectrogram(x,128,120,128,1000,"twosided"); % 频率范围: 双边
subplot(313);
spectrogram(x,128,120,128,1000,"centered"); % 频率范围: 中心

% 语法13：[___] = spectrogram(___,Name,Value)
spectrogram(x,128,120,128,1000,"MinThreshold",-50); % 低于-50的全部置成0

% 语法14：spectrogram(___,freqloc)
spectrogram(x,128,120,128,1000,"yaxis"); % 将频率轴放置在纵坐标上

% Part6------酷炫地展示spectrogram的结果
% ========================================


%% Part6------酷炫地展示spectrogram的结果
%  ========================================
% colorbar off; % 关闭能量条，一般不用关闭
% waterfall(X,Y,Z) 创建瀑布图，这是一种沿 y 维度有部分帷幕的网格图。这会产生一种“瀑布”效果。
% 该函数将矩阵 Z 中的值绘制为由 X 和 Y 定义的 x-y 平面中的网格上方的高度。边颜色因 Z 指定的高度而异。

figure(13); % 采集值
% Wear_bend_1为处理后的第1列数据，可以先用Signal Analyzer处理再导出Wear_bend_1
% Wear_bend_2为处理后的第2列数据，可以先用Signal Analyzer处理再导出Wear_bend_2
x = Wear_bend_1;     % 变量可以更换
Fs = 1000; % 采样频率
% colormap jet; % 改变显示配色
subplot(221);
spectrogram(x,128,120,128,1000,"yaxis"); % 平面二维展示
title('Acquisition value - time-frequency graph display (2D)','FontSize',12);xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);
subplot(222);
spectrogram(x,128,120,128,1000,"yaxis");
view(-45,45); % 调整三维显示
title('Acquisition value - time-frequency graph display (3D)','FontSize',12); xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);zlabel('Amplitude','FontSize',12);
subplot(223);
% 语法8：[s,f,t] = spectrogram(x,window,noverlap,f,fs)
% s 为输出结果---输入信号x的短时傅里叶变换；
% f 表示显示的频谱范围，f是一个向量，长度跟输出s的行数相同；
% t 表示显示的时间范围，t是一个向量，长度跟s的列数相同；
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % 给定采样率, 频率向量 , Fs = 1000
waterfall(t,f,abs(s)); % 绘制瀑布图
view(360,0); % 调整瀑布图的二维显示
title('Acquisition Value -STFT Waterfall Map (2D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Collected value -STFT output','FontSize',12);
subplot(224);
% 语法8：[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % 给定采样率, 频率向量 , Fs = 1000
waterfall(t,f,abs(s)); % 绘制瀑布图
title('Acquisition Value -STFT Waterfall Map (3D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Collected value -STFT output','FontSize',12);

figure(14); % 功率值
% Wear_bend_1为处理后的第1列数据，可以先用Signal Analyzer处理再导出Wear_bend_1
% Wear_bend_2为处理后的第2列数据，可以先用Signal Analyzer处理再导出Wear_bend_2
x = Wear_bend_2;     % 变量可以更换
Fs = 1000; % 采样频率
% colormap jet; % 改变显示配色 -----------------------------------------------------------------
subplot(221);
spectrogram(x,128,120,128,1000,"yaxis"); % 平面二维展示
title('Power Value - time-frequency graph display (2D)','FontSize',12);xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);
subplot(222);
spectrogram(x,128,120,128,1000,"yaxis");
view(-45,45); % 调整三维显示
title('Power Value - time-frequency graph display (3D)','FontSize',12); xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);zlabel('Amplitude','FontSize',12);
subplot(223);
% 语法8：[s,f,t] = spectrogram(x,window,noverlap,f,fs)
% s 为输出结果---输入信号x的短时傅里叶变换；
% f 表示显示的频谱范围，f是一个向量，长度跟输出s的行数相同；
% t 表示显示的时间范围，t是一个向量，长度跟s的列数相同；
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % 给定采样率, 频率向量 , Fs = 1000
waterfall(t,f,abs(s)); % 绘制瀑布图
view(360,0); % 调整瀑布图的二维显示
title('Power Value -STFT Waterfall Diagram (2D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Power value -STFT output','FontSize',12);
subplot(224);
% 语法8：[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % 给定采样率, 频率向量 , Fs = 1000
waterfall(t,f,abs(s)); % 绘制瀑布图
title('Power Value -STFT Waterfall Diagram (3D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Power value -STFT output','FontSize',12);


% 时频分析（JTFA）即时频联合域分析，作为分析时变非平稳信号的有力工具，清楚地描述了信号频率随时间的变化关系。
% 功能：使用短时傅里叶变换得到信号的频谱图。
% 语法：
% [S,F,T,P]=spectrogram(x,window,noverlap,nfft,fs)
% [S,F,T,P]=spectrogram(x,window,noverlap,F,fs)
% 说明：当使用时无输出参数，会自动绘制频谱图；有输出参数，则会返回输入信号的短时傅里叶变换。
% 当然也可以从函数的返回值S,F,T,P绘制频谱图，具体参见例子。
% 参数：
% x---输入信号的向量。默认情况下，即没有后续输入参数，x将被分成8段分别做变换处理，如果x不能被平分成8段, 则会做截断处理。
% 默认情况下，其他参数的默认值为：window---窗函数，默认为nfft长度的海明窗Hamming；
% noverlap---每一段的重叠样本数，默认值是在各段之间产生50%的重叠；
% nfft---做FFT变换的长度，默认为256和大于每段长度的最小2次幂之间的最大值。另外，此参数除了使用一个常量外，还可以指定一个频率向量F；
% fs---采样频率，默认值归一化频率。
% Window---窗函数，如果window为一个整数，x将被分成window段，每段使用Hamming窗函数加窗。
% 如果window是一个向量，x将被分成length(window)段，每一段使用window向量指定的窗函数加窗。
% 所以如果想获取specgram函数的功能，只需指定一个256长度的Hann窗。
% Noverlap---各段之间重叠的采样点数。它必须为一个小于window或length(window)的整数。
% 其意思为两个相邻窗不是尾接着头的，而是两个窗有交集，有重叠的部分。
% Nfft---计算离散傅里叶变换的点数。它需要为标量。
% Fs---采样频率Hz，如果指定为[]，默认为1Hz。
% S---输入信号x的短时傅里叶变换。它的每一列包含一个短期局部时间的频率成分估计，时间沿列增加，频率沿行增加。
% 如果x是长度为Nx的复信号，则S为nfft行k列的复矩阵，其中k取决于window，如果window为一个标量，则k = fix((Nx-noverlap)/(window-noverlap))；
% 如果window为向量，则k = fix((Nx-noverlap)/(length(window)-noverlap))。对于实信号x，如果nfft为偶数，则S的行数为(nfft/2+1)，
% 如果nfft为奇数，则行数为(nfft+1)/2，列数同上。
% F---在输入变量中使用F频率向量，函数会使用Goertzel方法计算在F指定的频率处计算频谱图。
% 指定的频率被四舍五入到与信号分辨率相关的最近的DFT容器(bin)中。而在其他的使用nfft语法中，短时傅里叶变换方法将被使用。
% 对于返回值中的F向量，为四舍五入的频率，其长度等于S的行数。
% T---频谱图计算的时刻点，其长度等于上面定义的k，值为所分各段的中点。
% P---能量谱密度PSD(Power Spectral Density)，对于实信号，P是各段PSD的单边周期估计；
% 对于复信号，当指定F频率向量时，P为双边PSD。P矩阵的元素计算公式如下P(I,j)=k|S(I,j)|2，
% 其中的的k是实值标量，定义如下对于单边PSD，计算公式如下，其中w(n)表示窗函数，Fs为采样频率，在0频率和奈奎斯特频率处，分子上的因子2改为1；
% ————————————————
% 原文链接：https://blog.csdn.net/shenziheng1/article/details/53868684
        