% �������źŴ������š�
%% ��ʶ������ź�
% 3�༡��: ������\�ļ�\ƽ����
% ������ź��������
% ʱ����ǿ���� �� Ƶ��60-80Hz

%% �����źŵĲɼ�
% �����豸��������
% �����缫�����������ο��缫����ؽڣ����ص缫���豸�ӵأ�


%% 
% �����ź�Ԥ����
% ���ţ���Ƶ�������˶�α�����ĵ�����
load('Wear_bend.mat'); %��������%%%%%%%%%%%%%%%%%%%%%%
figure(1);
plot(Wear_bend,'DisplayName','Wear_bend');
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);
Fs = 1000; %���ò���Ƶ��

fdatool; % ������˲����Ĺ�����
% ���ȵ��Bandpass, �ٵ��IIR, Ȼ����Options��ѡ��passband, Fs����Ϊ1000, 
% Fstop1����Ϊ20, Fpass1����Ϊ30,  Fpass2����Ϊ350, Fstop2����Ϊ400, �������
% ��File�е��Export, ��Export As �и�ΪObjects����Variable Name�и�Ϊfilt_bp, ����ѡ�·�


%% ���濪ʼ�˲�

Wear_bend_bp =  filter(filt_bp,Wear_bend); % ��ͨ�˲� 30-350
figure(2);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp');
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);

%% ����50Hz�ݲ��˲���(���пɲ���)
% ��fdatool��������Response Type��ѡ���һ���е�Notching, ��Design Method�е�IIR��ѡ��Single Notch
% Fs����Ϊ1000, Fnotch����Ϊ50, �������·���QΪ60, �������
% ��File�е��Export, ��Export As �и�ΪObjects����Variable Name�и�Ϊfilt_50, ����ѡ�·�
Wear_bend_50 =  filter(filt_50,Wear_bend_bp); % 50Hz�ݲ�
figure(3);
subplot(121);
plot(Wear_bend_bp);
subplot(122);
plot(Wear_bend_50);

%50Hz�ݲ����ͨ�˲��ȶ�,��50Hz�ݲ��˲������޷�������
figure(4);
plot(Wear_bend_bp(:,1));
hold on
plot(Wear_bend_50(:,1));
hold off
%������50Hz�ݲ����ͨ�˲����,û�д�������,����Ժ���50Hz�ݲ���һ����


%% �����������ź�
figure(5);
plot(Wear_bend_bp(:,1));
hold on
plot(Wear_bend_bp(:,2)); 
%plot(Wear_bend_bp(:,2)+400);  % ��λչʾ2���ɼ��źŵĲ���
hold off
legend('Collected value','Power value','FontSize',12); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);

%% ����ʹ��APP�е�Signal Analyzer
% ���滯�˲����������ƣ�����д���룩
% �����BվUP: https://www.bilibili.com/video/BV1Eg411S7LC/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93


%% �����ź�ʱ������
figure(6);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp'); % ѡȡ���ݵ���빤����, ��8����
% ���ֵ����Сֵ��������ֵRMS������ֵƽ��ֵ������
emg1 = Wear_bend_bp(:,1);
emg2 = Wear_bend_bp(:,2);
inde = [cursor_info.DataIndex];

% ʱ�������������ֵ
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.Max(i,j) = max(emg(star:term));
    end
end

% ʱ������������Сֵ
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.Min(i,j) = min(emg(star:term));
    end
end


% ʱ����������������ֵRMS
% ������ֵ�ȶ���ǿ
for i = 1:2
    emg = Wear_bend_bp(:,i);
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.RMS(i,j) = rms(emg(star:term));
    end
end


% ʱ��������������ֵƽ��ֵ
for i = 1:2
    emg = abs(Wear_bend_bp(:,i));
    for j = 1:4
        star = inde(2*j);
        term = inde(2*j-1);
        fea.ABS_mean(i,j) = mean(emg(star:term));
    end
end


% ʱ��������������
figure(7); % �������ΰ�����
subplot(211);
[upper1,lower1] = envelope(emg1,150,'rms');
plot(emg1); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); legend('Collected value','Upper envelope of the collected value','Lower envelope of the acquired value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off
subplot(212);
[upper2,lower2] = envelope(emg2,150,'rms');
plot(emg2); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); legend('Power value','Upper envelope of the power value','Lower envelope of power value','FontSize',8);xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12);hold off

% ������ȡһ���ź�
figure(8); % ���Ƴ�ȡ��һ���źŵİ�����
emg1_seg = emg1(inde(8):inde(1)); % ָ�� ע������8�����ݵ�
subplot(211);
[upper1,lower1] = envelope(emg1_seg,150,'rms');
plot(emg1_seg); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); legend('Collected value','The upper envelope of the collected value','Lower envelope of the acquired value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off
emg2_seg = emg2(inde(8):inde(1)); % ָ��
subplot(212);
[upper2,lower2] = envelope(emg2_seg,150,'rms');
plot(emg2_seg); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); legend('Power value','Upper envelope of the power value','Lower envelope of power value','FontSize',8); xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off

% ��2��ͨ�������߻�����һ��
figure(9); % �������ư�����
plot(upper1,'LineWidth',1.5); hold on; plot(upper2,'LineWidth',1.5); hold on; 
plot(lower1,'LineWidth',1.5); hold on; plot(lower2,'LineWidth',1.5); 
legend('Upper envelope of the collected value','Upper envelope of the power value','Lower envelope of the acquired value','Lower envelope of power value','FontSize',8); 
xlabel('Time/ms','FontSize',12); ylabel('Myoelectric strength','FontSize',12); hold off; 

%% �����ź�Ƶ������
inde = [cursor_info.DataIndex];
seg = 1; % seg = 1��ʾ���ǵ�һ���ź�, seg�����޸� %%%%%%%%%%%%%%%%%%%  Notice: 8������ȡ4�Σ�seg = 1, 2, 3, 4
% Ƶ����������seg = 1, 2, 3, 4���θ�ִֵ�����δ����м���, �����ڹ������пɼ�����FDF       %%%%%%%%%%%%%%%%%%% 
init = inde(2*seg);
term = inde(2*seg-1);
ch1 = Wear_bend_bp(init:term,1);
ch2 = Wear_bend_bp(init:term,2);
L = length(ch1);
Fs = 1000;

% fft  ���ٸ���Ҷ�任 , Ƶ����Ϣ , ��Ƶͼ
% ch1��ch2�ķ�Ƶͼ�ȶ�
% ch1��fft    ͨ��1
ch1_fft = fft(ch1);
ch1_P2 = abs(ch1_fft/L);
ch1_P1 = ch1_P2(1:L/2+1);
ch1_P1(2:end-1) = 2*ch1_P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure(10);plot(f,ch1_P1);
legend('Acquisition value (first paragraph)','Power value (first paragraph)','FontSize',12); % ��һ�α�ʾ seg = 1���ڶ��α�ʾ seg = 2������������
xlabel('Frequency (Hz)','FontSize',12); ylabel('Maginitude (dB)','FontSize',12);hold on;

% ch2��fft    ͨ��2
ch2_fft = fft(ch2);
ch2_P2 = abs(ch2_fft/L);
ch2_P1 = ch2_P2(1:L/2+1);
ch2_P1(2:end-1) = 2*ch2_P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,ch2_P1);
legend('Acquisition value (first paragraph)','Power value (first paragraph)','FontSize',12); % ��һ�α�ʾ seg = 1���ڶ��α�ʾ seg = 2������������
xlabel('Frequency (Hz)','FontSize',12); ylabel('Maginitude (dB)','FontSize',12);hold off;



% Ƶ���ֵƽ��ֵ
FDF.AF_AM(1,seg) = mean(ch1_P1);
FDF.AF_AM(2,seg) = mean(ch2_P1);

% ����Ƶ��
FDF.AF_CF(1,seg) = sum(f'.*ch1_P1)/sum(ch1_P1);
FDF.AF_CF(2,seg) = sum(f'.*ch2_P1)/sum(ch2_P1);

% Ƶ�ʷ���
FDF.AF_FVAR(1,seg) = sum(((f-FDF.AF_CF(1,seg)).^2)'.*ch1_P1)/sum(ch1_P1);
FDF.AF_FVAR(2,seg) = sum(((f-FDF.AF_CF(2,seg)).^2)'.*ch2_P1)/sum(ch2_P1);

% ��ֵƵ�� , ���ڹ�����ͼ
% ����ch1����ֵƵ��
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

% ����ch2����ֵƵ��
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


%% �����ź�ʱƵ������
% ��������1��APP������ �����źŷ�����(Signal Analyzer)�� ��2��spectrogram����

% ������1��ԭʼ�źŵ���
% Bվ����: https://www.bilibili.com/video/BV1ye4y1q7v3/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93

% ������2�����ݱ�����Ҫ��Signal Analyzer�����䵼��
figure(11);
spectrogram(Wear_bend_1,256,250,1000,1000,'yaxis'); % Wear_bend_1Ϊ�����ĵ�1�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_1
figure(12);
spectrogram(Wear_bend_2,256,250,1000,1000,'yaxis'); % Wear_bend_2Ϊ�����ĵ�2�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_2


%% ȫ������spectrogram����
% ��Ƶ���ӣ�https://www.bilibili.com/video/BV14F41157Gz/?spm_id_from=333.999.0.0&vd_source=76446368a03ef22caf531595d665cb93

% Part1------spectrogram������ɶ?
% spectrogram������ʵ����--------��ʱ����Ҷ�任��STFT��

% Part2------ΪʲôҪ�õ�spectrogram����?
% ��Ϊ����Ҷ�任��FFT���޷���������һ���ź��е�˲ʱƵ���Ƕ��٣���Ҫ�ɶ�ʱ����Ҷ�任��STFT���������STFT�ܹ�Ӧ��ʱ���ƽ�ȵ��ź�

% Part3------spectrogram�����Ĺ�����������?
% Ҫ��1�����������ݵ�ĸ�������Ҫ��2���ص����������ڴ��ص����ݵ�ĸ�����

% Part4------���滯spectrogram
% signal analyzer (APP������) --------�򵥷�������

% Part5------���׼ȷ�����ʹ�ô���spectrogram
% �ص�
% ------------------------------------------------------------------------------
% doc spectrogram  % �鿴spectrogram�����İ����ĵ�
x = Wear_bend_1;     % �������Ը���

% �﷨1��s = spectrogram(x)
s = spectrogram(x);   % Ĭ�Ͻ����ݷֳ�8��
spectrogram(x)          % �������, ������ͼ��

% �﷨2��s = spectrogram(x,window)
spectrogram(x,hann(128)); % �ڶ�������Ĭ��ʹ��hamming, �ڶ�������Ĭ�ϴ�����һ��
% ����Ҷ�任�Ĵ���Ϊ128/2 + 1

% �﷨3��s = spectrogram(x,window,noverlap)
spectrogram(x,128 ,120); % ������������ӳ�ˣ��ص���Խ��,��ô���еĸ���Ҷ�任��Խ��

% �﷨4��s = spectrogram(x,window,noverlap,nfft)
spectrogram(x,128,120,128); % ���ĸ�����Ĭ��Ϊ256

% �﷨5��[s,w,t] = spectrogram(___)
[s,w,t] = spectrogram(x,128,120,128); % sΪ��������wΪ��׼��Ƶ�ʣ�tΪʱ�� 
waterfall(t,w,abs(s)); % �����ٲ�ͼ

% �﷨6��[s,f,t] = spectrogram(___,fs)
[s,f,t] = spectrogram(x,128,120,128,1000); % sΪ��������wΪ��׼��Ƶ�ʣ�tΪʱ�� , Fs = 1000
waterfall(t,f,abs(s)); % �����ٲ�ͼ

% �﷨7��[s,w,t] = spectrogram(x,window,noverlap,w)
[s,w,t] = spectrogram(x,128,120, 1:0.1:3.14); % ָ����׼����Ƶ������
waterfall(t,w,abs(s)); % �����ٲ�ͼ

% �﷨8��[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % ����������, Ƶ������
waterfall(t,f,abs(s)); % �����ٲ�ͼ

% �﷨9��[___,ps] = spectrogram(___)
[s,f,t,ps] = spectrogram(x,128,120,1:500,Fs,'power'); % ������
[s,f,t,psd] = spectrogram(x,128,120,1:500,Fs,'PSD'); % �����ܶ���

% �﷨10��[___] = spectrogram(___,'reassigned')
[s,f,t] = spectrogram(x,128,120,1000,'reassigned');
waterfall(t,f,abs(s)); % �����ٲ�ͼ
% ------------------------------------------------------
subplot(211); % ���﷨6���бȽϣ���ʱƵͼ
spectrogram(x,128,120,128,1000);
subplot(212); 
spectrogram(x,128,120,1000,'reassigned'); % ÿһ��Ƶ��bin��ȡ���ֵ������ʾ

% �﷨11��[___,ps,fc,tc] = spectrogram(___)
[s,f,t,ps,fc,tc] = spectrogram(x,128,120,128,1000);

% �﷨12��[___] = spectrogram(___,freqrange)
subplot(311);
spectrogram(x,128,120,128,1000,"onesided"); % Ƶ�ʷ�Χ: ����
subplot(312);
spectrogram(x,128,120,128,1000,"twosided"); % Ƶ�ʷ�Χ: ˫��
subplot(313);
spectrogram(x,128,120,128,1000,"centered"); % Ƶ�ʷ�Χ: ����

% �﷨13��[___] = spectrogram(___,Name,Value)
spectrogram(x,128,120,128,1000,"MinThreshold",-50); % ����-50��ȫ���ó�0

% �﷨14��spectrogram(___,freqloc)
spectrogram(x,128,120,128,1000,"yaxis"); % ��Ƶ�����������������

% Part6------���ŵ�չʾspectrogram�Ľ��
% ========================================


%% Part6------���ŵ�չʾspectrogram�Ľ��
%  ========================================
% colorbar off; % �ر���������һ�㲻�ùر�
% waterfall(X,Y,Z) �����ٲ�ͼ������һ���� y ά���в����Ļ������ͼ��������һ�֡��ٲ���Ч����
% �ú��������� Z �е�ֵ����Ϊ�� X �� Y ����� x-y ƽ���е������Ϸ��ĸ߶ȡ�����ɫ�� Z ָ���ĸ߶ȶ��졣

figure(13); % �ɼ�ֵ
% Wear_bend_1Ϊ�����ĵ�1�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_1
% Wear_bend_2Ϊ�����ĵ�2�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_2
x = Wear_bend_1;     % �������Ը���
Fs = 1000; % ����Ƶ��
% colormap jet; % �ı���ʾ��ɫ
subplot(221);
spectrogram(x,128,120,128,1000,"yaxis"); % ƽ���άչʾ
title('Acquisition value - time-frequency graph display (2D)','FontSize',12);xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);
subplot(222);
spectrogram(x,128,120,128,1000,"yaxis");
view(-45,45); % ������ά��ʾ
title('Acquisition value - time-frequency graph display (3D)','FontSize',12); xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);zlabel('Amplitude','FontSize',12);
subplot(223);
% �﷨8��[s,f,t] = spectrogram(x,window,noverlap,f,fs)
% s Ϊ������---�����ź�x�Ķ�ʱ����Ҷ�任��
% f ��ʾ��ʾ��Ƶ�׷�Χ��f��һ�����������ȸ����s��������ͬ��
% t ��ʾ��ʾ��ʱ�䷶Χ��t��һ�����������ȸ�s��������ͬ��
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % ����������, Ƶ������ , Fs = 1000
waterfall(t,f,abs(s)); % �����ٲ�ͼ
view(360,0); % �����ٲ�ͼ�Ķ�ά��ʾ
title('Acquisition Value -STFT Waterfall Map (2D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Collected value -STFT output','FontSize',12);
subplot(224);
% �﷨8��[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % ����������, Ƶ������ , Fs = 1000
waterfall(t,f,abs(s)); % �����ٲ�ͼ
title('Acquisition Value -STFT Waterfall Map (3D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Collected value -STFT output','FontSize',12);

figure(14); % ����ֵ
% Wear_bend_1Ϊ�����ĵ�1�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_1
% Wear_bend_2Ϊ�����ĵ�2�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_2
x = Wear_bend_2;     % �������Ը���
Fs = 1000; % ����Ƶ��
% colormap jet; % �ı���ʾ��ɫ -----------------------------------------------------------------
subplot(221);
spectrogram(x,128,120,128,1000,"yaxis"); % ƽ���άչʾ
title('Power Value - time-frequency graph display (2D)','FontSize',12);xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);
subplot(222);
spectrogram(x,128,120,128,1000,"yaxis");
view(-45,45); % ������ά��ʾ
title('Power Value - time-frequency graph display (3D)','FontSize',12); xlabel('Time/s','FontSize',12);ylabel('Frequency/Hz','FontSize',12);zlabel('Amplitude','FontSize',12);
subplot(223);
% �﷨8��[s,f,t] = spectrogram(x,window,noverlap,f,fs)
% s Ϊ������---�����ź�x�Ķ�ʱ����Ҷ�任��
% f ��ʾ��ʾ��Ƶ�׷�Χ��f��һ�����������ȸ����s��������ͬ��
% t ��ʾ��ʾ��ʱ�䷶Χ��t��һ�����������ȸ�s��������ͬ��
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % ����������, Ƶ������ , Fs = 1000
waterfall(t,f,abs(s)); % �����ٲ�ͼ
view(360,0); % �����ٲ�ͼ�Ķ�ά��ʾ
title('Power Value -STFT Waterfall Diagram (2D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Power value -STFT output','FontSize',12);
subplot(224);
% �﷨8��[s,f,t] = spectrogram(x,window,noverlap,f,fs)
[s,f,t] = spectrogram(x,128,120,1:500,Fs); % ����������, Ƶ������ , Fs = 1000
waterfall(t,f,abs(s)); % �����ٲ�ͼ
title('Power Value -STFT Waterfall Diagram (3D)','FontSize',12); xlabel('Time/s','FontSize',12); ylabel('Frequency/Hz','FontSize',12);zlabel('Power value -STFT output','FontSize',12);


% ʱƵ������JTFA����ʱƵ�������������Ϊ����ʱ���ƽ���źŵ��������ߣ�������������ź�Ƶ����ʱ��ı仯��ϵ��
% ���ܣ�ʹ�ö�ʱ����Ҷ�任�õ��źŵ�Ƶ��ͼ��
% �﷨��
% [S,F,T,P]=spectrogram(x,window,noverlap,nfft,fs)
% [S,F,T,P]=spectrogram(x,window,noverlap,F,fs)
% ˵������ʹ��ʱ��������������Զ�����Ƶ��ͼ���������������᷵�������źŵĶ�ʱ����Ҷ�任��
% ��ȻҲ���ԴӺ����ķ���ֵS,F,T,P����Ƶ��ͼ������μ����ӡ�
% ������
% x---�����źŵ�������Ĭ������£���û�к������������x�����ֳ�8�ηֱ����任�������x���ܱ�ƽ�ֳ�8��, ������ضϴ���
% Ĭ������£�����������Ĭ��ֵΪ��window---��������Ĭ��Ϊnfft���ȵĺ�����Hamming��
% noverlap---ÿһ�ε��ص���������Ĭ��ֵ���ڸ���֮�����50%���ص���
% nfft---��FFT�任�ĳ��ȣ�Ĭ��Ϊ256�ʹ���ÿ�γ��ȵ���С2����֮������ֵ�����⣬�˲�������ʹ��һ�������⣬������ָ��һ��Ƶ������F��
% fs---����Ƶ�ʣ�Ĭ��ֵ��һ��Ƶ�ʡ�
% Window---�����������windowΪһ��������x�����ֳ�window�Σ�ÿ��ʹ��Hamming�������Ӵ���
% ���window��һ��������x�����ֳ�length(window)�Σ�ÿһ��ʹ��window����ָ���Ĵ������Ӵ���
% ����������ȡspecgram�����Ĺ��ܣ�ֻ��ָ��һ��256���ȵ�Hann����
% Noverlap---����֮���ص��Ĳ���������������Ϊһ��С��window��length(window)��������
% ����˼Ϊ�������ڴ�����β����ͷ�ģ������������н��������ص��Ĳ��֡�
% Nfft---������ɢ����Ҷ�任�ĵ���������ҪΪ������
% Fs---����Ƶ��Hz�����ָ��Ϊ[]��Ĭ��Ϊ1Hz��
% S---�����ź�x�Ķ�ʱ����Ҷ�任������ÿһ�а���һ�����ھֲ�ʱ���Ƶ�ʳɷֹ��ƣ�ʱ���������ӣ�Ƶ���������ӡ�
% ���x�ǳ���ΪNx�ĸ��źţ���SΪnfft��k�еĸ���������kȡ����window�����windowΪһ����������k = fix((Nx-noverlap)/(window-noverlap))��
% ���windowΪ��������k = fix((Nx-noverlap)/(length(window)-noverlap))������ʵ�ź�x�����nfftΪż������S������Ϊ(nfft/2+1)��
% ���nfftΪ������������Ϊ(nfft+1)/2������ͬ�ϡ�
% F---�����������ʹ��FƵ��������������ʹ��Goertzel����������Fָ����Ƶ�ʴ�����Ƶ��ͼ��
% ָ����Ƶ�ʱ��������뵽���źŷֱ�����ص������DFT����(bin)�С�����������ʹ��nfft�﷨�У���ʱ����Ҷ�任��������ʹ�á�
% ���ڷ���ֵ�е�F������Ϊ���������Ƶ�ʣ��䳤�ȵ���S��������
% T---Ƶ��ͼ�����ʱ�̵㣬�䳤�ȵ������涨���k��ֵΪ���ָ��ε��е㡣
% P---�������ܶ�PSD(Power Spectral Density)������ʵ�źţ�P�Ǹ���PSD�ĵ������ڹ��ƣ�
% ���ڸ��źţ���ָ��FƵ������ʱ��PΪ˫��PSD��P�����Ԫ�ؼ��㹫ʽ����P(I,j)=k|S(I,j)|2��
% ���еĵ�k��ʵֵ�������������¶��ڵ���PSD�����㹫ʽ���£�����w(n)��ʾ��������FsΪ����Ƶ�ʣ���0Ƶ�ʺ��ο�˹��Ƶ�ʴ��������ϵ�����2��Ϊ1��
% ��������������������������������
% ԭ�����ӣ�https://blog.csdn.net/shenziheng1/article/details/53868684
        