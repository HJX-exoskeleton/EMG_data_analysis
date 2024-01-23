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
Fs = 1000; %���ò���Ƶ��

fdatool; % ������˲����Ĺ�����
% ���ȵ��Bandpass, �ٵ��IIR, Ȼ����Options��ѡ��passband, Fs����Ϊ1000, 
% Fstop1����Ϊ20, Fpass1����Ϊ30,  Fpass2����Ϊ350, Fstop2����Ϊ400, �������
% ��File�е��Export, ��Export As �и�ΪObjects����Variable Name�и�Ϊfilt_bp, ����ѡ�·�


%% ���濪ʼ�˲�

Wear_bend_bp =  filter(filt_bp,Wear_bend); % ��ͨ�˲� 30-350
figure(2);
plot(Wear_bend_bp,'DisplayName','Wear_bend_bp');


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
hold off


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
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); hold off
subplot(212);
[upper2,lower2] = envelope(emg2,150,'rms');
plot(emg2); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); hold off

% ������ȡһ���ź�
figure(8); % ���Ƴ�ȡ��һ���źŵİ�����
emg1_seg = emg1(inde(8):inde(1)); % ָ�� ע������8�����ݵ�
subplot(211);
[upper1,lower1] = envelope(emg1_seg,150,'rms');
plot(emg1_seg); hold on; 
plot(upper1,'LineWidth',2); hold on; plot(lower1,'LineWidth',2); hold off
emg2_seg = emg2(inde(8):inde(1)); % ָ��
subplot(212);
[upper2,lower2] = envelope(emg2_seg,150,'rms');
plot(emg2_seg); hold on; 
plot(upper2,'LineWidth',2); hold on; plot(lower2,'LineWidth',2); hold off

% ��2��ͨ�������߻�����һ��
figure(9); % �������ư�����
plot(upper1,'LineWidth',1.5); hold on; plot(upper2,'LineWidth',1.5); hold on; 
plot(lower1,'LineWidth',1.5); hold on; plot(lower2,'LineWidth',1.5); hold off; 


%% �����ź�Ƶ������
inde = [cursor_info.DataIndex];
seg = 1; % seg = 1��ʾ���ǵ�һ���ź�, seg�����޸� %%%%%%%%%%%%%%%%%%%
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
figure(10);plot(f,ch1_P1);hold on;

% ch2��fft    ͨ��2
ch2_fft = fft(ch2);
ch2_P2 = abs(ch2_fft/L);
ch2_P1 = ch2_P2(1:L/2+1);
ch2_P1(2:end-1) = 2*ch2_P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,ch2_P1);hold off;

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
% ������1��
% Bվ����: https://www.bilibili.com/video/BV1ye4y1q7v3/?spm_id_from=pageDriver&vd_source=76446368a03ef22caf531595d665cb93

% ������2��
figure(11);
spectrogram(Wear_bend_1,256,250,1000,1000,'yaxis'); % Wear_bend_1Ϊ�����ĵ�1�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_1
figure(12);
spectrogram(Wear_bend_2,256,250,1000,1000,'yaxis'); % Wear_bend_2Ϊ�����ĵ�2�����ݣ���������Signal Analyzer�����ٵ���Wear_bend_2

        