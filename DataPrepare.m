%% 此部分程序通过整理 .mat 格式的原始光谱数据，对其进行插值，得到用于训练的光谱数据
%  包括originalSpectra, labels, sampleLens
clear;
close;
% 加载 .mat 数据
Normal = loadSerumSpectra('MockNormalSpectra.mat');
Esophageal = loadSerumSpectra('MockECSpectra.mat');
Lung = loadSerumSpectra('MockLCSpectra.mat');

% 在工作区间进行线性插值
% 首先手动输入区间范围
defaultRange = {'500', '1650'};
prompt = {'请输入光谱下界(cm^-1)', '请输入光谱上界(cm^-1)'};
dlgtittle = '请输入光谱区间';
dims = [1 35]; % 定义输入框大小
answer = inputdlg(prompt,dlgtittle,dims,defaultRange);
% 增强鲁棒性操作
if isempty(answer)
    disp('用户取消输入，使用默认区间');
    lb = str2num(defaultRange{1});
    ub = str2num(defaultRange{2});
else
    lb = str2num(answer{1});
    ub = str2num(answer{2});
end

% 对测量区间符合输入要求的光谱进行小波变换分解,计算质量分数
% 预分配空间
originalSpectra = [];
labels = {};
spectrum_id = {};
sampleLens = [];
sampleLabels = {};

% 以循环的形式依次读入各个病理分类的数据
% 将结构体储存在元胞数组中
structCells = {Normal, Esophageal, Lung};
structSampleLabels = {'正常','食管癌','肺癌'};

for k = 1:length(structCells)
    currentStruct = structCells{k};
    currentSampleLabel = structSampleLabels{k};
    for i = 1:length(currentStruct)
        % 只计入被整理入光谱矩阵的样本光谱数目
        sampleLen = 0;
        sampleLabels = [sampleLabels;currentSampleLabel];
        for j = 1:length(currentStruct(i).SpectraID)
            % 对某个样本上的光谱首先进行测量区间检定
            range = currentStruct(i).SpectraRange(j);
            parts = split(range,'-');
            lowerBound = str2num(parts{1});
            upperBound = str2num(parts{2});
            if (lb<lowerBound) || (ub>upperBound)
                break;  % 若光谱的测量区间与输入的工作区间不匹配，则不录入该条光谱
            end
            % 整理光谱数据、标签、编号
            spectra_data = [currentStruct(i).wavenumber{j,:};currentStruct(i).intensity{j,:}];
            label = currentStruct(i).Labels{j};
            currentSpectrumID = strcat(currentStruct(i).SampleID, '-',currentStruct(i).SpectraID{j});
            % 为光谱添加两端数据点，方便采用端点值进行外插
            xq = lb:1:ub;
            x_ext = [spectra_data(1,1)-1,spectra_data(1,:),spectra_data(1,end)+1];
            y_ext = [spectra_data(2,1),spectra_data(2,:),spectra_data(2,end)];
            y = interp1(x_ext,y_ext,xq,'linear','extrap');
            originalSpectra = [originalSpectra;y];
            labels = [labels;label];
            spectrum_id = [spectrum_id; currentSpectrumID];
            sampleLen = sampleLen + 1;
        end
        sampleLens = [sampleLens; sampleLen];
    end
end

% 储存原始光谱数据
save('OriginalData.mat','originalSpectra','labels','spectrum_id','xq','sampleLens','sampleLabels');

%% 此部分代码通过离散小波变换计算光谱质量分数，整理输入 CNN 的光谱矩阵
% 暂时仍然保持 db8 基函数，进行11层分解
load('OriginalData.mat','originalSpectra','labels','spectrum_id','xq', 'sampleLens','sampleLabels');
level = 11; wname = 'db8';
normalizedSpectra = [];
MHER = [];
MLER = [];

% 首先对原始光谱进行 Z-score 归一化
for i = 1:size(originalSpectra,1)
    meanSpectrum = mean(originalSpectra(i,:));
    stdSpectrum = std(originalSpectra(i,:));
    normalizedSpectra(i,:) = (originalSpectra(i,:) - meanSpectrum) / stdSpectrum;
end

% 对归一化后的光谱计算质量控制判据
Area1 =  sum(normalizedSpectra(:, 60:80), 2);
Area2 = sum(normalizedSpectra(:,120:140),2);
interval = normalizedSpectra(:,490:510);
diff1 = max(interval,[],2) - min(interval,[],2);
% idx1 = find(Area2 > 50 & Area1 < 20 & Area1 > 0 & diff1 > 0.7);

% 构造重构矩阵
[m, n] = size(normalizedSpectra);
reconSpectra = zeros(m,n,level +1);
% 对所有光谱遍历
for i = 1:m
    vec = normalizedSpectra(i,:);
    [c, l] = wavedec(vec, level,'db8'); % 对第 i 条光谱进行频率分解

    % 对所有分解级数遍历
    for j = 1:level
        D_j = wrcoef('d',c,l,wname,j);  % 求得第 j 级重构细节光谱
        reconSpectra(i,:,j) = D_j;
    end
    % 补充近似信号
    c_approx = c;
    c_approx(l(1)+1 : end) = 0;   % 将细节系数部分全部置零
    approx = waverec(c_approx, l, 'db8');
    plot(xq, approx, 'g--');
    % approx = wrcoef('a',c,l,wname);
    % plot(xq,approx);
    reconSpectra(i,:,level+1) = approx;
end

% 验证小波变换结果的准确性，理论上误差矩阵 diff = 0
diff = normalizedSpectra - sum(reconSpectra,3);
a = max(max(abs(diff)));
if (max(abs(diff)) < 1e-8)
    disp('光谱的小波分解与重构无误');
else 
    disp('重构光谱之和不等于原始光谱');
end

% 根据能量判据计算信噪比,暂时用第7级信号代表中频带，第1级细节代表高频带，近似信号代表低频带
Energy_medium = sum(reconSpectra(:,:,7).^2,2);
Energy_high = sum(reconSpectra(:,:,1).^2,2);
Energy_low = sum(reconSpectra(:,:,12).^2,2);
MHER = Energy_medium ./ Energy_high;
MLER = Energy_medium ./ Energy_low;

% 计算质量分数随着光谱类别的分布情况
categories = unique(labels);
numCategories = length(categories);

MHER_mean = zeros(numCategories,1);
MHER_var = zeros(numCategories,1);
MLER_mean = zeros(numCategories,1);
MLER_var = zeros(numCategories,1);

% 对所有光谱种类循环
for i = 1:numCategories
    % 获取当前类别数据的索引
    idx = strcmp(labels, categories{i});
    % 提取当前类别的质量分数
    mher_i = MHER(idx);
    mler_i = MLER(idx);
    % 计算均值与方差
    MHER_mean(i) = mean(mher_i);
    MHER_var(i) = var(mher_i);
    MLER_mean(i) = mean(mler_i);
    MLER_var(i) = var(mler_i);
end

% 以表格形式展示对比结果
resultTable = table(categories, MHER_mean, MHER_var, MLER_mean, MLER_var);
disp(resultTable);

% 将小波分解后的数据储存
% 根据小波分解级数构造文件名称
name = strcat('WaveletData',num2str(level),'.mat');
save(name,'reconSpectra','labels','spectrum_id','xq','sampleLens','sampleLabels','MHER','MLER','Area1','Area2','diff1');
