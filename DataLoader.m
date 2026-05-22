% 此程序旨在对 Excel 格式的多癌种数据进行预处理与格式化，为后续的泛癌种血清识别研究提供符合要求的数据集。
% 数据输入为 Excel 格式的光谱数据，输出为 .mat 格式的光谱数据
clear;
% 调用图形界面，选择包含 Excel 文件的文件夹
folderPath = uigetdir(pwd, 'Select folder containing serum Excel files');
% 报错调试
if folderPath == 0
    disp('User canceled folder selection');
    return;
end
% 获取文件夹名称，作为最后生成的 mat 文件名
[~, folderName] = fileparts(folderPath);
matFileName = [folderName, '.mat'];
% 获取文件夹内所有.xlsx格式文件的路径
files = dir(fullfile(folderPath, '*.xlsx*'));

% 构造结构体数组储存各个样本上的光谱数据
% NumSpectra 一个样本一个，wavenumber 与 intensity 为 m×n 维度矩阵，m为样本上光谱数目
SerumSamples = struct('SampleID',{},'NumSpectra',{},'SpectraID',{},'Labels',{},'SpectraRange',{},'wavenumber',{},'intensity',{});

% 从目标文件夹中读取*.xlsx信息，从每个Excel文件中读取光谱信息，光谱标签并储存
n = length(files);
for i = 1:n
    FileName = files(i).name;
    SpectraFilePath = strcat(folderPath,'\',FileName);
    [numData, textData, ~] = xlsread(SpectraFilePath,'info');
    SerumSamples(i).SampleID = erase(FileName,'.xlsx');
    SerumSamples(i).NumSpectra = size(numData,1);
    % 将结构体中随光谱循环的变量初始化
    SpectraID ={};
    Labels = {};
    SpectraRange = {};
    wavenumber = {};
    intensity = {};
    for j = 1:SerumSamples(i).NumSpectra
        SpectraID = [SpectraID;num2str(numData(j))];
        Labels = [Labels;textData(j+1,3)];
        SpectraRange = [SpectraRange;textData(j+1,2)];
        spectrum = readmatrix(SpectraFilePath,'Sheet', SpectraID{j});
        % 原始数据不进行基线拉平，不插值，直接储存在wavenumber, intensity中
        wavenumber = [wavenumber;spectrum(:,1)'];
        intensity = [intensity;spectrum(:,2)'];
    end
    % 将某样本上的数据赋值到结构体上
    SerumSamples(i).SpectraID = SpectraID;
    SerumSamples(i).Labels = Labels;
    SerumSamples(i).SpectraRange = SpectraRange;
    SerumSamples(i).wavenumber = wavenumber;
    SerumSamples(i).intensity = intensity;
    msg = ['Sample ',num2str(i),' finished'];
    disp(msg);
end

save(matFileName,'SerumSamples');



    