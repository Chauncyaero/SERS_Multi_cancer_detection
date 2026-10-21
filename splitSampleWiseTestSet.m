function [trainIdx, testIdx, trainSampleIdx, testSampleIdx] = splitSampleWiseTestSet(sampleLens, sampleLabels, testRatio)
% 按照样本数量进行训练验证集、测试集的随机划分
% sampleLens: 每个样本的光谱数（列向量）
% testRatio: 测试集按样本数量划分比例（0~1之间）
% 返回：
%   trainIdx: 总光谱数维度的逻辑索引，表示哪些属于训练集
%   testIdx:   总光谱数维度的逻辑索引，表示哪些属于测试集

    numSamples = length(sampleLens);
    totalSpectraNumber = sum(sampleLens);

    % 单种子计算时需要解锁种子，多种子计算的时候记得注释掉种子
    % rng(0);
    % rng(42);
    % rng(123);
    % rng(2024);

    % 借助cvpartition函数划分测试集
    cvTest = cvpartition(sampleLabels,'HoldOut',testRatio);  % HoldOut表示单次划分，sampleLabels输入会尝试进行分层划分
    trainSampleIdx = training(cvTest);
    testSampleIdx = test(cvTest);

    % 计算每个样本上光谱的起始与结束索引号
    sampleStartIdx = cumsum([1;sampleLens(1:end-1)]);
    sampleEndIdx = cumsum(sampleLens);

    testIdx = false(totalSpectraNumber,1);  % 首先将测试集索引置零
    for i = find(testSampleIdx') % 将各测试集样本对应的光谱索引号置一
        testIdx(sampleStartIdx(i):sampleEndIdx(i)) = true;
    end

    trainIdx = ~testIdx;
end
