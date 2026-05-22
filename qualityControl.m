function [idx_high_quality, newSampleLens] = qualityControl(sampleLens, qualityScores)
    % 通过输入的质量筛选阈值，对读取的光谱矩阵进行筛选
    % 首先获取需要控制的质量分数种类数目
    % 质量控制算法进行了优化与简化，保留必备的索引筛选功能，将训练数据和标签的筛选移出，以扩大该函数的适用范围

    % n_quality = size(qualityScores,2);
    % 
    % % 生成图形界面的默认阈值与提示性文本
    % defaultThresholds = repmat({'0.0'},1,n_quality); % 默认阈值为 1,n 维度的 0.0 元胞
    % prompt = cell(1,n_quality);
    % for i = 1:n_quality
    %     prompt{i} = sprintf('请输入第%d种质量分数的控制阈值',i);
    % end
    % dlgtittle = '光谱质量控制';
    % dims = [1 35]; % 定义输入框大小
    % 
    % answer = inputdlg(prompt,dlgtittle,dims,defaultThresholds);
    % 
    % % 增强鲁棒性操作
    % if isempty(answer)
    %     disp('用户取消输入，使用默认阈值');
    %     qualityThresholds = str2double(defaultThresholds);
    % else
    %     qualityThresholds = str2double(answer);
    % end
    % 
    % % 检查无效输入，将数组中的 nan 置零并输出警告
    % if(any(isnan(qualityThresholds)))
    %     warning('部分输入值无效，已将其置零');
    %     qualityThresholds(isnan(qualityThresholds)) = 0;
    % end
    % % 将 qualityThresholds 转置为行向量，方便之后进行比较
    % qualityThresholds = qualityThresholds';
    % 
    % idx_high_quality = all(qualityScores >= qualityThresholds,2);

    % 采用更加显示的质量控制标准
    idx_low_quality = (qualityScores(:,1) > -15) & (qualityScores(:,1) < 21) & ...
                   (qualityScores(:,2) > 55) & (qualityScores(:,3) > 0.6);
    % idx_low_quality = (qualityScores(:,1) > 0) & (qualityScores(:,1) < 20) & ...
    %                (qualityScores(:,2) > 50) & (qualityScores(:,3) > 0.7);
    idx_high_quality = ~idx_low_quality;

    % 根据预筛选结果重新计算 sampleLens
    cum_counts = cumsum(sampleLens);    % 对 sampleLens 进行累计求和
    % 计算每个样本上的起始与终止编号索引
    start_idx = [1; cum_counts(1:end-1)+1];
    end_idx = cum_counts;
    newSampleLens = zeros(size(sampleLens));
    % 对所有样本进行遍历，计算各个样本上筛选后剩余的光谱数量
    for i = 1:length(sampleLens)
        sampleSpectraIdx = start_idx(i):end_idx(i);
        retainedSpectraIdx = idx_high_quality(sampleSpectraIdx);
        newSampleLens(i) = sum(retainedSpectraIdx);
    end
end