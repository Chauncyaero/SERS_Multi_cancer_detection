% 此程序参照对混淆矩阵的统计方式，采用Grad-CAM，对2D数据的importance矩阵进行计算和统计
% 数据处理流程改编自 2D 数据的分级识别代码
close;
clear;
load('WaveletData11.mat') % 小波分解数据储存在上一级文件夹中
X = permute(reconSpectra,[3,2,1]);  % 交换重构矩阵的维度，从原来的[样本 波数 等级]变为[等级 波数 样本]
X = reshape(X, [size(X,1),size(X,2),1,size(X,3)]);  % 插入通道维度，以满足网络输入要求
Y = labels;

% 归一化处理（每通道独立归一化）
for i = 1:size(X,1)
    for j = 1:size(X,3)
        for k = 1:size(X,4)
            channel_data = X(i,:,j,k);
            X(i,:,j,k) = (channel_data - mean(channel_data(:))) / std(channel_data(:));
        end
    end
end

% % 先进行质量控制，之后对经过预筛选后的光谱矩阵进行移除标签或者修改标签等操作
% 数据进行质量控制
qualityScores = [Area1, Area2,diff1];
% [idx_high_quality, sampleLens] = qualityControl(sampleLens, qualityScores);
% % 调试所用，不进行质量控制，全部设为高质量光谱
idx_high_quality = true(length(labels), 1);
idx_low_quality = ~idx_high_quality;
removed_spectrum_id = spectrum_id(idx_low_quality);

% 根据质量控制后的索引，筛选所需的数据
X = X(:,:,:,idx_high_quality);
Y = Y(idx_high_quality);
spectrum_id = spectrum_id(idx_high_quality);

% 保留原始标签信息，方便后续快速映射
Y_original = Y; 
% 将标签转换为分类格式
Y_categorical = categorical(Y);

classes = categories(Y_categorical);
n_classes = length(classes);

fprintf('开始训练CNN模型...\n');

% 提取标签种类数目
uniqueLabels = unique(Y_categorical);
numCategories = numel(uniqueLabels);

% 根据需求决定是多种子计算还是单种子计算
% ROC曲线绘制需要在单种子下进行
seeds = [42, 123, 2023, 2024, 2025, 777, 888, 999, 7, 12345];
% seeds = [42];
num_seeds = length(seeds);

% 定义混淆矩阵，第三维度为折数，第四维度为种子数
confusionMatrix = zeros(numCategories,numCategories,10,num_seeds);
% 参照混淆矩阵的计算和统计方法，计算importance矩阵
importance_Normal = zeros(size(X,1),size(X,2),10,num_seeds);
importance_Cancer = zeros(size(X,1),size(X,2),10,num_seeds);
importance_EC = zeros(size(X,1),size(X,2),10,num_seeds);
importance_LC = zeros(size(X,1),size(X,2),10,num_seeds);

for seed_idx = 1:num_seeds
    rng(seeds(seed_idx));

    % 随机划分固定的测试集
    testRatio = 0.2;    % 测试集比例
    sampleLabels_cat = categorical(sampleLabels);
    [trainValIdx, testIdx, trainValSampleIdx, testSampleIdx] = splitSampleWiseTestSet(sampleLens, sampleLabels_cat, testRatio); % Sample Wise划分数据
    testData = X(:,:,:,testIdx);
    testLabels = Y_categorical(testIdx);
    test_spectrum_id = spectrum_id(testIdx);
    trainValData = X(:,:,:,trainValIdx);
    trainValLabels = Y_categorical(trainValIdx);
    trainValSampleLens = sampleLens(trainValSampleIdx); % 训练验证集同样需要样本级划分，因此需要保留sampleLens和sampleLabels信息
    trainValSampleLabels = sampleLabels_cat(trainValSampleIdx);

    % 剩余数据用于K折交叉验证
    K = 10;
    lr_L1 = 0.001;
    MaxEpochs_L1 = 20;
    BatchSize_L1 = 16;
    L2_L1 = 0.01;
    ConvChannel_L1 = 64;
    lr_L2 = 0.00005;
    MaxEpochs_L2 = 40;
    BatchSize_L2 = 16;
    L2_L2 = 0.5;
    ConvChannel_L2 = 32;
    % 仅对训练+验证部分做K-fold
    % 用cvpartition对训练验证集做样本级的分层交叉验证划分
    cv = cvpartition(trainValSampleLabels, 'KFold', K);

    % 初始化多级监控指标
    train_Accuracy_L1 = zeros(K,1);
    val_Accuracy_L1 = zeros(K,1);
    test_Accuracy_L1 = zeros(K,1);
    all_TPR_L1 = [];
    train_Accuracy_L2 = zeros(K,1);
    val_Accuracy_L2 = zeros(K,1);
    test_Accuracy_L2 = zeros(K,1);
    all_TPR_L2_EC = [];
    all_TPR_L2_LC = [];

    test_Accuracy_overall = zeros(K,1);

    for i = 1:K
        % 获取当前fold中的训练集和验证集样本索引
        trainSampleIdx = training(cv,i);
        valSampleIdx = test(cv,i);

        % 提取当前fold中训练集和验证集的光谱索引
        % 计算训练测试集上每个样本光谱的起始与结束索引号
        sampleStartIdx = cumsum([1;trainValSampleLens(1:end-1)]);
        sampleEndIdx = cumsum(trainValSampleLens);
        % 首先将验证集索引初始化为零,索引向量长度为训练验证集上的光谱长度
        valIdx = false(length(trainValLabels),1);  
        % 将各验证集样本对应的光谱索引号置一
        for j = find(valSampleIdx') 
            valIdx(sampleStartIdx(j):sampleEndIdx(j)) = true;
        end
        trainIdx = ~valIdx;

        % 提取带有原始三分类标签的训练集和验证集数据
        trainData = trainValData(:,:,:,trainIdx);
        trainLabels = trainValLabels(trainIdx);
        valData = trainValData(:,:,:,valIdx);
        valLabels = trainValLabels(valIdx);

        % 构造第一级分类数据和标签
        % 数据本质上不需要变化，标签需要将食管癌与肺癌合并为癌症
        trainData_L1 = trainData;
        trainLabels_L1 = mergecats(trainLabels, {'食管癌', '肺癌'}, '癌症');
        valData_L1 = valData;
        valLabels_L1 = mergecats(valLabels, {'食管癌', '肺癌'}, '癌症');

        % 构造第二级分类数据和标签
        % 本质上是剔除正常标签的数据，标签继承自原始标签即可，索引从一级标签中获取 癌症 标签的位置
        isCancer = (trainLabels_L1 == '癌症');
        trainData_L2 = trainData(:,:,:,isCancer);
        trainLabels_L2 = trainLabels(isCancer);
        trainLabels_L2 = removecats(trainLabels_L2); % 删除幽灵类别
        isCancer = (valLabels_L1 == '癌症');
        valData_L2 = valData(:,:,:,isCancer);
        valLabels_L2 = valLabels(isCancer);
        valLabels_L2 = removecats(valLabels_L2); % 删除幽灵类别

        % 定义并训练第一级 CNN，输出 正常 与 癌症
        layers_L1 = [
            % 输入层
            imageInputLayer([size(X,1) size(X,2) size(X,3)], 'Name', 'input', 'Normalization', 'none')
            % 第一个卷积块
            convolution2dLayer(3, ConvChannel_L1, 'Padding', 'same', 'Name', 'conv1')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')
            maxPooling2dLayer(2, 'Stride', [2, 2], 'Name', 'pool1')
            % 第二个卷积块
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool2')
            % 第三个卷积块
            convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')
            maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool3')
            % 全连接层
            fullyConnectedLayer(256, 'Name', 'fc1')
            fullyConnectedLayer(128, 'Name', 'fc2')
            % 输出层
            fullyConnectedLayer(2, 'Name', 'fc_final')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
            ];
        options_L1 = trainingOptions('adam', ...
            'InitialLearnRate', lr_L1, ...
            'MaxEpochs', MaxEpochs_L1, ...
            'MiniBatchSize', BatchSize_L1, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'VerboseFrequency', 10, ...
            'ExecutionEnvironment', 'gpu', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 10, ...
            'L2Regularization', L2_L1);
        % 模型训练
        net_L1 = trainNetwork(trainData_L1, trainLabels_L1, layers_L1, options_L1);
        net_dl_L1 = dlnetwork(layerGraph(layers_L1(1:end-1))); % 去掉最后classification层

        % 定义并训练第二级 CNN，输出 食管癌 与 肺癌
        if ~isempty(trainData_L2)
            layers_L2 = [   % 结构与 L1 相同，仅输入数据不同
                imageInputLayer([size(X,1) size(X,2) size(X,3)], 'Name', 'input', 'Normalization', 'none')
                convolution2dLayer(3, ConvChannel_L2, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool1')
                convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool2')
                convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
                batchNormalizationLayer('Name', 'bn3')
                reluLayer('Name', 'relu3')
                maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool3')
                fullyConnectedLayer(256, 'Name', 'fc1')
                fullyConnectedLayer(128, 'Name', 'fc2')
                fullyConnectedLayer(2, 'Name', 'fc_final')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            options_L2 = trainingOptions('adam', ...
            'InitialLearnRate', lr_L2, ...
            'MaxEpochs', MaxEpochs_L2, ...
            'MiniBatchSize', BatchSize_L2, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'VerboseFrequency', 10, ...
            'ExecutionEnvironment', 'gpu', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 10, ...
            'L2Regularization', L2_L2);
            net_L2 = trainNetwork(trainData_L2, trainLabels_L2, layers_L2, options_L2);
            net_dl_L2 = dlnetwork(layerGraph(layers_L2(1:end-1)));
        else
            net_L2 = [];
        end

        % 计算第一级监控指标
        predictedLabels_L1_train = classify(net_L1,trainData_L1);
        train_Accuracy_L1(i) = sum(predictedLabels_L1_train == trainLabels_L1) / numel(trainLabels_L1);
        predictedLabels_L1_val = classify(net_L1, valData_L1);
        val_Accuracy_L1(i) = sum(predictedLabels_L1_val == valLabels_L1) / numel(valLabels_L1);
        % 由于第一级网络的性质，改造第一级的测试集标签
        testLabels_L1 = mergecats(testLabels, {'食管癌', '肺癌'}, '癌症');
        predictedLabels_L1_test = classify(net_L1, testData);
        test_Accuracy_L1(i) = sum(predictedLabels_L1_test == testLabels_L1) / numel(testLabels_L1);

        % 采用Grad-CAM计算L1网络的importance矩阵
        % 提取正常与癌症光谱的 idx
        correctMask = predictedLabels_L1_test == testLabels_L1;
        normalMask = (testLabels_L1 == '正常') & correctMask;
        cancerMask = (testLabels_L1 == '癌症') & correctMask;
        normalIdx = find(normalMask);
        cancerIdx = find(cancerMask);
        classNames_L1 = net_L1.Layers(end).Classes;
        classIdx_normal = find(classNames_L1 == '正常'); 
        classIdx_cancer = find(classNames_L1 == '癌症');

        % 预分配测试集上同类别importance矩阵的内存
        importance_n = zeros(size(X,1),size(X,2),numel(normalIdx));
        importance_c = zeros(size(X,1),size(X,2),numel(cancerIdx));

        % 对 Normal 光谱计算 CAM
        for n = 1:numel(normalIdx)
            idx = normalIdx(n);
            feature = testData(:,:,:,idx);
            % Grad-CAM
            importance = gradCAM(net_dl_L1, feature, classIdx_normal, 'ReductionLayer', 'fc_final','FeatureLayer', 'pool1');
            % ReLU
            importance = max(importance,0);
            % min max 归一化
            importance = importance - min(importance(:));
            imp_max = max(importance(:));          % 将 importance 展成一维后求全局最大值
            if imp_max > 0                         % 防止除以零
                importance = importance / imp_max;
                % 存入矩阵 importance_n 中
                importance_n(:,:,n) = importance;
            else
                warning('测试集正常光谱 %d 的 Grad-CAM 最大值非正数（max = %.4f），跳过归一化。', idx, imp_max);
                continue;   % 跳过当前光谱，继续下一个
            end
        end
        importance_Normal(:,:,i,seed_idx) = mean(importance_n,3);
        % 对 Cancer 光谱计算 CAM
        for n = 1:numel(cancerIdx)
            idx = cancerIdx(n);
            feature = testData(:,:,:,idx);
            % Grad-CAM
            importance = gradCAM(net_dl_L1, feature, classIdx_cancer, 'ReductionLayer', 'fc_final','FeatureLayer', 'pool1');
            % ReLU
            importance = max(importance,0);
            % min max 归一化
            importance = importance - min(importance(:));
            imp_max = max(importance(:));          % 将 importance 展成一维后求全局最大值
            if imp_max > 0                         % 防止除以零
                importance = importance / imp_max;
                % 存入矩阵 importance_c 中
                importance_c(:,:,n) = importance;
            else
                warning('测试集癌症光谱 %d 的 Grad-CAM 最大值非正数（max = %.4f），跳过归一化。', idx, imp_max);
                continue;   % 跳过当前光谱，继续下一个
            end
        end
        importance_Cancer(:,:,i,seed_idx) = mean(importance_c,3);


        % 计算第二级监控指标，此处计算第二级的独立测试准确率（即理想条件，输入样本全部为癌变）
        predictedLabels_L2_train = classify(net_L2,trainData_L2);
        train_Accuracy_L2(i) = sum(predictedLabels_L2_train == trainLabels_L2) / numel(trainLabels_L2);
        predictedLabels_L2_val = classify(net_L2, valData_L2);
        val_Accuracy_L2(i) = sum(predictedLabels_L2_val == valLabels_L2) / numel(valLabels_L2);
        % 由于第二级网络的性质，改造第二级的测试集数据及标签
        isCancer = (testLabels_L1 == '癌症');
        testData_L2 = testData(:,:,:,isCancer); % 将测试集中客观存在为癌症的光谱输入第二级
        testLabels_L2 = testLabels(isCancer);
        test_L2 = removecats(testLabels_L2); % 删除幽灵类别
        predictedLabels_L2_test = classify(net_L2, testData_L2);
        test_Accuracy_L2(i) = sum(predictedLabels_L2_test == testLabels_L2) / numel(testLabels_L2);

        % 采用Grad-CAM计算L2网络的importance矩阵
        % 提取食管癌与肺癌光谱的 idx
        correctMask = predictedLabels_L2_test == testLabels_L2;
        esophagealMask = (testLabels_L2 == '食管癌') & correctMask;
        lungMask = (testLabels_L2 == '肺癌') & correctMask;
        esophagealIdx = find(esophagealMask);
        lungIdx = find(lungMask);
        classNames_L2 = net_L2.Layers(end).Classes;
        classIdx_esophageal = find(classNames_L2 == '食管癌'); 
        classIdx_lung = find(classNames_L2 == '肺癌');

        % 预分配测试集上同类别importance矩阵的内存
        importance_esophageal = zeros(size(X,1),size(X,2),numel(esophagealIdx));
        importance_lung = zeros(size(X,1),size(X,2),numel(lungIdx));

        % 对 Esophageal 光谱计算 CAM
        for n = 1:numel(esophagealIdx)
            idx = esophagealIdx(n);
            feature = testData_L2(:,:,:,idx);
            % Grad-CAM
            importance = gradCAM(net_dl_L2, feature, classIdx_esophageal, 'ReductionLayer', 'fc_final','FeatureLayer', 'pool1');
            % ReLU
            importance = max(importance,0);
            % min max 归一化
            importance = importance - min(importance(:));
            imp_max = max(importance(:));          % 将 importance 展成一维后求全局最大值
            if imp_max > 0                         % 防止除以零
                importance = importance / imp_max;
                % 存入矩阵 importance_esophageal 中
                importance_esophageal(:,:,n) = importance;
            else
                warning('测试集食管癌光谱 %d 的 Grad-CAM 最大值非正数（max = %.4f），跳过归一化。', idx, imp_max);
                continue;   % 跳过当前光谱，继续下一个
            end
        end
        importance_EC(:,:,i,seed_idx) = mean(importance_esophageal,3);
        % 对 Lung 光谱计算 CAM
        for n = 1:numel(lungIdx)
            idx = lungIdx(n);
            feature = testData_L2(:,:,:,idx);
            % Grad-CAM
            importance = gradCAM(net_dl_L2, feature, classIdx_lung, 'ReductionLayer', 'fc_final','FeatureLayer', 'pool1');
            % ReLU
            importance = max(importance,0);
            % min max 归一化
            importance = importance - min(importance(:));
            imp_max = max(importance(:));          % 将 importance 展成一维后求全局最大值
            if imp_max > 0                         % 防止除以零
                importance = importance / imp_max;
                % 存入矩阵 importance_lung 中
                importance_lung(:,:,n) = importance;
            else
                warning('测试集肺癌光谱 %d 的 Grad-CAM 最大值非正数（max = %.4f），跳过归一化。', idx, imp_max);
                continue;   % 跳过当前光谱，继续下一个
            end
        end
        importance_LC(:,:,i,seed_idx) = mean(importance_lung,3);


        % 计算整体的级联指标
        pred_L1 = classify(net_L1, testData);
        % 最终预测标签初始化
        final_pred = categorical(repmat({'正常'}, size(pred_L1)), categories(testLabels));
        isCancer = (pred_L1 == '癌症');
        if any(isCancer) % 存在被第一级预测为癌症的光谱
            testData_L2 = testData(:,:,:,isCancer); % 将第一级网络识别为癌症的光谱输入第二级
            pred_L2 = classify(net_L2,testData_L2); % 第二级网络输出为食管癌或肺癌
            final_pred(isCancer) = pred_L2; % 将第二级网络的识别结果映射回整体预测标签中
            test_Accuracy_overall(i) = sum(final_pred == testLabels) / numel(testLabels);
        end

        % 计算测试集上的混淆矩阵
        cm = confusionmat(testLabels, final_pred);
        confusionMatrix(:,:,i,seed_idx) = cm;

        % 输出每个 fold 训练的结果
        % 构造第一行前缀（Fold Complete. 部分）
        foldPrefix = sprintf('%d Fold Complete. ', i);
        % 输出第一行
        fprintf('%sLevel 1 Validation Accuracy = %.4f, Test Accuracy = %.4f\n', foldPrefix, val_Accuracy_L1(i), test_Accuracy_L1(i));
        % 输出第二行，用与 foldPrefix 等宽的空格对齐
        fprintf('%sLevel 2 Validation Accuracy = %.4f, Test Accuracy = %.4f, Overall Test Accuracy = %.4f\n', ...
            repmat(' ', 1, length(foldPrefix)), val_Accuracy_L2(i), test_Accuracy_L2(i), test_Accuracy_overall(i));
    end

    % 10CV结束后计算平均指标
    averageTestAccuracy(seed_idx) = mean(test_Accuracy_overall);

    % 输出每个 seed 训练的结果
    disp([...
        'Seed ', num2str(seeds(seed_idx)), ' Complete. ', ...
        'Overall Test Accuracy = ', num2str(averageTestAccuracy(seed_idx), '%.4f')]);
end
% 输出采用所有 seed 训练后的平均结果
disp(['All seeds completed.', ...
      'Average Overall Test Accuracy =', num2str(mean(averageTestAccuracy),'%.4f')]);

% 保存 Importance 数据
save('Importance_2D_hierarchical.mat','importance_Normal','importance_Cancer','importance_EC','importance_LC');


