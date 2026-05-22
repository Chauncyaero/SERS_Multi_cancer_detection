% 此程序采用多级 CNN 模型对二维 DWT 特征进行识别
% 数据处理流程改编自 2D 数据的分类代码
close;
clear;
load('WaveletData11.mat')
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

% % 根据需求对光谱进行质量控制，之后对经过预筛选后的光谱矩阵进行移除标签或者修改标签等操作
% qualityScores = [Area1, Area2,diff1];
% [idx_high_quality, sampleLens] = qualityControl(sampleLens, qualityScores);
% 不进行质量控制
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

fprintf('Begin training CNN models...\n');

% 提取标签种类数目
uniqueLabels = unique(Y_categorical);
numCategories = numel(uniqueLabels);
% 定义混淆矩阵，第三维度为折数，第四维度为种子数
confusionMatrix = zeros(numCategories,numCategories,10,10);

% 根据需求决定是多种子计算还是单种子计算
% ROC曲线绘制需要在单种子下进行
seeds = [42, 123, 2023, 2024, 2025, 777, 888, 999, 7, 12345];
% seeds = [42];
num_seeds = length(seeds);

for seed_idx = 1:num_seeds
    rng(seeds(seed_idx));

    % 根据随机数种子，随机划分固定的测试集
    testRatio = 0.1;    % 测试集比例
    sampleLabels_cat = categorical(sampleLabels);
    [trainValIdx, testIdx, trainValSampleIdx] = splitSampleWiseTestSet(sampleLens, sampleLabels_cat, testRatio); % Sample Wise划分数据
    testData = X(:,:,:,testIdx);
    testLabels = Y_categorical(testIdx);
    test_spectrum_id = spectrum_id(testIdx);
    trainValData = X(:,:,:,trainValIdx);
    trainValLabels = Y_categorical(trainValIdx);
    trainValSampleLens = sampleLens(trainValSampleIdx); % 训练验证集同样需要样本级划分，因此需要保留sampleLens和sampleLabels信息
    trainValSampleLabels = sampleLabels_cat(trainValSampleIdx);

    % 剩余数据用于K折交叉验证
    K = 10;
    lr_L1 = 0.0005;
    MaxEpochs_L1 = 20;
    BatchSize_L1 = 64;
    L2_L1 = 0.0005;
    ConvChannel_L1 = 32;
    lr_L2 = 0.0005;
    MaxEpochs_L2 = 20;
    BatchSize_L2 = 64;
    L2_L2 = 0.0001;
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

        % 对第一级泛癌识别绘制 ROC 曲线
        [~,scores_L1] = classify(net_L1, testData);
        idxCancer = find(strcmp(categories(trainLabels_L1), '癌症'));
        y_pred_prob_L1 = scores_L1(:,idxCancer); % 在L1等级将癌症分类当作正类
        thresholds = flip(0:0.01:1);
        FPR_L1 = zeros(1,length(thresholds));
        TPR_L1 = zeros(1,length(thresholds));
        for k = 1:length(thresholds)
            thres = thresholds(k);
            preds_L1 = y_pred_prob_L1 >= thres;
            TP_L1 = sum((preds_L1 == 1) & (testLabels_L1 == '癌症'));
            FP_L1 = sum((preds_L1 == 1) & (testLabels_L1 == '正常'));
            TN_L1 = sum((preds_L1 == 0) & (testLabels_L1 == '正常'));
            FN_L1 = sum((preds_L1 == 0) & (testLabels_L1 == '癌症'));
            FPR_L1(k) = FP_L1 / (FP_L1 + TN_L1 + eps);
            TPR_L1(k) = TP_L1 / (TP_L1 + FN_L1 + eps);
        end
        % 进行插值
        standard_FPR = 0:0.01:1;
        % 插值前对计算的TPR，FPR进行清洗
        [FPR_L1_clean, TPR_L1_clean] = prepare_roc_for_interp(FPR_L1, TPR_L1);
        TPR_L1_interp = interp1(FPR_L1_clean, TPR_L1_clean, standard_FPR, 'linear', 'extrap');
        % 截断至 [0, 1] 范围
        TPR_L1_interp = max(0, min(1, TPR_L1_interp));
        all_TPR_L1 = [all_TPR_L1;TPR_L1_interp];


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

        % 对第二级泛癌识别绘制 ROC 曲线
        [~,scores_L2] = classify(net_L2, testData_L2);
        idxEC = find(strcmp(categories(trainLabels_L2), '食管癌'));
        idxLC = find(strcmp(categories(trainLabels_L2), '肺癌'));
        y_pred_prob_L2_EC = scores_L2(:,idxEC); 
        y_pred_prob_L2_LC = scores_L2(:,idxLC); % 在L2等级将两个癌种均作为正类计算两次
        thresholds = flip(0:0.01:1);

        % 食管癌 EC 作为正分类时
        FPR_L2_EC = zeros(1,length(thresholds));
        TPR_L2_EC = zeros(1,length(thresholds));
        for k = 1:length(thresholds)
            thres = thresholds(k);
            preds_L2_EC = y_pred_prob_L2_EC >= thres;
            TP_L2_EC = sum((preds_L2_EC == 1) & (testLabels_L2 == '食管癌'));
            FP_L2_EC = sum((preds_L2_EC == 1) & (testLabels_L2 == '肺癌'));
            TN_L2_EC = sum((preds_L2_EC == 0) & (testLabels_L2 == '肺癌'));
            FN_L2_EC = sum((preds_L2_EC == 0) & (testLabels_L2 == '食管癌'));
            FPR_L2_EC(k) = FP_L2_EC / (FP_L2_EC + TN_L2_EC + eps);
            TPR_L2_EC(k) = TP_L2_EC / (TP_L2_EC + FN_L2_EC + eps);
        end
        % 进行插值
        standard_FPR = 0:0.01:1;
        % 插值前对计算的TPR，FPR进行清洗
        [FPR_L2_EC_clean, TPR_L2_EC_clean] = prepare_roc_for_interp(FPR_L2_EC, TPR_L2_EC);
        TPR_L2_EC_interp = interp1(FPR_L2_EC_clean, TPR_L2_EC_clean, standard_FPR, 'linear', 'extrap');
        % 截断至 [0, 1] 范围
        TPR_L2_EC_interp = max(0, min(1, TPR_L2_EC_interp));
        all_TPR_L2_EC = [all_TPR_L2_EC;TPR_L2_EC_interp];

        % 肺癌 LC 作为正分类时
        FPR_L2_LC = zeros(1,length(thresholds));
        TPR_L2_LC = zeros(1,length(thresholds));
        for k = 1:length(thresholds)
            thres = thresholds(k);
            preds_L2_LC = y_pred_prob_L2_LC >= thres;
            TP_L2_LC = sum((preds_L2_LC == 1) & (testLabels_L2 == '肺癌'));
            FP_L2_LC = sum((preds_L2_LC == 1) & (testLabels_L2 == '食管癌'));
            TN_L2_LC = sum((preds_L2_LC == 0) & (testLabels_L2 == '食管癌'));
            FN_L2_LC = sum((preds_L2_LC == 0) & (testLabels_L2 == '肺癌'));
            FPR_L2_LC(k) = FP_L2_LC / (FP_L2_LC + TN_L2_LC + eps);
            TPR_L2_LC(k) = TP_L2_LC / (TP_L2_LC + FN_L2_LC + eps);
        end
        % 进行插值
        standard_FPR = 0:0.01:1;
        % 插值前对计算的TPR，FPR进行清洗
        [FPR_L2_LC_clean, TPR_L2_LC_clean] = prepare_roc_for_interp(FPR_L2_LC, TPR_L2_LC);
        TPR_L2_LC_interp = interp1(FPR_L2_LC_clean, TPR_L2_LC_clean, standard_FPR, 'linear', 'extrap');
        % 截断至 [0, 1] 范围
        TPR_L2_LC_interp = max(0, min(1, TPR_L2_LC_interp));
        all_TPR_L2_LC = [all_TPR_L2_LC;TPR_L2_LC_interp];

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

    % 绘制三条 ROC 曲线，多种子下可注释
    mean_TPR_L1 = mean(all_TPR_L1)';
    mean_TPR_L2_EC = mean(all_TPR_L2_EC)';
    mean_TPR_L2_LC = mean(all_TPR_L2_LC)';
    standard_FPR = [0;standard_FPR'];
    mean_TPR_L1 = [0;mean_TPR_L1];
    mean_TPR_L2_EC = [0;mean_TPR_L2_EC];
    mean_TPR_L2_LC = [0;mean_TPR_L2_LC];
    figure
    plot(standard_FPR,mean_TPR_L1,'g','LineWidth',2);
    hold on;
    plot(standard_FPR,mean_TPR_L2_EC,'r','LineWidth',2); % 红色表示 EC
    plot(standard_FPR,mean_TPR_L2_LC,'Color', [1 0.5 0],'LineWidth',2); % 橙色表示 LC
    hold off;
    AUC_L1 = trapz(standard_FPR,mean_TPR_L1);
    AUC_L2_EC = trapz(standard_FPR,mean_TPR_L2_EC);
    AUC_L2_LC = trapz(standard_FPR,mean_TPR_L2_LC);
    title(['ROC Curve (AUC_L_1 = ', num2str(AUC_L1), ', AUC_L_2 = ', num2str(AUC_L2_EC),', ',num2str(AUC_L2_LC),')']);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');

    % 输出每个 seed 训练的结果
    disp([...
        'Seed ', num2str(seeds(seed_idx)), ' Complete. ', ...
        'Overall Test Accuracy = ', num2str(averageTestAccuracy(seed_idx), '%.4f')]);
end
% 输出采用所有 seed 训练后的平均结果
disp(['All seeds completed.', ...
      'Average Overall Test Accuracy =', num2str(mean(averageTestAccuracy),'%.4f')]);

% % 保存混淆矩阵
% save('confusion_matrix_2D_hierachical.mat','confusionMatrix','uniqueLabels');

%% 对每个Fold，每个Seed的混淆矩阵求和，调换肺癌和食管癌的位置
cm_sum = sum(confusionMatrix, [3 4]);
swap_order = [1 3 2];
cm_workflow = cm_sum(swap_order, swap_order);
fprintf('Aggregated Confusion Matrix\n');
disp(cm_workflow)
