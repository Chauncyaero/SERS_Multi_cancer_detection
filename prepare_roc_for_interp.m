function [FPR_unique, TPR_unique] = prepare_roc_for_interp(FPR_raw, TPR_raw)
    % 确保列向量
    FPR_raw = FPR_raw(:);
    TPR_raw = TPR_raw(:);
    
    % 按 FPR 升序排列（通常 roc_curve 返回已经是升序，但以防万一）
    [FPR_sorted, idx] = sort(FPR_raw);
    TPR_sorted = TPR_raw(idx);
    
    % 去重：对于相同的 FPR，保留最大的 TPR（保证 ROC 曲线在上方）
    [FPR_unique, ia, ~] = unique(FPR_sorted, 'last');
    TPR_unique = TPR_sorted(ia);
    
    % 可选：强制起点为 (0, ?) 和终点为 (1, ?) 以确保插值不越界
    if FPR_unique(1) > 0
        FPR_unique = [0; FPR_unique];
        TPR_unique = [0; TPR_unique];   % 或使用 TPR_unique(1)
    end
    if FPR_unique(end) < 1
        FPR_unique = [FPR_unique; 1];
        TPR_unique = [TPR_unique; 1];
    end
end