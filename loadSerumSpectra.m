% 加载指定 .mat 光谱数据文件，并以文件名称命名结构体 SerumSamples 
function SerumSamples = loadSerumSpectra(filename)
    % 去掉扩展名
    [~, name] = fileparts(filename);
    S = load(filename);
    % 以文件名给变量命名
    SerumSamples = S.SerumSamples;
end