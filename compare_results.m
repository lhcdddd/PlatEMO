% Script to compare IGD performance of REMO and REMOss on DTLZ2

% Define paths
path_REMO = 'e:\PlatEMO\PlatEMO\Data\REMO\';
path_REMOss = 'e:\PlatEMO\PlatEMO\Data\REMOss\';

% Number of runs
runs = 30;
igd_REMO = zeros(runs, 1);
igd_REMOss = zeros(runs, 1);

fprintf('Calculating IGD for 30 runs...\n');

for i = 1:runs
    % Load REMO data
    file_REMO = fullfile(path_REMO, sprintf('REMO_DTLZ2_M3_D10_%d.mat', i));
    if isfile(file_REMO)
        data_REMO = load(file_REMO);
        % In PlatEMO, the final population is typically an array of SOLUTION objects
        if iscell(data_REMO.result)
            pop = data_REMO.result{end};
        else
            pop = data_REMO.result;
        end
        % Extract objs via method or property
        try
            objs = pop.objs;
        catch
            objs = [pop.obj];
        end
        
        % Calculate IGD (True PF for DTLZ2 can be generated via uniform points)
        PF = UniformPoint(10000, 3);
        PF = PF ./ repmat(sqrt(sum(PF.^2, 2)), 1, 3); % DTLZ2 PF is a sphere
        
        Distance = min(pdist2(PF, objs), [], 2);
        igd_REMO(i) = mean(Distance);
    end
    
    % Load REMOss data
    file_REMOss = fullfile(path_REMOss, sprintf('REMOss_DTLZ2_M3_D10_%d.mat', i));
    if isfile(file_REMOss)
        data_REMOss = load(file_REMOss);
        if iscell(data_REMOss.result)
            pop = data_REMOss.result{end};
        else
            pop = data_REMOss.result;
        end
        try
            objs = pop.objs;
        catch
            objs = [pop.obj];
        end
        
        Distance = min(pdist2(PF, objs), [], 2);
        igd_REMOss(i) = mean(Distance);
    end
end

% Filter out zeros in case some files were missing
igd_REMO = igd_REMO(igd_REMO > 0);
igd_REMOss = igd_REMOss(igd_REMOss > 0);

% Compute statistics
mean_REMO = mean(igd_REMO);
std_REMO = std(igd_REMO);
mean_REMOss = mean(igd_REMOss);
std_REMOss = std(igd_REMOss);

fprintf('\n--- Performance Comparison (IGD on DTLZ2, M=3, D=10) ---\n');
fprintf('Algorithm | Mean IGD  | Std Dev\n');
fprintf('---------------------------------\n');
fprintf('REMO      | %.6f | %.6f\n', mean_REMO, std_REMO);
fprintf('REMOss    | %.6f | %.6f\n', mean_REMOss, std_REMOss);

% Wilcoxon rank sum test for statistical significance
p_value = ranksum(igd_REMO, igd_REMOss);
fprintf('\nWilcoxon rank sum test p-value: %.4e\n', p_value);
if p_value < 0.05
    if mean_REMOss < mean_REMO
        fprintf('Conclusion: REMOss is significantly BETTER than REMO.\n');
    else
        fprintf('Conclusion: REMOss is significantly WORSE than REMO.\n');
    end
else
    fprintf('Conclusion: No significant difference between the two algorithms.\n');
end

exit;
