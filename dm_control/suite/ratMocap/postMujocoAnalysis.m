function out = postMujocoAnalysis(vid, varargin)%can be called without argument for batch analysis of all vids within a folder
    %varargin takes cell arrays: variables and varNames, in that order
    if length(varargin) == 2
        variables = varargin{1};
        varNames = varargin{2};
    else
        variables = {'tendonLen' 'qpos'};
        varNames = {'tendonNames' 'qnames'};
    end
    
    if ~exist('vid', 'var') || isempty(vid)
        clear vid
        filePath = uigetdir;
        vids = dir(fullfile(filePath, '*vid*.mat'));
        vids = {vids(1:end).name};
        v1 = 1;
        for v0 = 1:length(vids)
            newVid = load(vids{v0});
            try
                vid(v1) = newVid;
                vidNames{v1} = vids{v0};
                v1 = v1 + 1;
            end
        end
    elseif isstruct(vid) && length(vid) > 1
        vidNames = arrayfun(@(x) sprintf('snippet#%i', x), 1:length(vid), 'UniformOutput', false);
    else
        vidNames{1} = vid.model;
    end
    
    for v = 1:length(vid)
        out.badFrames(v) = nansum(vid(v).badFrame);
        legendStr{v} = sprintf('%s (%i *bad* frame(s))', ...
            replace(replace(replace(vidNames{v}, '.mat', ''), '_', ':'), filesep, ':'), out.badFrames(v));
    end
    
    for i = 1:length(variables)
        figure
        hold on
        title(variables{i})
        xlim([0 (length(vid(1).(varNames{i})) + 1)])
        xticks(0:(length(vid(1).(varNames{i})) + 1))
        xticklabels([''; strip(string(vid(1).(varNames{i}))); ''])
        out.(variables{i}).names = strip(string(vid(1).(varNames{i})));
        xticklabel_rotate()   
        for v = 1:length(vid)
            errorbar(nanmean(vid(v).(variables{i})), nanvar(vid(v).(variables{i})))
            out.(variables{i}).means(v, :) = nanmean(vid(v).(variables{i}));
            out.(variables{i}).variances(v, :) = nanvar(vid(v).(variables{i}));
        end 
%         out.means.(variables{i}) = nanmean(out.(variables{i}).means);
%         out.variances.(variables{i}) = nanvar(out.(variables{i}).means);
%         legend(legendStr);
    end
    
    return
end
