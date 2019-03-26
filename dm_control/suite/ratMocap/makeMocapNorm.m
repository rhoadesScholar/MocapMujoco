function [markers_scaled, medianPose, g] = makeMocapNorm(inData, varName, scaleFactor) %%%%%%%%%%####PARENTALSCALING UNFINISHED
%     scaled marker data
%     load('ratMocap_baseGraph.mat', 'bg');
%     bg.Edges.Weight ./ (g.Edges.Weight/1000)
    
    
    [g, medianPose] = getGraph(inData, varName);
    if ~exist('scaleFactor', 'var')
        scaleFactor = 1.25;
    elseif isnumeric(scaleFactor)
        scaleFactor = scaleFactor/getFemur(g);
    elseif islogical(scaleFactor) && scaleFactor
        markers_scaled = parentalScaling(g, varName);
    elseif contains(fields(scaleFactor), 'bg')
        bg = scaleFactor.bg;
        spineIds = sum(contains(bg.Edges.EndNodes, 'Spine'), 2) == 2;
        scaleFactor = nanmean(bg.Edges.Weight(spineIds) ./ g.Edges.Weight(spineIds));
    end

    if isnumeric(scaleFactor)
        markers_preproc = inData.(varName);
        markers = inData.markernames;
        for m = 1:length(markers)
            markers_scaled.(markers{m}) = markers_preproc.(markers{m})*scaleFactor;
        end
    end
    
    return
end

function scaled = parentalScaling(g, varName)
    
%     kidCnt = zeros(max(links(:,1)), 1);
%     parents = 1:max(links(:,1));
%     for i = parents
%         kidCnt(i) = sum(links(:,1) == i);
%     end
%     
%     [~, parentRank] = sort(kidCnt', 'descend');    
%     parentRank = parentRank(1:end-sum(kidCnt==0));
%     
%     kidLst = {};
%     taken = false(length(links), 1);
%     nodeKidCnt = zeros(max(parentRank), 1);
%     for i = parentRank
%         kids = [links(:,1) == i] & ~taken;
%         if any(kids)
%             kidLst{i} = links(kids, 2)';
%             for ii = kidLst{i}
%                 taken = taken | [links(:,2) == ii];
%             end
%             nodeKidCnt(i) = length(kidLst{i});
%         end
%     end
end

function [g, medianPose] = getGraph(inData, varName)
%makes graph object with nodes and edges described by links, 
%and edge weights described by the median distance between markers in medianPose
    markernames = inData.markernames;
    medianPose = getMedianPose(inData.(varName));
    dst = dist(medianPose');
    
    links = cell2mat(inData.links');
    lnkDst = arrayfun(@(i) dst(links(i,1), links(i,2)), 1:length(links));
    
    
    g = graph(links(:,1), links(:,2), lnkDst, markernames);
end

function meanFemur = getFemur(g)
    kneeIs = contains(g.Edges.EndNodes,'knee','IgnoreCase',true);
    hipIs = contains(g.Edges.EndNodes,'hip','IgnoreCase',true);
    femurIs = find(sum([kneeIs, hipIs], 2) == 2);
    meanFemur = mean(g.Edges.Weight(femurIs));
end

function medianPose = getMedianPose(markers)
    mark = fields(markers);
    medianPose = zeros(length(mark), 3);
    for m = 1:length(mark)
        medianPose(m, :) = nanmedian(markers.(mark{m}))/1000;%ASSUMES NEEDED CONVERSION FROM MM to M
    end
    
    bilats = mark(arrayfun(@(y) strcmpi(y, 'L') || strcmpi(y, 'R'), cellfun(@(x) x(end), mark)));
    bilats = unique(cellfun(@(x) x(1:end-1), bilats, 'UniformOutput', false));
    bilats = bilats(cellfun(@(x) any(contains(mark, [x 'L'])) && any(contains(mark, [x 'R'])), bilats));
    for b = 1:length(bilats)
        bilatMedian = nanmedian(abs(medianPose(contains(mark, bilats{b}), :)));
        medianPose(contains(mark, [bilats{b} 'L']), :) = bilatMedian .* sign(medianPose(contains(mark, [bilats{b} 'L']), :));
        medianPose(contains(mark, [bilats{b} 'R']), :) = bilatMedian .* sign(medianPose(contains(mark, [bilats{b} 'R']), :));
    end
    medianPose(:, 3) = medianPose(:, 3) - min((medianPose(:,3)));%ASSUMES UP/DOWN IS 3RD COLUMN
    return
end

function medianPose = getMoveMedianPose(markers, move_frames)
    mark = fields(markers);
    for m = 1:length(mark)
        moveMarkers.(mark{m}) = markers.(mark{m})(move_frames, :);
    end
    medianPose = getMedianPose(moveMarkers);
    return
end

function g = getMedianGraph(inData, varName)
    markers = inData.(varName);
    mark = fields(markers);
    pose = nan(length(mark), size(markers.(mark{1}), 1), size(markers.(mark{1}), 2));
    parfor m = 1:length(mark)
        pose(m, :, :) = markers.(mark{m})/1000;%ASSUMES NEEDED CONVERSION FROM MM to M
    end
    
    dsts = nan(size(pose, 2), length(mark), length(mark));
    parfor f = 1:size(pose, 2)
        dsts(f, :, :) = dist(squeeze(pose(:, f, :))');
    end
    
    medianDst = squeeze(nanmedian(dsts));
    links = cell2mat(inData.links');
    lnkDst = arrayfun(@(i) medianDst(links(i,1), links(i,2)), 1:length(links));
    
    g = graph(links(:,1), links(:,2), lnkDst, mark);
    
    endNodes = g.Edges.EndNodes;
    
    bilats = mark(arrayfun(@(y) strcmpi(y, 'L') || strcmpi(y, 'R'), cellfun(@(x) x(end), mark)));
    bilats = unique(cellfun(@(x) x(1:end-1), bilats, 'UniformOutput', false));
    bilats = bilats(cellfun(@(x) any(contains(mark, [x 'L'])) && any(contains(mark, [x 'R'])), bilats));
    
    toDo = true(length(lnkDst), 1);
    for b = 1:length(bilats)
        leftsToDo = ((cellfun(@(y) contains(y, [bilats{b} 'L']), endNodes(:, 1)) ...
            + cellfun(@(y) contains(y,[bilats{b} 'L']), endNodes(:, 2))) > 0).*toDo > 0;
        
        rigtsToDo = ((cellfun(@(y) contains(y, [bilats{b} 'R']), endNodes(:, 1)) ...
            + cellfun(@(y) contains(y, [bilats{b} 'R']), endNodes(:, 2))) > 0).*toDo > 0;
        
        bilatMedian = nanmedian(abs(medianPose(contains(mark, bilats{b}), :)));
        medianPose(contains(mark, [bilats{b} 'L']), :) = bilatMedian .* sign(medianPose(contains(mark, [bilats{b} 'L']), :));
        medianPose(contains(mark, [bilats{b} 'R']), :) = bilatMedian .* sign(medianPose(contains(mark, [bilats{b} 'R']), :));
    end
    
end


