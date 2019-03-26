function [outString, medianPose] = makeMocapMedian(markers_aligned_preproc, markernames, move_frames, base, scaleFactor)
    if isstruct(markers_aligned_preproc)
        if ~exist('scaleFactor', 'var')
            scaleFactor = 1;
        end
        medianPose = zeros(length(markernames), 3);
        for m = 1:length(markernames)
            moveMarkers.(markernames{m}) = markers_aligned_preproc.(markernames{m})(move_frames, :);
            medianPose(m, :) = (nanmedian(moveMarkers.(markernames{m}))/1000)*scaleFactor;
        end

        bilats = markernames(arrayfun(@(y) strcmpi(y, 'L') || strcmpi(y, 'R'), cellfun(@(x) x(end), markernames)));
        bilats = unique(cellfun(@(x) x(1:end-1), bilats, 'UniformOutput', false));
        bilats = bilats(cellfun(@(x) any(contains(markernames, [x 'L'])) && any(contains(markernames, [x 'R'])), bilats));
        for b = 1:length(bilats)
            bilatMedian = nanmedian(abs(medianPose(contains(markernames, bilats{b}), :)));
            medianPose(contains(markernames, [bilats{b} 'L']), :) = bilatMedian .* sign(medianPose(contains(markernames, [bilats{b} 'L']), :));
            medianPose(contains(markernames, [bilats{b} 'R']), :) = bilatMedian .* sign(medianPose(contains(markernames, [bilats{b} 'R']), :));
        end
        medianPose(:, 3) = medianPose(:, 3) - min((medianPose(:,3)));%ASSUMES UP/DOWN IS 3RD COLUMN

        outString = '';
        strings = getStrings(base);
        for m = 1:length(markernames)
            outString = sprintf('%s%s%f %f %f%s\n', outString, strings{m, 1}, medianPose(m, 1), medianPose(m, 2), medianPose(m, 3), strings{m, 2});
        end
    else
        outString = '';
        strings = getScaffold(base);
        pos = getScaffoldPos();
        for m = 1:size(pos, 1)
            outString = sprintf('%s%s%f %f %f%s\n', outString, strings{m, 1}, pos(m, 1), pos(m, 2), pos(m, 3), strings{m, 2});
        end        
    end

    return
end

function strings = getStrings(base)

    stringBase = {'<body name="HeadFm" mocap="true" pos="' '"><site name="HeadFm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadBm" mocap="true" pos="' '"><site name="HeadBm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadLm" mocap="true" pos="' '"><site name="HeadLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="SpineFm" mocap="true" pos="' '"><site name="SpineFm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineMm" mocap="true" pos="' '"><site name="SpineMm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineLm" mocap="true" pos="' '"><site name="SpineLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="Offset1m" mocap="true" pos="' '"><site name="Offset1m" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="Offset2m" mocap="true" pos="' '"><site name="Offset2m" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="HipLm" mocap="true" pos="' '"><site name="HipLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="HipRm" mocap="true" pos="' '"><site name="HipRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="ElbowLm" mocap="true" pos="' '"><site name="ElbowLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ArmLm" mocap="true" pos="' '"><site name="ArmLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderLm" mocap="true" pos="' '"><site name="ShoulderLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderRm" mocap="true" pos="' '"><site name="ShoulderRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ElbowRm" mocap="true" pos="' '"><site name="ElbowRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ArmRm" mocap="true" pos="' '"><site name="ArmRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="KneeRm" mocap="true" pos="' '"><site name="KneeRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="KneeLm" mocap="true" pos="' '"><site name="KneeLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinLm" mocap="true" pos="' '"><site name="ShinLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinRm" mocap="true" pos="' '"><site name="ShinRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'};

    stringMain = {'<body name="HeadF" mocap="true" pos="' '"><site name="HeadF" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadB" mocap="true" pos="' '"><site name="HeadB" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadL" mocap="true" pos="' '"><site name="HeadL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="SpineF" mocap="true" pos="' '"><site name="SpineF" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineM" mocap="true" pos="' '"><site name="SpineM" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineL" mocap="true" pos="' '"><site name="SpineL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="Offset1" mocap="true" pos="' '"><site name="Offset1" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="Offset2" mocap="true" pos="' '"><site name="Offset2" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="HipL" mocap="true" pos="' '"><site name="HipL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="HipR" mocap="true" pos="' '"><site name="HipR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="ElbowL" mocap="true" pos="' '"><site name="ElbowL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ArmL" mocap="true" pos="' '"><site name="ArmL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderL" mocap="true" pos="' '"><site name="ShoulderL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderR" mocap="true" pos="' '"><site name="ShoulderR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ElbowR" mocap="true" pos="' '"><site name="ElbowR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ArmR" mocap="true" pos="' '"><site name="ArmR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="KneeR" mocap="true" pos="' '"><site name="KneeR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="KneeL" mocap="true" pos="' '"><site name="KneeL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinL" mocap="true" pos="' '"><site name="ShinL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinR" mocap="true" pos="' '"><site name="ShinR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'};
    
    if base
        strings = stringBase;
    else
        strings = stringMain;
    end
    
    return
end

function strings = getScaffold(base)

    stringBase = {'<body name="SpineFm" mocap="true" pos="' '"><site name="SpineFm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="Offset1m" mocap="true" pos="' '"><site name="Offset1m" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="Offset2m" mocap="true" pos="' '"><site name="Offset2m" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="SpineMm" mocap="true" pos="' '"><site name="SpineMm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineLm" mocap="true" pos="' '"><site name="SpineLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="HipLm" mocap="true" pos="' '"><site name="HipLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="KneeLm" mocap="true" pos="' '"><site name="KneeLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinLm" mocap="true" pos="' '"><site name="ShinLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="HipRm" mocap="true" pos="' '"><site name="HipRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="KneeRm" mocap="true" pos="' '"><site name="KneeRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="ShinRm" mocap="true" pos="' '"><site name="ShinRm" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="HeadFm" mocap="true" pos="' '"><site name="HeadFm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadLm" mocap="true" pos="' '"><site name="HeadLm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadBm" mocap="true" pos="' '"><site name="HeadBm" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="ShoulderLm" mocap="true" pos="' '"><site name="ShoulderLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ElbowLm" mocap="true" pos="' '"><site name="ElbowLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'   
    '<body name="ArmLm" mocap="true" pos="' '"><site name="ArmLm" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderRm" mocap="true" pos="' '"><site name="ShoulderRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ElbowRm" mocap="true" pos="' '"><site name="ElbowRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ArmRm" mocap="true" pos="' '"><site name="ArmRm" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'};

    stringMain = {'<body name="SpineF" mocap="true" pos="' '"><site name="SpineF" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="Offset1" mocap="true" pos="' '"><site name="Offset1" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="Offset2" mocap="true" pos="' '"><site name="Offset2" type="sphere" size="0.001" pos="0 0 0" rgba="0.75 0 0.5 1"/></body>'
    '<body name="SpineM" mocap="true" pos="' '"><site name="SpineM" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="SpineL" mocap="true" pos="' '"><site name="SpineL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0 0 1"/></body>'
    '<body name="HipL" mocap="true" pos="' '"><site name="HipL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="KneeL" mocap="true" pos="' '"><site name="KneeL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="ShinL" mocap="true" pos="' '"><site name="ShinL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0.75 0.75 1"/></body>'
    '<body name="HipR" mocap="true" pos="' '"><site name="HipR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="KneeR" mocap="true" pos="' '"><site name="KneeR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="ShinR" mocap="true" pos="' '"><site name="ShinR" type="sphere" size="0.001" pos="0 0 0" rgba="0 1 0 1"/></body>'
    '<body name="HeadF" mocap="true" pos="' '"><site name="HeadF" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadL" mocap="true" pos="' '"><site name="HeadL" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="HeadB" mocap="true" pos="' '"><site name="HeadB" type="sphere" size="0.001" pos="0 0 0" rgba="0 0 1 1"/></body>'
    '<body name="ShoulderL" mocap="true" pos="' '"><site name="ShoulderL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ElbowL" mocap="true" pos="' '"><site name="ElbowL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'   
    '<body name="ArmL" mocap="true" pos="' '"><site name="ArmL" type="sphere" size="0.001" pos="0 0 0" rgba="1 0.75 0 1"/></body>'
    '<body name="ShoulderR" mocap="true" pos="' '"><site name="ShoulderR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ElbowR" mocap="true" pos="' '"><site name="ElbowR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'
    '<body name="ArmR" mocap="true" pos="' '"><site name="ArmR" type="sphere" size="0.001" pos="0 0 0" rgba="1 1 1 1"/></body>'};
    
    if base
        strings = stringBase;
    else
        strings = stringMain;
    end
    
    return
end

function pos = getScaffoldPos()
    
    pos = [0.024    -0.00039   0.085;    
        0.013     0.017     0.082;    
        -0.023     0.02      0.091;    
        -0.014    -0.00024   0.1;      
        -0.063    -3.7e-05   0.097;    
        -0.084     0.02      0.073;    
        -0.048     0.053     0.063;    
        -0.046     0.067     0.034;    
        -0.085    -0.019     0.073;    
        -0.049    -0.053     0.063;    
        -0.046    -0.067     0.034;    
        0.09     -0.025     0.09;     
        0.067     0.023     0.096;    
        0.067    -0.025     0.096;    
        0.034     0.016     0.064;    
        0.016     0.04      0.023;    
        0.027     0.039     0.016;    
        0.033    -0.017     0.064;    
        0.015    -0.041     0.023;    
        0.026    -0.039     0.016];

    return
end
