function data = getMujoRatVid(varargin)
    
    %HARD CODE MODEL FEMUR LENGTH FOR NOW:
    modelFemurLength = 50.1473;
    
    %parse arguments
    par = inputParser;    
    addParameter(par, 'fileName', {}, @checkFile);
    addParameter(par, 'varName', {'markers_preproc'}, @isvarname);
    addParameter(par, 'fpsOut', {}, @isnumeric);
    addParameter(par, 'model_filename', {}, @checkFile);
    addParameter(par, 'maxRenderTime', {}, @isnumeric);
    addParameter(par, 'minRenderTime', {}, @isnumeric);
    addParameter(par, 'start_frame', {}, @isnumeric);
    addParameter(par, 'max_frame', {}, @isnumeric);
    addParameter(par, 'qvelMax', {}, @isnumeric);
    addParameter(par, 'qvelMean', {}, @isnumeric);
    addParameter(par, 'qaccMax', {}, @isnumeric);
    addParameter(par, 'qaccMean', {}, @isnumeric);    
    addParameter(par, 'play', false);
    
    parse(par, varargin{:})    
    vars = par.Parameters(structfun(@(x) ~isempty(x), par.Results));
    vars = vars(~[strcmpi(vars, 'fileName') + strcmpi(vars, 'varName')]);
    
    %get meta-data and normalize marker coordinates
    if isempty(par.Results.fileName)
        [inName, inPath] = uigetfile('*.mat', 'Choose marker data file');
        if isequal(inName, 0)
            data = 'Failure'
            return
        else
            inFile = fullfile(inPath, inName);
            inName = split(inName, '.');
            inName = inName{1};
        end
    else
        [inFile, inName, inPath] = getFile(par.Results.fileName);
    end
    
    fprintf('Loading point clouds....');
    try
        inData = load(inFile);
        disp('done.');
    catch
        disp('failed.');
        return
    end
    
    fprintf('Scaling point clouds....');
    try
        markers_preproc = inData.(par.Results.varName{1});
        markernames = inData.markernames;
        fps = inData.fps;    

        markers_scaled = makeMocapNorm(inData, par.Results.varName{1}, modelFemurLength);
        disp('done.');
    catch
        disp('failed.');
        return
    end
    
%     ###########################
    %save normalized coordinates (and metadata) in -v7
     
    fprintf('Saving scaled point clouds....');
    try   
        scaledFile = fullfile(inPath, [inName '_scaled.mat']);
        save(scaledFile, 'fps', 'markers_scaled', 'markernames', 'markers_preproc', '-v7')
        disp('done.');
    catch
        disp('failed.');
        return
    end
%     ###########################
    
    %setup and run python-mujoco rendering    
    parentPath = split(inPath, filesep);
    parentPath = join(parentPath(1:end-1), filesep);
    outPath = join([parentPath, "dataOutput"], filesep);
    outName = fullfile(outPath, inName + "vid_rendered" + datestr(datetime, 'yyyymmmdd_HHMM'));
    
    evalString = sprintf('python ratMocap.py --silent=True --fileName=%s --outName=%s --varName=markers_scaled'...
        , scaledFile, outName);    
    
    outName = [char(outName) '.mat'];
    for v = 1:length(vars)
        if contains(vars{v}, 'filename')
            var = getFile(par.Results.(vars{v}));
            evalString = sprintf('%s --%s=%s', evalString, vars{v}, var);
        else
            var = par.Results.(vars{v});
            if isnumeric(var)
                evalString = sprintf('%s --%s=%i', evalString, vars{v}, var);    
            elseif islogical(var)
                if var
                    evalString = sprintf('%s --%s=True', evalString, vars{v});
                else
                    evalString = sprintf('%s --%s=False', evalString, vars{v});
                end
            else
                evalString = sprintf('%s --%s=%s', evalString, vars{v}, var);                
            end
        end                    
    end
    
%     if count(py.sys.path,'') == 0
%         insert(py.sys.path, int32(0), '');
%     end
    insert(py.sys.path, int32(0), pwd);
    insert(py.sys.path, int32(0), '');
    disp('Rendering................');
    s = system(evalString);
    if s == 0
        disp('Rendering done.')
    else
        disp('Rendering failed.')
    end        
    
    %load python-mujoco rendered .mat and analyze
%     ##########################
    fprintf('Loading output......');
    try
        mujData = load(outName);
        disp('done.')
    catch
        fprintf('Failed to load mujoco output %s\n', outName);
    end
    
    fprintf('Analyzing output......')
    try
        analysis = postMujocoAnalysis(mujData);
        disp('done.')
        %save analysis
        %     ##########################        
            %decided to only save analysis data, could make one big final file
            %instead
        fprintf('Saving analyses.....')
        try
            analysisFile = [char(outName) '_analysis.mat'];
            save(analysisFile, 'analysis', '-v7')
            disp('done.')
        catch
            fprintf('Failed to save %s\n', analysisFile);
        end
    catch
        disp('failed.')
    end
end

function bool = checkFile(name)
    d = dir(['*' filesep name '*']);
    bool = ~isempty(d);
    return
end

function [fileFull, fileName, filePath] = getFile(name)
    d = dir(['*' filesep name '*']);
    fileFull = fullfile(d.folder, d.name);
    fileName = split(d.name, '.');
    fileName = fileName{1};
    filePath = d.folder;
    return
end