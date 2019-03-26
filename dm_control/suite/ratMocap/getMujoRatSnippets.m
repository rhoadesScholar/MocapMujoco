function [success, data] = getMujoRatSnippets(varargin)
    success = false;
    tic
%     %HARD CODE MODEL FEMUR LENGTH FOR NOW:
%     modelFemurLength = 50.1473;
%     
    %parse arguments
    par = inputParser;    
    addParameter(par, 'fileName', {}, @checkFile);
    addParameter(par, 'varName', {'agg_preproc'}, @isvarname);
    addParameter(par, 'fpsOut', 60, @isnumeric);
    addParameter(par, 'model_filename', 'ratMocap.xml', @checkFile);%ASSUMES THAT THERE WILL BE A .MAT FILE WITH THE GRAPH OBJECT OF THE SKELETON BASE POSE
    addParameter(par, 'maxRenderTime', {}, @isnumeric);
    addParameter(par, 'minRenderTime', {}, @isnumeric);
    addParameter(par, 'qvelMax', {}, @isnumeric);
    addParameter(par, 'qvelMean', {}, @isnumeric);
    addParameter(par, 'qaccMax', {}, @isnumeric);
    addParameter(par, 'qaccMean', {}, @isnumeric);    
    addParameter(par, 'record', {}, @islogical);    
    addParameter(par, 'qOnly', {}, @islogical);     
    
    parse(par, varargin{:})    
    vars = par.Parameters(structfun(@(x) ~isempty(x), par.Results));
    vars = vars(~[strcmpi(vars, 'fileName') + strcmpi(vars, 'varName')]);
    parsed = par.Results;
    varName = parsed.varName{1};
    
    %make options string for python-mujoco call
    options = '--silent=True ';
    for v = 1:length(vars)
        if contains(vars{v}, 'filename')
            var = getFile(parsed.(vars{v}));
            options = sprintf('%s --%s=%s', options, vars{v}, var);
        else
            var = parsed.(vars{v});
            if isnumeric(var)
                options = sprintf('%s --%s=%f', options, vars{v}, var);    
            elseif islogical(var)
                if var
                    options = sprintf('%s --%s=True', options, vars{v});
                else
                    options = sprintf('%s --%s=False', options, vars{v});
                end
            else
                options = sprintf('%s --%s=%s', options, vars{v}, var);                
            end
        end                    
    end
        
    %get meta-data and normalize marker coordinates
    if isempty(parsed.fileName)
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
        [inFile, inName, inPath] = getFile(parsed.fileName);
    end
    
    fprintf('Loading point clouds....');
    try
        load(inFile, 'snippetstruct');
        inData = snippetstruct;
        markernames = inData{1}.mocapstruct_reduced_agg.markernames;        
        links = inData{1}.mocapstruct_reduced_agg.links;
        fps = inData{1}.mocapstruct_reduced_agg.fps/inData{1}.params.snippet_res;
        [model_filename, model_name, model_path] = getFile(parsed.model_filename);
        parsed.model_filename = model_filename;
        baseGraph = load([model_path filesep model_name '.mat'], 'bg');
        disp('done.');
    catch
        disp('failed.');
        return
    end
    
    
    %     ###########################
        %start parallel rendering loop
    fprintf('Rendering %i snippets....', length(inData));
    
    tempMujocoFiles = {};
    pc = parcluster('local');
    pc.JobStorageLocation = tempdir;
    parfor s = 1:length(inData)%should be doable as parfor...
        try
            markers_preproc = inData{s}.(varName);
            inData{s}.markernames = markernames;            
            inData{s}.links = links;
            markers_scaled = makeMocapNorm(inData{s}, varName, baseGraph);%CHANGE SCALING APPROACH
        
        %     ###########################
            %save normalized coordinates (and metadata) in -v7
            try   
                tempScaled = savetempscaled(fps, markers_scaled, markernames, markers_preproc);

            %     ###########################    
                %setup and run python-mujoco rendering    
                tempMujoco = tempname;
                evalString = sprintf('python ratMocap.py --fileName=%s --outName=%s --varName=markers_scaled %s', tempScaled, tempMujoco, options);    

                tempScaled = [char(tempScaled) '.mat'];
%                 tempScaledFiles{s} = tempScaled;

                insert(py.sys.path, int32(0), pwd);
                insert(py.sys.path, int32(0), '');
                sys = system(evalString);
                if sys ~= 0
                    fprintf('Rendering #%i failed.\n', s)
                else                                
                    tempMujoco = [char(tempMujoco) '.mat'];
                    tempMujocoFiles{s} = tempMujoco;
                    %try to load
                    try
                        mujData(s) = loadFile(tempMujoco);
                    catch
                        fprintf('Failed to load mujoco output #%i, will try again outside parallel loop\n', s);
                    end
                end     
                delete(tempScaled)
                
            catch
                fprintf('Failed to save scaled output #%i\n', s);
            end
        catch
            fprintf('Failed to scale #%i\n', s);
        end
    end
    
    %load python-mujoco rendered .mat(s)
        %     ##########################
    for s = 1:length(tempMujocoFiles)
        if ~exist('mujData', 'var') || length(mujData) < s || isempty(mujData(s))   
            try
                mujData(s) = load(tempMujocoFiles{s});
            catch
                fprintf('Failed to load mujoco output #%i\n', s);
            end
        end
    end
        %analyze python-mujoco rendered .mat(s)
    %     ##########################
    
    fprintf('Analyzing output......')
    try
        analysis = postMujocoAnalysis(mujData);
        disp('done.')        
    catch
        disp('failed.')
    end
    
    
    %save everything
    %     ##########################        
    fprintf('Saving.....')
    
    parentPath = split(inPath, filesep);
    parentPath = join(parentPath(1:end-1), filesep);
    outPath = join([parentPath, "dataOutput"], filesep);
    outName = fullfile(outPath, inName + "_snippetsRendered" + datestr(datetime, 'yyyymmmdd_HHMM'));
    try
        save(outName, 'mujData', 'analysis')
        disp('done.')
        success = true;
    catch
        fprintf('Failed to save %s\n', outName);
    end    
    
    tempFiles = tempMujocoFiles;%[tempScaledFiles tempMujocoFiles];
    cellfun(@(x) delete(x), tempFiles)
    load('gong.mat', 'y');
    sound(y)
    toc
    delete(gcp('nocreate'));
end

function bool = checkFile(name)
    d = dir(name);
    bool = ~isempty(d);
    if ~bool
        d = dir(['*' filesep name '*']);
        bool = ~isempty(d);
    end
    
    return
end

function [fileFull, fileName, filePath] = getFile(name)
    d = dir(name);
    if isempty(d)
        d = dir(['*' filesep name '*']);
    end

    fileFull = fullfile(d.folder, d.name);
    fileName = split(d.name, '.');
    fileName = fileName{1};
    filePath = d.folder;
    return
end

function temp = savetempscaled(fps, markers_scaled, markernames, markers_preproc) %#ok<INUSD>
    temp = tempname;
    save(temp, 'fps', 'markers_scaled', 'markernames', 'markers_preproc', '-v7');
    return
end

function mujData = loadFile(tempMujoco)
    mujData = load(tempMujoco);
end
    