function [success] = playMocapSnippets(varargin)%renders snippets in temp files, then plays back with option to save (spacebar)
    success = false;
    tic
    
    %parse arguments
    par = inputParser;    
    addParameter(par, 'fileName', {}, @checkFile);
    addParameter(par, 'fpsOut', 30, @isnumeric);
    addParameter(par, 'model_filename', 'ratMocap.xml', @checkFile);%ASSUMES THAT THERE WILL BE A .MAT FILE WITH THE GRAPH OBJECT OF THE SKELETON BASE POSE    
    addParameter(par, 'start_frame', {}, @isnumeric);
    addParameter(par, 'max_frame', {}, @isnumeric);    
    addParameter(par, 'dpi', {}, @isnumeric);  
    addParameter(par, 'record', true, @islogical);    
    addParameter(par, 'play', false, @islogical);     
    addParameter(par, 'mocap', true, @islogical); 
    
    parse(par, varargin{:})    
    vars = par.Parameters(structfun(@(x) ~isempty(x), par.Results));
    vars = vars(~[strcmpi(vars, 'fileName') + strcmpi(vars, 'varName')]);
    parsed = par.Results;
    
    %make options string for python-mujoco call
    options = '';
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
        assign(vars{v}, var);  
    end
        
    if isempty(parsed.fileName)
        [inName, inPath] = uigetfile('*.mat', 'Choose snippet data file');
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
    
    fprintf('Loading snippets....');
    try
        inData = load(inFile);
        mujData = inData.mujData;
        analyses = inData.analysis;
        [model_filename, model_name, model_path] = getFile(parsed.model_filename);
        parsed.model_filename = model_filename;
        disp('done.');
    catch
        disp('failed.');
        return
    end
    
    
    %     ###########################
        %start parallel rendering loop
    fprintf('Rendering %i snippets....', length(mujData));
    
    tempVidFiles = {};
    pc = parcluster('local');
    pc.JobStorageLocation = tempdir;
    parfor s = 1:length(mujData)%SHOULD work, but keeps failing...
%     for s = 1:length(mujData)
        
    %     ###########################
        %save snippets coordinates (and metadata) in -v7
        try   
            if mocap
                tempSnippet = savetempsnippetMocap(mujData(s).fpsOut, mujData(s).mocap_pos, mujData(s).qpos, mujData(s).badFrame);
            else
                tempSnippet = savetempsnippet(mujData(s).fpsOut, mujData(s).qpos, mujData(s).badFrame);
            end

        %     ###########################    
            %setup and run python-mujoco rendering    
            tempVid = tempname;
            evalString = sprintf('python ratMocap_playSnippets.py --fileName=%s --outName=%s %s', tempSnippet, tempVid, options);    

            tempSnippet = [char(tempSnippet) '.mat'];

            insert(py.sys.path, int32(0), pwd);
            insert(py.sys.path, int32(0), '');
            sys = system(evalString);
            if sys ~= 0
                fprintf('Rendering #%i failed.\n', s)
            else                                
                tempVid = [char(tempVid) '.mp4'];
                tempVidFiles{s} = tempVid;
                %try to load
                try
                    vids{s} = loadVid(tempVid);
                catch
                    fprintf('Failed to load video output #%i, will try again outside parallel loop\n', s);
                end
            end     
            delete(tempSnippet)

        catch
            fprintf('Failed to save snippet temp file#%i\n', s);
        end
    end
    
    %load python-mujoco rendered .mat(s)
        %     ##########################
    for s = 1:length(tempVidFiles)
        if ~exist('vids', 'var') || length(vids) < s || isempty(vids{s})   
            try
                vids{s} = loadVid(tempVidFiles{s});
            catch
                fprintf('Failed to load video output #%i\n', s);
            end
        end
    end    
    
    %play/save everything
    %     ##########################        
    fig = figure;
    fig.KeyPressFcn = @(x, y) evalin('caller', 'stop = true;');
    fig.BusyAction = 'cancel';
    waitforbuttonpress
    success = 0;
    for s = 1:length(tempVidFiles)
        stop = false;
        try
            fig = showAnalyses(analyses, fig, s);
            while hasFrame(vids{1}) && ~stop %play video
                vidFrame = readFrame(vids{1});
                image(vidFrame);
                drawnow limitrate
                while (1/vids{1}.FrameRate) > toc
                end
            end
            if ~stop, waitforbuttonpress; end
            saved = strcmp(get(fig, 'CurrentCharacter'), ' ');%PRESS SPACEBAR TO SAVE
            fig.CurrentCharacter = '-';
            
            if saved
                if ~isfolder(sprintf('%s%s%s', inPath, filesep, inName))
                    mkdir(sprintf('%s%s%s', inPath, filesep, inName));
                end
                fprintf('Saving.....')
                saved = copyfile(tempVidFiles{s}, sprintf('%s%s%s%ssnippet%i.mp4', inPath, filesep, inName, filesep, s));
                if saved
                    disp('saved.')
                else
                    fprintf('Failed to save video #%i\n', s);
                end
            end
            success = success + 1;
        catch
            fprintf('Failed to play video #%i\n', s);
        end
        
        vids = vids(2:end);
    end
    
    tempFiles = tempVidFiles;%[tempScaledFiles tempMujocoFiles];
    cellfun(@(x) delete(x), tempFiles)
    toc
    delete(gcp('nocreate'));
    success = success/length(mujData);
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

function temp = savetempsnippetMocap(fpsOut, mocap_pos, qpos, badFrame) %#ok<INUSD>
    temp = tempname;
    save(temp, 'fpsOut', 'mocap_pos', 'qpos', 'badFrame', '-v7');
    return
end

function temp = savetempsnippet(fpsOut, qpos, badFrame) %#ok<INUSD>
    temp = tempname;
    save(temp, 'fpsOut', 'qpos', 'badFrame', '-v7');
    return
end

function vid = loadVid(tempMujoco)
    vid = VideoReader(tempMujoco);
    return
end
    
function assign(var, val)
    assignin('caller', var, val)
    return
end

function fig = showAnalyses(data, fig, s)
    variables = fields(data);
    variables = variables(cellfun(@(x) isstruct(data.(x)), variables));
    m = length(variables);
    n = length(variables) + 1;
    
    figure(fig)    
    for i = 1:length(variables)
        if s == 1
            subplot(m, n, n*i);
            hold on
            title(variables{i})
            xlim([0 (length(data.(variables{i}).names) + 1)])
            xticks(0:(length(data.(variables{i}).names) + 1))
            xticklabels([''; data.(variables{i}).names; ''])
            xticklabel_rotate()   
        else
            subplot(m, n, n*i)
            hold on
        end
        errorbar(data.(variables{i}).means(s, :), data.(variables{i}).variances(s, :))
    end
    
    %direct focus to movie portion of figure
    a = 1:m*n;
    a = a(mod(a, n) > 0);
    subplot(m, n, a)
    set(gca, 'Visible', 'off');
    set(gca, 'BusyAction', 'cancel')
    
    return
end