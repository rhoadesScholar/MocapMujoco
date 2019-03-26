function success = getBatchMujoRatSnippets(files, varargin)
    success = false(length(files), 1);
    
%     pc = parcluster('local');
%     pc.JobStorageLocation = tempdir;
%     parfor f = 1:length(files)
    for f = 1:length(files)
        fileName = getFile(files{f});
        success(f) = getMujoRatSnippets('fileName', fileName, varargin{:})
    end

%     delete(gcp('nocreate'));
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