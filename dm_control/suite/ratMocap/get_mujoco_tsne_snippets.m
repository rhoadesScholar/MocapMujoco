function get_mujoco_tsne_snippets(analysisstruct,savedirectory)

    % collect 8 s snippets around points for a single condition
    if ~exist('savedirectory', 'var')
        savedirectory = 'Y:\Jesse\Data\mujoco_snippets\';
    end
    conditionhere = 1;
    filename = strcat('mujocosnippets_',analysisstruct.conditionnames{conditionhere},'.mat');

    % get 8s, 60 Hz snippets
    params.snippet_size = 300*4;
    params.snippet_res = 5;
    params.snippet_frac = 1;

    [agg_mocap_structs,agg_snippetinds,agg_mocap_preproc] = collect_mocap_snippets_allzvals(analysisstruct,conditionhere,params);

    analysisstruct.agg_mocap_structs_snippets=agg_mocap_structs;
    analysisstruct.agg_snippetinds=agg_snippetinds;
    analysisstruct.agg_preproc=agg_mocap_preproc;

    snippetstruct = cell(1,analysisstruct.density_objects);
    for kk =1:analysisstruct.density_objects

        %get the indicies of the first cluster (or a few) in this region
     valstosample =   find(analysisstruct.annotation_vec{conditionhere,end} == kk);
     if numel(valstosample)
        clusterind = randsample(valstosample,1);

        %get the snippet inds
       snippetvals = find( analysisstruct.agg_snippetinds{conditionhere}==clusterind);

        %get the mocap values1
        fieldnames_mocap = fieldnames(analysisstruct.agg_mocap_structs_snippets{conditionhere});
        for mm = 1:numel(fieldnames_mocap)
        snippetstruct{kk}.aligned_mocap.(fieldnames_mocap{mm}) = analysisstruct.agg_mocap_structs_snippets{conditionhere}.(fieldnames_mocap{mm})(snippetvals,:);
            snippetstruct{kk}.agg_preproc.(fieldnames_mocap{mm}) =analysisstruct.agg_preproc{conditionhere}.(fieldnames_mocap{mm})(snippetvals,:);
        end
     end
    end
   %save some metadata in case its useful
    snippetstruct{1}.params = params;
    snippetstruct{1}.mocapstruct_reduced_agg = analysisstruct.mocapstruct_reduced_agg{conditionhere};

    save(strcat(savedirectory,filesep,filename),'snippetstruct');

end

function [agg_mocap_structs,agg_snippetinds,agg_mocap_preproc] = collect_mocap_snippets_allzvals(analysisstruct)

    %num_snippets = 60;
    snippet_size = 300;
    snippet_res = 5;
    snippet_frac = 1;
    %loop over each cluster, collect inds and files for each animal
    cluster_inds = cell(numel(analysisstruct.conditionnames),1);
    cluster_files = cell(numel(analysisstruct.conditionnames),1);
    cluster_filenumbers = cell(numel(analysisstruct.conditionnames),1);
    presence_matrix = zeros(numel(analysisstruct.conditionnames),1);

    agg_mocap_structs = cell(numel(analysisstruct.conditionnames),1);
    agg_snippetinds = cell(numel(analysisstruct.conditionnames),1);
    agg_mocap_preproc = cell(numel(analysisstruct.conditionnames),1);

        for kk = 1%:numel(analysisstruct.conditionnames)
            relative_inds = find(analysisstruct.condition_inds == kk);
            if numel(relative_inds)
         %       presence_matrix(kk,ll) = 1;
                absolute_inds = analysisstruct.frames_with_good_tracking{kk}(relative_inds);
                %num_exs = min(num_snippets,numel(absolute_inds));
                absolute_inds_restriced =absolute_inds;% sort(randsample(absolute_inds,num_exs),'ASCEND'); %absolute_inds(1:num_exs);
                filelengths=     cat(1,0,cumsum(analysisstruct.filesizes{kk}{1}));
                %find the appropriate file
                absolute_files_restriced = (absolute_inds_restriced-filelengths')';
                filenumbers = zeros(1,numel(absolute_inds_restriced));
                for jj =1:size(absolute_files_restriced,2)
                    filenumbers(jj) = find(absolute_files_restriced(:,jj)>0,1,'last');
                end
                %get indicies modulo the file
                cluster_filenumbers{kk} = (filenumbers);
                cluster_files{kk} = analysisstruct.mocapnames{kk}(filenumbers);
                cluster_inds{kk} =      absolute_files_restriced(sub2ind(size(absolute_files_restriced),filenumbers,1:numel(filenumbers)));
            end
        end


    snippet_sum = -snippet_size:snippet_res:snippet_size;
    % loop over all
    tic
    for kk = 1%:numel(analysisstruct.conditionnames)
        unique_files = unique(cat(2,cluster_filenumbers{kk,:}));
        for jj = unique_files
            aa = (analysisstruct.mocapnames{kk}{jj});
            aligned_markers =   aa.markers_aligned_preproc;
             markers_preproc =   aa.markers_preproc;

            if ~exist('markernames','var')
                markernames = fieldnames(aligned_markers);
            end
                    fprintf(' for condition %f for unique file %f of %f \n',kk,jj,numel(unique_files));


                    %get clusters with the same filenames
                    clustershere =  find(     cluster_filenumbers{kk} == jj);
                    indshere = bsxfun(@plus,cluster_inds{kk}(clustershere),snippet_sum');
                    indshere(indshere<1) = 1;
                    indshere(indshere>analysisstruct.filesizes{kk}{1}(jj)) = analysisstruct.filesizes{kk}{1}(jj);

                    %mark the boundaries between individual examples
                    if numel( agg_snippetinds{kk})
                        uniquenum_here = bsxfun(@plus,1:numel(clustershere),max(agg_snippetinds{kk}));
                        agg_snippetinds{kk} = cat(1,agg_snippetinds{kk}, reshape(repmat(uniquenum_here,numel(snippet_sum),1),[],1));
                    else
                        uniquenum_here = 1:numel(clustershere);
                        agg_snippetinds{kk} = reshape(repmat(uniquenum_here,numel(snippet_sum),1),[],1);
                    end

                    if isempty(agg_mocap_structs{kk})
                        for nn =1:numel(markernames)
                            agg_mocap_structs{kk}.(markernames{nn}) = ...
                                aligned_markers.(markernames{nn})(indshere,:);
                               agg_mocap_preproc{kk}.(markernames{nn}) = ...
                                markers_preproc.(markernames{nn})(indshere,:);
                        end
                    else
                        for nn =1:numel(markernames)
                            agg_mocap_structs{kk}.(markernames{nn}) = cat(1, agg_mocap_structs{kk}.(markernames{nn}),...
                                aligned_markers.(markernames{nn})(indshere,:));
                                       agg_mocap_preproc{kk}.(markernames{nn}) = cat(1, agg_mocap_preproc{kk}.(markernames{nn}),...
                                markers_preproc.(markernames{nn})(indshere,:));
                        end
                    end


        end
    end
end