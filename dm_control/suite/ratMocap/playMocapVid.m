function playMocapVid(vid, fps, loop)
    if ~exist('loop', 'var')
        loop = false;
    end

    if ~exist('fps', 'var')
        fps = 30;
    end

    fig = figure;
    again = true;
    str = '';
    while again
        for i = 1:size(vid, 1)-1
            imshow(squeeze(vid(i,:,:,:)));
            pause(1/fps)
            str = [str get(fig, 'CurrentCharacter')];
            if ~isempty(str)
                break
            end
        end
        again = loop && isempty(str);
    end
    close(fig)
    
end