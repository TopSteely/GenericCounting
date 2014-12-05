% Compile anisotropic gaussian filter
if(~exist('SelectiveSearchCodeIJCV/anigauss'))
    mex SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss_mex.c SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss.c -output SelectiveSearchCodeIJCV/anigauss
end

if(~exist('SelectiveSearchCodeIJCV/mexCountWordsIndex'))
    mex SelectiveSearchCodeIJCV/Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('SelectiveSearchCodeIJCV/mexFelzenSegmentIndex'))
    mex SelectiveSearchCodeIJCV/Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output SelectiveSearchCodeIJCV/mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.8;
VOCinit;
VOCopts.annopath = '/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/Annotations/%s.xml';
images = '/home/t/Schreibtisch/Thesis/VOCdevkit1/VOC2007/JPEGImages/%s.jpg';
gts = '/home/t/Schreibtisch/Thesis/Rois/GroundTruth/%s.txt';
random_negs = '/home/t/Schreibtisch/Thesis/Rois/RandomNegs/%sn.txt';
tic
for i = 1:9963
    if exist(sprintf(images, num2str(i,'%06d' )), 'file')
        im = imread(sprintf(images, num2str(i,'%06d' )));
    else
        continue
    end
    i
    neg = 0;
    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % get ground truth bounding boxes from annotations
    rec = PASreadrecord(sprintf(VOCopts.annopath,num2str(i,'%06d' )));
    objects = rec.objects;
    clear gb;
    for obj = 1:size(objects, 2)
%            cl = get_class_index(objects(obj).class);
        % only taking objects with same class into account?
        g = objects(obj).bbox; %bounding box of object
        ground_truth = [g(1) g(2) g(3)-g(1) g(4)-g(2)];
        if exist('gb')
            gb = [gb;g];
        else
            gb = g;
        end
    end
    csvwrite(sprintf(gts, num2str(i,'%06d' )), gb)
    tries = 1;
    clear pb
    for negs = 1:size(objects, 2)
        while neg == 0 && tries < 6
            o = 0;
            idx = randperm(length(boxes));
            xperm = boxes(idx(1), :);
            proposal = [xperm(2) xperm(1) xperm(3)-xperm(1) xperm(4)-xperm(2)];
            for obj = 1:size(objects, 2)
                g = objects(obj).bbox; %bounding box of object
                ground_truth = [g(1) g(2) g(3)-g(1) g(4)-g(2)];
                [int, p_overlap] = overlap(im, ground_truth, proposal, 'none');
                if p_overlap > 0
                    o = 1;
                    continue;
                end
            end
            if o == 0
                neg = 1;
                if exist('pb')
                    pb = [pb;proposal];
                else
                    pb = proposal;
                end
            end
            tries = tries + 1;
        end
        neg = 0;
    end
    if exist('pb')
        csvwrite(sprintf(random_negs, num2str(i,'%06d' )), pb)
    end
end
toc

%% plot statistics
%resh = reshape(overlaps, 20, 1000, 9963);
%[persons, birds, cats, cows, dogs, horses, sheep, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps);
