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

overlaps = ones(9963, 1000, 20) * -1;
all_objects = ones(9963, 1000) * -1;
max_objects = 0;
max_objects_i = 0;
tic
index = 1;
for i = 1:9963
    if exist(sprintf(images, num2str(i,'%06d' )), 'file')
        im = imread(sprintf(images, num2str(i,'%06d' )));
    else
        continue
    end
    i
    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % get ground truth bounding boxes from annotations
    rec = PASreadrecord(sprintf(VOCopts.annopath,num2str(i,'%06d' )));
    objects = rec.objects;
    if size(objects, 2) > max_objects
        max_objects = size(objects, 2);
        max_objects_i = i;
    end
    for b = 1:size(boxes, 1)
        for obj = 1:size(objects, 2)
            cl = get_class_index(objects(obj).class);
            % only taking objects with same class into account?
            g = objects(obj).bbox; %bounding box of object
            ground_truth = [g(1) g(2) g(3)-g(1) g(4)-g(2)];
            proposal = [boxes(b, 1) boxes(b, 2) boxes(b, 3)-boxes(b, 1) boxes(b, 4)-boxes(b, 2)];
            [int, p_overlap] = overlap(im, ground_truth, proposal, 'rounded');
            if overlaps(i, b, cl) == -1
                overlaps(i, b, cl) = p_overlap;
            else
                overlaps(i, b, cl) = overlaps(i, b, cl) + p_overlap;
            end
            if all_objects(i, b) == -1
                all_objects(i, b) = p_overlap;
            else
                all_objects(i, b) = all_objects(i, b) + p_overlap;
            end
        end
        objects_vs_windowsize(index, :) = [all_objects(i, b) (proposal(3) * proposal(4)) / (size(im,1) * size(im,2))];
        index = index + 1;
    end
end
toc
save statistics.mat overlaps all_objects objects_vs_windowsize

%% plot statistics
%resh = reshape(overlaps, 20, 1000, 9963);
%[persons, birds, cats, cows, dogs, horses, sheep, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps);
