% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
end

if(~exist('mexCountWordsIndex'))
    mex Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
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
overlaps = ones(9963, 600, 20) * -1;
tic
for i = 1:9963
    if exist(sprintf(images, num2str(i,'%06d' )), 'file')
        im = imread(sprintf(images, num2str(i,'%06d' )));
    else
        continue
    end
    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % get ground truth bounding boxes from annotations
    rec = PASreadrecord(sprintf(VOCopts.annopath,num2str(i,'%06d' )));
    objects = rec.objects;
    for b = 1:size(boxes, 1)
        for obj = 1:size(objects, 2)
            cl = get_class_index(objects(obj).class);
            ground_truth = objects(obj).bbox; %bounding box of object
            % overlap???
            [int, overlaps(i, b, cl)] = overlap(ground_truth, boxes(b, :));
            % plot
%                 figure;
%                 imshow(im)
%                 hold on
%                 rectangle('Position', ground_truth, 'edgecolor', 'r', 'Linewidth', 2.5)
%                 rectangle('Position', boxes(b, :), 'edgecolor', 'g', 'Linewidth', 2.5)
%                 rectangle('Position', int, 'edgecolor', 'b', 'Linewidth', 2.5)
%                 hold off
        end
    end
end
toc
save statistics.mat overlaps

%% plot statistics
%resh = reshape(overlaps, 20, 645, 9963);
[persons, birds, cats, cows, dogs, horses, sheep, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps);
