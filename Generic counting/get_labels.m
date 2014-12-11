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
labels = '/home/t/Schreibtisch/Thesis/Labels/%s_thresh.txt';
tic
for i = 1:9963
    i
    % get ground truth bounding boxes from annotations
    rec = PASreadrecord(sprintf(VOCopts.annopath,num2str(i,'%06d' )));
    objects = rec.objects;
    for obj = 1:size(objects, 2)
        g = objects(obj).bbox; %bounding box of object
        main_box = [g(1) g(2) g(3)-g(1) g(4)-g(2)];
        lbl = 0;
        for ob = 1:size(objects, 2)
            other = objects(ob).bbox;
            other_box = [other(1) other(2) other(3)-other(1) other(4)-other(2)];
            [int, p_overlap] = overlap(other_box, main_box, 'threshold');
            lbl = lbl + p_overlap;
        end
        if obj == 1
            lbls = lbl;
        else
            lbls = [lbls;lbl];
        end
    end
    csvwrite(sprintf(labels, num2str(i,'%06d' )), lbls)
end
toc

%% plot statistics
%resh = reshape(overlaps, 20, 1000, 9963);
%[persons, birds, cats, cows, dogs, horses, sheep, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps);
