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
labels_p = '/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/%s_partial.txt';
labels_c = '/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/%s_complete.txt';
labels_t = '/home/t/Schreibtisch/Thesis/SS_Boxes/Labels/%s_threshold.txt';
boxes_ = '/home/t/Schreibtisch/Thesis/SS_Boxes/%s.txt';

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
    [boxes, ~ ,~ ,~] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % get ground truth bounding boxes from annotations
    rec = PASreadrecord(sprintf(VOCopts.annopath,num2str(i,'%06d' )));
    objects = rec.objects;
    for b = 1:size(boxes, 1)
        lbl_p = 0;
        lbl_t = 0;
        lbl_c = 0;
%         figure;
%         imshow(im)
        proposal = [boxes(b, 2) boxes(b, 1) boxes(b, 4)-boxes(b, 2) boxes(b, 3)-boxes(b, 1)];
%         rectangle('Position', proposal, 'edgecolor', 'b', 'Linewidth', 3)
        for obj = 1:size(objects, 2)
            cl = get_class_index(objects(obj).class);
            % only taking objects with same class into account?
            g = objects(obj).bbox; %bounding box of object
            ground_truth = [g(1) g(2) g(3)-g(1) g(4)-g(2)];
            %rectangle('Position', ground_truth, 'edgecolor', 'g', 'Linewidth', 3)
            [~, p_overlap] = overlap(ground_truth, proposal, 'rounded');
            lbl_p = lbl_p + p_overlap;
            [int, p_overlap] = overlap(ground_truth, proposal, 'complete');
            %rectangle('Position', int, 'edgecolor', 'r', 'Linewidth', 1.5)
            lbl_c = lbl_c + p_overlap;
            [~, p_overlap] = overlap(ground_truth, proposal, 'threshold');
            lbl_t = lbl_t + p_overlap;
        end
        %title(sprintf('Threshold objects(green boxes) in blue proposal box: %s', num2str(lbl_t) ))
        % append all bounding boxes for image
        % -1 because of matlab->python
        if b == 1
            pb = [proposal(1) proposal(2) proposal(3)+proposal(1) proposal(4)+proposal(2)]-1;
        else
            pb = [pb;[proposal(1) proposal(2) proposal(3)+proposal(1) proposal(4)+proposal(2)]-1];
        end
        if b == 1
            lbls_p = lbl_p;                                                                                                                
            lbls_c = lbl_c;
            lbls_t = lbl_t;
        else
            lbls_p = [lbls_p;lbl_p];
            lbls_c = [lbls_c;lbl_c];
            lbls_t = [lbls_t;lbl_t];
        end
    
    end 
    % output bounding boxes
    csvwrite(sprintf(boxes_, num2str(i,'%06d' )), pb)
    % output labels
    csvwrite(sprintf(labels_p, num2str(i,'%06d' )), lbls_p)
    csvwrite(sprintf(labels_c, num2str(i,'%06d' )), lbls_c)
    csvwrite(sprintf(labels_t, num2str(i,'%06d' )), lbls_t)
end
toc

%% plot statistics
%resh = reshape(overlaps, 20, 1000, 9963);
%[persons, birds, cats, cows, dogs, horses, sheep, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps);
