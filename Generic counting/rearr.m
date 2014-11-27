function [persons, birds, cats, cows, dogs, horses, sheeps, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps)
    % create array per class
    persons = overlaps(:,:,1);
    birds = overlaps(:,:,2);
    cats = overlaps(:,:,3);
    cows = overlaps(:,:,4);
    dogs = overlaps(:,:,5);
    horses = overlaps(:,:,6);
    sheeps = overlaps(:,:,7);
    aeroplanes = overlaps(:,:,8);
    bicycles = overlaps(:,:,9);
    boats = overlaps(:,:,10);
    buses = overlaps(:,:,11);
    cars = overlaps(:,:,12);
    motorbikes = overlaps(:,:,13);
    trains = overlaps(:,:,14);
    bottles = overlaps(:,:,15);
    chairs = overlaps(:,:,16);
    diningtables = overlaps(:,:,17);
    pottedplants = overlaps(:,:,18);
    sofas = overlaps(:,:,19);
    tvmonitors = overlaps(:,:,20);
    % delete rows/columns, without entry (-1 was the initialization)
    persons(all(persons==-1,2),:) = [];
    persons( :, all(persons==-1,1) ) = [];
    birds(all(birds==-1,2),:) = [];
    birds( :, all(birds==-1,1) ) = [];
    cats(all(cats==-1,2),:) = [];
    cats( :, all(cats==-1,1) ) = [];
    cows(all(cows==-1,2),:) = [];
    cows( :, all(cows==-1,1) ) = [];
    dogs(all(dogs==-1,2),:) = [];
    dogs( :, all(dogs==-1,1) ) = [];
    horses(all(horses==-1,2),:) = [];
    horses( :, all(horses==-1,1) ) = [];
    sheeps(all(sheeps==-1,2),:) = [];
    sheeps( :, all(sheeps==-1,1) ) = [];
    aeroplanes(all(aeroplanes==-1,2),:) = [];
    aeroplanes( :, all(aeroplanes==-1,1) ) = [];
    bicycles(all(bicycles==-1,2),:) = [];
    bicycles( :, all(bicycles==-1,1) ) = [];
    boats(all(boats==-1,2),:) = [];
    boats( :, all(boats==-1,1) ) = [];
    buses(all(buses==-1,2),:) = [];
    buses( :, all(buses==-1,1) ) = [];
    cars(all(cars==-1,2),:) = [];
    cars( :, all(cars==-1,1) ) = [];
    motorbikes(all(motorbikes==-1,2),:) = [];
    motorbikes( :, all(motorbikes==-1,1) ) = [];
    trains(all(trains==-1,2),:) = [];
    trains( :, all(trains==-1,1) ) = [];
    bottles(all(bottles==-1,2),:) = [];
    bottles( :, all(bottles==-1,1) ) = [];
    chairs(all(chairs==-1,2),:) = [];
    chairs( :, all(chairs==-1,1) ) = [];
    diningtables(all(diningtables==-1,2),:) = [];
    diningtables( :, all(diningtables==-1,1) ) = [];
    pottedplants(all(pottedplants==-1,2),:) = [];
    pottedplants( :, all(pottedplants==-1,1) ) = [];
    sofas(all(sofas==-1,2),:) = [];
    sofas( :, all(sofas==-1,1) ) = [];
    tvmonitors(all(tvmonitors==-1,2),:) = [];
    tvmonitors( :, all(tvmonitors==-1,1) ) = [];
    
    [uniques,numUnique] = count_unique(persons);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - persons')
    [uniques,numUnique] = count_unique(birds);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - birds')
    [uniques,numUnique] = count_unique(cats);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - cats')
    [uniques,numUnique] = count_unique(cows);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - cows')
    [uniques,numUnique] = count_unique(dogs);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - dogs')
    [uniques,numUnique] = count_unique(horses);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - horses')
    [uniques,numUnique] = count_unique(sheeps);
    figure
    nbins = size(uniques)-1;
    hist(log(numUnique(2:end)), nbins)
    title('Histogram of objects per window - sheep')
    [uniques,numUnique] = count_unique(aeroplanes);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - aeroplanes')
    [uniques,numUnique] = count_unique(bicycles);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - bicycles')
    [uniques,numUnique] = count_unique(boats);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - boats')
    [uniques,numUnique] = count_unique(buses);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - buses')
    [uniques,numUnique] = count_unique(cars);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - cars')
    [uniques,numUnique] = count_unique(motorbikes);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - motorbikes')
    [uniques,numUnique] = count_unique(trains);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - trains')
    [uniques,numUnique] = count_unique(bottles);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - bottles')
    [uniques,numUnique] = count_unique(chairs);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - chairs')
    [uniques,numUnique] = count_unique(diningtables);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - diningtables')
    [uniques,numUnique] = count_unique(pottedplants);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - pottedplants')
    [uniques,numUnique] = count_unique(sofas);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - sofas')
    [uniques,numUnique] = count_unique(tvmonitors);
    figure
    hist(log(numUnique(2:end)))
    title('Histogram of objects per window - tvmonitors')
end