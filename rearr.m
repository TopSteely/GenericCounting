function [persons, birds, cats, cows, dogs, horses, sheeps, aeroplanes, bicycles, boats, buses, cars, motorbikes, trains, bottles, chairs, diningtables, pottedplants, sofas, tvmonitors] = rearr(overlaps)
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
    persons(all(persons==0,2),:) = [];
    persons( :, ~any(persons,1) ) = [];
    birds(all(birds==0,2),:) = [];
    birds( :, ~any(birds,1) ) = [];
    cats(all(cats==0,2),:) = [];
    cats( :, ~any(cats,1) ) = [];
    cows(all(cows==0,2),:) = [];
    cows( :, ~any(cows,1) ) = [];
    dogs(all(dogs==0,2),:) = [];
    dogs( :, ~any(dogs,1) ) = [];
    horses(all(horses==0,2),:) = [];
    horses( :, ~any(horses,1) ) = [];
    sheeps(all(sheeps==0,2),:) = [];
    sheeps( :, ~any(sheeps,1) ) = [];
    aeroplanes(all(aeroplanes==0,2),:) = [];
    aeroplanes( :, ~any(aeroplanes,1) ) = [];
    bicycles(all(bicycles==0,2),:) = [];
    bicycles( :, ~any(bicycles,1) ) = [];
    boats(all(boats==0,2),:) = [];
    boats( :, ~any(boats,1) ) = [];
    buses(all(buses==0,2),:) = [];
    buses( :, ~any(buses,1) ) = [];
    cars(all(cars==0,2),:) = [];
    cars( :, ~any(cars,1) ) = [];
    motorbikes(all(motorbikes==0,2),:) = [];
    motorbikes( :, ~any(motorbikes,1) ) = [];
    trains(all(trains==0,2),:) = [];
    trains( :, ~any(trains,1) ) = [];
    bottles(all(bottles==0,2),:) = [];
    bottles( :, ~any(bottles,1) ) = [];
    chairs(all(chairs==0,2),:) = [];
    chairs( :, ~any(chairs,1) ) = [];
    diningtables(all(diningtables==0,2),:) = [];
    diningtables( :, ~any(diningtables,1) ) = [];
    pottedplants(all(pottedplants==0,2),:) = [];
    pottedplants( :, ~any(pottedplants,1) ) = [];
    sofas(all(sofas==0,2),:) = [];
    sofas( :, ~any(sofas,1) ) = [];
    tvmonitors(all(tvmonitors==0,2),:) = [];
    tvmonitors( :, ~any(tvmonitors,1) ) = [];
    
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
    hist(log(numUnique(2:end)))
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