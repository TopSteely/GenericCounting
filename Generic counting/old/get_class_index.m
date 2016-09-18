function cl = get_class_index(class)
    switch class
       case 'person'
          cl = 1;
       case 'bird'
          cl = 2;
       case 'cat'
          cl = 3;
       case 'cow'
          cl = 4;
       case 'dog'
          cl = 5;
       case 'horse'
          cl = 6;
       case 'sheep'
          cl = 7;
       case 'aeroplane'
          cl = 8;
       case 'bicycle'
          cl = 9;
       case 'boat'
          cl = 10;
       case 'bus'
          cl = 11;
       case 'car'
          cl = 12;
       case 'motorbike'
          cl = 13;
       case 'train'
          cl = 14;
       case 'bottle'
          cl = 15;
       case 'chair'
          cl = 16;
       case 'diningtable'
          cl = 17;
       case 'pottedplant'
          cl = 18;
       case 'sofa'
          cl = 19;
       case 'tvmonitor'
          cl = 20;
       otherwise
          class
    end
end