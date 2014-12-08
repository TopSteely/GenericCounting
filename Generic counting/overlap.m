function [intersection, p] = overlap(A, B, string)
    left = max(A(1), B(1));
    right = min(A(1)+A(3), B(1)+B(3));
    bottom = max(A(2), B(2));
    top = min(A(2)+A(4), B(2)+B(4));
    intersection = [left bottom right-left top-bottom];
    surface_intersection = intersection(3) * intersection(4);
    surface_ground_truth = A(3)  * A(4);
    % round to 0.1 decimal
    p = surface_intersection / surface_ground_truth;
    switch string
        case 'complete'
            if p > 0
                p = 1;
            end
        case 'threshold'
            if p > 0.5
                p = 1;
            end
        case 'rounded'
            p = round(surface_intersection / surface_ground_truth* 10)/10;
    end
    if any(intersection<0)
        %imshow(im)
        %rectangle('Position', A, 'edgecolor', 'r', 'Linewidth', 3)
        %rectangle('Position', B, 'edgecolor', 'g', 'Linewidth', 3)
        %rectangle('Position', int, 'edgecolor', 'b', 'Linewidth', 1.5)            
        p = 0;
    end
    if p < 0
        p = 0;
    end
    if p>1
        'whaaaaaat??????'
    end
end