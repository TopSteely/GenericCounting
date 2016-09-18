function list = erase_duplicates(list)
    i = 2;
    while i < size(list, 1)
        if abs(list(i, 1) - list(i+1, 1)) < 0.01
            list(i, 2) = list(i, 2) + list(i+1, 2);
            list(i+1, :) = [];
            i = i - 1;
        end
        i = i + 1;
    end
end