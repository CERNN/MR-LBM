clc
clear

global NX NY NZ

% LBM domain size
N = 64;
NX = N;
NY = N;
NZ = N;

duct_radius = 30;  % Radius of the circular duct

% Initialize index
index = 0;

%% Generate circular duct walls
for y = 1:NY
    for x = 1:NX
        for z = 1:NZ
            % Compute distance from center
            dist = sqrt((x - (NX-1)/2)^2 + (y - (NY-1)/2)^2);
            
            % Only include voxels outside the circular duct
            if dist > duct_radius
                index = index + 1;
                xyz(index,1) = x;
                xyz(index,2) = y;
                xyz(index,3) = z;
            end
        end
    end
end

% Save voxels in a CSV file
fID = fopen('circular_duct.csv', 'w');
for i = 1:length(xyz)-1
    fprintf(fID, '%d,%d,%d\n', xyz(i,:));
end
fprintf(fID, '%d,%d,%d', xyz(end,:));
fclose(fID);
