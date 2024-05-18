clc
clear
close all

N = 64

gridX = N; 
gridY = N;  
gridZ = N;  
stlFile = 'sphere.stl'

[gridOUTPUT] = VOXELISE(gridX,gridY,gridZ,stlFile);

index = 0;
for i = 1:gridX
    for j = 1:gridY
        for k = 1:gridZ
            %%index = i + gridX*(j + gridY*(k));
            if gridOUTPUT(i,j,k) == true
                index = index +1;
                xyz(index,1) = i;
                xyz(index,2) = j;
                xyz(index,3) = k;
            end
        end
    end
end

%scatter3(xyz(:,1),xyz(:,2),xyz(:,3),'kx')
% save voxels in voxels.xyz file

fID = fopen('stl.csv' , 'w');
%fprintf(fID,'%d\n',length(xyz));  %to add number of atoms to first line
%fprintf(fID,'\n');      %to leave one blank line
for i=1:length(xyz)-1 
    fprintf(fID,'%d,%d,%d\n' ,  xyz(i,:));
end
fprintf(fID,'%d,%d,%d' ,  xyz(end,:));
fclose(fID);