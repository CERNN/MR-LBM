global NX NY NZ

%LBM domain size
N = 256;
NX = N;
NY = N;
NZ = N;

L = 32;


%% parallel plates
        index = 0;
for y = 0:L-1;
    for x=0:L-1
        for z=0:L-1
            index = index +1;
            xyz(index,1) = x;
            xyz(index,2) = y;
            xyz(index,3) = z;
        end
    end
end

xyz(:,1) = xyz(:,1) + NX/2 - L/2;
xyz(:,2) = xyz(:,2) + NY/2 - L/2;
xyz(:,3) = xyz(:,3) + NZ/2 - L/2;


% save voxels in voxels.xyz file
fID = fopen('cube.csv' , 'w');
%fprintf(fID,'%d\n',length(xyz));  %to add number of atoms to first line
%fprintf(fID,'\n');      %to leave one blank line
for i=1:length(xyz)-1 
    fprintf(fID,'%d,%d,%d\n' ,  xyz(i,:));
end
fprintf(fID,'%d,%d,%d' ,  xyz(end,:));
fclose(fID);


