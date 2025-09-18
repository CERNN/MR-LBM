global NX NY NZ

%LBM domain size
N = 32;
NX = N;
NY = N;
NZ = N;


%% parallel plates
        index = 0;
y = 0;
for x=0:NX-1
    for z=0:NZ-1
        index = index +1;
        xyz(index,1) = x;
        xyz(index,2) = y;
        xyz(index,3) = z;
    end
end
y = N-1;
for x=0:NX-1
    for z=0:NZ-1
        index = index +1;
        xyz(index,1) = x;
        xyz(index,2) = y;
        xyz(index,3) = z;
    end
end

% save voxels in voxels.xyz file
fID = fopen('plates.csv' , 'w');
%fprintf(fID,'%d\n',length(xyz));  %to add number of atoms to first line
%fprintf(fID,'\n');      %to leave one blank line
for i=1:length(xyz)-1 
    fprintf(fID,'%d,%d,%d\n' ,  xyz(i,:));
end
fprintf(fID,'%d,%d,%d' ,  xyz(end,:));
fclose(fID);


