global NX NY NZ

%LBM domain size
N = 256;
NX = N;
NY = 64;
NZ = N;

L = 64;


%% parallel plates
        index = 0;
for y = NY/2 - L/2:L-1 + NY/2 - L/2
    for x= NX/2 - L/2:L-1+ NX/2 - L/2
        for z= NZ/2 - L/2:L-1+ NZ/2 - L/2
            index = index +1;
            xyz(index,1) = x ;
            xyz(index,2) = y ;
            xyz(index,3) = z ;
        end
    end
end

y = 0
for x= NX/2 - L/2-1:L-1+ NX/2 - L/2+1
    for z= NZ/2 - L/2-1:L-1+ NZ/2 - L/2+1
        index = index +1;
        xyz(index,1) = x ;
        xyz(index,2) = y ;
        xyz(index,3) = z ;
    end
end


% save voxels in voxels.xyz file
fID = fopen('mounted_cube .csv' , 'w');
%fprintf(fID,'%d\n',length(xyz));  %to add number of atoms to first line
%fprintf(fID,'\n');      %to leave one blank line
for i=1:length(xyz)-1 
    fprintf(fID,'%d,%d,%d\n' ,  xyz(i,:));
end
fprintf(fID,'%d,%d,%d' ,  xyz(end,:));
fclose(fID);


