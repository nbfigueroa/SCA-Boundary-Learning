function [ D ] = create_distanceFeatures( R1, R2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

R1_J1 = R1(:,1:3);R1_J2 = R1(:,4:6);R1_J3 = R1(:,7:9);R1_J4 = R1(:,10:12);R1_J5 = R1(:,13:15); R1_J6 = R1(:,16:18);   
R2_J1 = R2(:,1:3);R2_J2 = R2(:,4:6);R2_J3 = R2(:,7:9);R2_J4 = R2(:,10:12);R2_J5 = R2(:,13:15); R2_J6 = R2(:,16:18);

D = zeros(length(R1),36);
for i=1:length(R1)
    
    D(i,:) = [norm(R1_J1(i,:) - R2_J1(i,:)) norm(R1_J1(i,:) - R2_J2(i,:)) norm(R1_J1(i,:) - R2_J3(i,:)) norm(R1_J1(i,:) - R2_J4(i,:)) norm(R1_J1(i,:) - R2_J5(i,:)) norm(R1_J1(i,:) - R2_J5(i,:)) ...
                    norm(R1_J2(i,:) - R2_J1(i,:)) norm(R1_J2(i,:) - R2_J2(i,:)) norm(R1_J2(i,:) - R2_J3(i,:)) norm(R1_J2(i,:) - R2_J4(i,:)) norm(R1_J2(i,:) - R2_J5(i,:)) norm(R1_J2(i,:) - R2_J5(i,:)) ...
                    norm(R1_J3(i,:) - R2_J1(i,:)) norm(R1_J3(i,:) - R2_J2(i,:)) norm(R1_J3(i,:) - R2_J3(i,:)) norm(R1_J3(i,:) - R2_J4(i,:)) norm(R1_J3(i,:) - R2_J5(i,:)) norm(R1_J3(i,:) - R2_J5(i,:)) ...
                    norm(R1_J4(i,:) - R2_J1(i,:)) norm(R1_J4(i,:) - R2_J2(i,:)) norm(R1_J4(i,:) - R2_J3(i,:)) norm(R1_J4(i,:) - R2_J4(i,:)) norm(R1_J4(i,:) - R2_J5(i,:)) norm(R1_J4(i,:) - R2_J5(i,:)) ...
                    norm(R1_J5(i,:) - R2_J1(i,:)) norm(R1_J5(i,:) - R2_J2(i,:)) norm(R1_J5(i,:) - R2_J3(i,:)) norm(R1_J5(i,:) - R2_J4(i,:)) norm(R1_J5(i,:) - R2_J5(i,:)) norm(R1_J5(i,:) - R2_J5(i,:)) ...
                    norm(R1_J3(i,:) - R2_J1(i,:)) norm(R1_J6(i,:) - R2_J2(i,:)) norm(R1_J6(i,:) - R2_J3(i,:)) norm(R1_J6(i,:) - R2_J4(i,:)) norm(R1_J6(i,:) - R2_J5(i,:)) norm(R1_J6(i,:) - R2_J5(i,:))];
end


end

