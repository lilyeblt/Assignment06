% TJ Lilyeblade
% 10/18/2017
% Assignment06

% This code should train a neural network to recognise
% written six digit numbers

clear all; close all; clc;

% Training image
Igray = imread('ann/training.jpg');

BW = ~imbinarize(Igray); 

SE = strel('disk',2);
BW2 = imdilate(BW, SE);

labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 1000 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);


% imshow(Igray,'border','tight');
% 
% hold on;
% for k = 1:num
%     rectangle('position',Ibox(:,k),'edgecolor','g','LineWidth',1);
%     
%     col = Ibox(1,k);
%     row = Ibox(2,k);
%     
%     text(col,row-50,sprintf('%2.2d',k), ...
%         'fontsize',16,'color','r','fontweight','bold');
% end
% 
% 
% 
% break;

for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    subImage = BW2(row1:row2, col1:col2);
    
    subImageScaled = imresize(subImage, [24 12]);
    
    TPattern(k,:) = subImageScaled(:)';
end



%TTarget = eye(10)';


TTarget = zeros(100,10);

for row = 1:10
    for col = 1:10
        TTarget( 10*(row-1) + col, row ) = 1;
    end
end




TPattern = TPattern';
TTarget = TTarget';


mynet = newff([zeros(288,1) ones(288,1)], [24 17 10], {'logsig' 'logsig' 'logsig'}, 'traingdx');
mynet.trainParam.epochs = 500;
mynet = train(mynet,TPattern,TTarget);


% mynet = feedforwardnet(24,'traingdx');
% mynet.trainParam.epochs = 500;
% mynet = train(mynet,TPattern,TTarget)

% Unknown image
Igray = imread('ann/480000.jpg');

BW = ~im2bw(Igray);

SE = strel('disk',2);
BW2 = imdilate(BW, SE); 

labels = bwlabel(BW2);
Iprops = regionprops(labels);

Iprops( [Iprops.Area] < 1000 ) = [];
num = length( Iprops );

Ibox = floor( [Iprops.BoundingBox] );
Ibox = reshape(Ibox,[4 num]);


for k = 1:num
    col1 = Ibox(1,k);
    col2 = Ibox(1,k) + Ibox(3,k);
    row1 = Ibox(2,k);
    row2 = Ibox(2,k) + Ibox(4,k);
    
    subImage = BW2(row1:row2, col1:col2);
    subImageScaled = imresize(subImage, [24 12]);
    UPattern(k,:) = subImageScaled(:)';
end

UPattern = UPattern';

Y = sim(mynet,UPattern)