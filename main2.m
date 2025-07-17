clc;
close all;
close all hidden;
clear;
warning off;
%%
[file,path] = uigetfile('*','select an image');  %Open file selection dialog box
input_img = imread([path,file]);  %Read image from graphics file
input_img = imresize(input_img,[227 227]);  %Resize image
figure;imshow(input_img);title('Input Image');  %Display image

%%
matlabroot = pwd;
datasetpath = fullfile(matlabroot, 'Dataset');
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7,'randomized');

count = imds.countEachLabel;

augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227 3],imdsValidation);



%%

net = squeezenet;

layers = [imageInputLayer([227 227 3])
    
    net(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
    
    fullyConnectedLayer(5) % modifying the fullyconnected layer with respect to classes
    
    softmaxLayer
    
    classificationLayer];

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",30,...
    "MiniBatchSize",25,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Verbose",false,...
    "Plots","training-progress");


[training, traininginfo] = trainNetwork(augimdsTrain,layers,opts); 

YPred = classify(training, input_img);
fprintf('The food item classified by SqueezeNet is %s\n',YPred);
fprintf('The training accuracy by SqueezeNet is %0.4f\n', mean(traininginfo.TrainingAccuracy));

%%

augimdsTrain1 = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation1 = augmentedImageDatastore([224 224 3],imdsValidation);

net1 = vgg16;

layers1 = [imageInputLayer([224 224 3])
    
    net1(2:end-2) %accessing the pretrained network layers from second layer to end-3 layers
    
    fullyConnectedLayer(5) % modifying the fullyconnected layer with respect to classes
    
    softmaxLayer
    
    classificationLayer];


opts1 = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",45,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Verbose",false,...
    "Plots","training-progress");


[training1, traininginfo1] = trainNetwork(augimdsTrain1,layers1,opts1); 


img = imresize(input_img,[224 224]);

YPred1 = classify(training1, img);
fprintf('The food item classified by VGG16 Net is %s\n',YPred1);
fprintf('The training accuracy by VGG16 Net is %0.4f\n', mean(traininginfo1.TrainingAccuracy));

