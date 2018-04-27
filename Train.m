function result = Train(path_to_train_images, path_to_train_labels, path_to_test_images, path_to_test_labels);
% Run on MATLAB R2018a, with Statistics and Machine Learning Toolbox and
% Computer Vision System Toolbox and Neural Network Toolbox Model for AlexNet Network 
% Model: AlexNet feartures based SVM Classification
% With windows10 64bit Home Edition, Intel core i9-7980XE, Nvidia GTX 1080Ti, about 10 minutes
tic
disp('Start loading');
%examples for input arguments:
%[train_data, train_labels ] = load_minst_database('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', -1);
%[test_data, test_labels] = load_minst_database('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', -1);

[train_data, train_labels ] = load_minst_database(path_to_train_images, path_to_train_labels, -1);
[test_data, test_labels] = load_minst_database(path_to_test_images, path_to_test_labels, -1);
toc
disp('Start preprocessing');
net = alexnet;
layer = 'fc6';
gpuarrayA = train_data(:,:,1);
image1 = imresize(gpuarrayA, [227 227], 'method', 'lanczos3');
preprocessedImage(:,:,1) = image1;
preprocessedImage(:,:,2) = image1;
preprocessedImage(:,:,3) = image1;
featuresTrain = activations(net,preprocessedImage,layer,'OutputAs','rows');
FeatureSize = length(featuresTrain);
trainingFeatures = zeros(60000, FeatureSize, 'single');
testFeatures = zeros(10000, FeatureSize, 'single');
for i = 1:60000
    gpuarrayA = train_data(:,:,i);
    image1 = imresize(gpuarrayA, [227 227], 'method', 'lanczos3');
    preprocessedImage(:,:,1) = image1;
    preprocessedImage(:,:,2) = image1;
    preprocessedImage(:,:,3) = image1;
    trainingFeatures(i, :) = activations(net,preprocessedImage,layer,'OutputAs','rows');
end
toc
for i = 1:10000
    gpuarrayA = test_data(:,:,i);
    image1 = imresize(gpuarrayA, [227 227], 'method', 'lanczos3');
    preprocessedImage(:,:,1) = image1;
    preprocessedImage(:,:,2) = image1;
    preprocessedImage(:,:,3) = image1;
    testFeatures(i, :) = activations(net,preprocessedImage,layer,'OutputAs','rows');
end
toc
disp('Start training')
classifier = fitcecoc(trainingFeatures, train_labels, 'Coding', 'onevsone', 'Learners', 'svm');
toc
disp('Saving model')
saveCompactModel(classifier,'SVMmodel');
toc
disp('Start testing')
predictedLabels = predict(classifier, testFeatures);
errors = find(predictedLabels~=test_labels);
errorrate = length(errors)/length(test_labels);
disp(1-errorrate)
result = 1-errorrate;
toc