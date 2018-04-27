function num = Test(imgPath, modelPath);
%example:
%Test('1.jpg', 'SVMmodel.mat')
model = loadCompactModel(modelPath);
net = alexnet;
layer = 'fc6';
gpuarrayA = imread(imgPath);
image1 = imresize(gpuarrayA, [227 227], 'method', 'lanczos3');
preprocessedImage(:,:,1) = image1;
preprocessedImage(:,:,2) = image1;
preprocessedImage(:,:,3) = image1;
feature = activations(net,preprocessedImage,layer,'OutputAs','rows');
num = predict(model, feature);
end