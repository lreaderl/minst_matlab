% https://stackoverflow.com/questions/39580926/how-do-i-load-in-the-mnist-digits-and-label-data-in-matlab
% https://github.com/jervisfm/Digit-Recognizer
% https://github.com/AlexScheitlin/matlab_minst

% Load images and labels of the train set.
% [images, labels] = load_minst_database('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', -1);

% Load images and labels of the test set.
% [images, labels] = load_minst_database('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', -1);

% Access the kth digit (image):
% digit = images(:,:,k);

% Access the kth label
% lbl = labels(k);

% Show kth image:
% imshow(uint8(images(:,:,k)))
% image(images(:,:,k))
function [images, labels] = load_minst_database(path_to_images, path_to_labels, show_log)
  %show_log: 1 or 0 to specify whether intermediate steps should be printed to the console or not.

  % Open the file with images.
  fid_images = fopen(path_to_images, 'r');

  % Open the file with labels.
  fid_labels = fopen(path_to_labels, 'r');

  % Read the magic numbers of both files.
  A = fread(fid_images, 1, 'uint32');
  magic_number_images = swapbytes(uint32(A)); % Should be 2051
  if (show_log == 1)
    fprintf('Magic Number - Images: %d\n', magic_number_images);
  end

  A = fread(fid_labels, 1, 'uint32');
  magic_number_labels = swapbytes(uint32(A)); % Should be 2049
  if (show_log == 1)
    fprintf('Magic Number - Labels: %d\n', magic_number_labels);
  end

  % Read the total number of images.
  % Ensure that this number matches with the total numbers of labels.
  A = fread(fid_images, 1, 'uint32');
  total_images = swapbytes(uint32(A));

  A = fread(fid_labels, 1, 'uint32');
  if total_images ~= swapbytes(uint32(A))
      error('Total number of images read from images and labels files are not the same.');
  end
  if (show_log == 1)
    fprintf('Total number of images: %d\n', total_images);
  end

  % Read the number of rows.
  A = fread(fid_images, 1, 'uint32');
  num_rows = swapbytes(uint32(A));

  % Read the number of columns.
  A = fread(fid_images, 1, 'uint32');
  num_cols = swapbytes(uint32(A));

  if (show_log == 1)
    fprintf('Dimensions of each digit: %d x %d\n\n', num_rows, num_cols);
  end

  % Store each image individually.
  images = zeros(num_rows, num_cols, total_images, 'uint8');
  for k = 1 : total_images
      % Read num_rows*num_cols pixels.
      A = fread(fid_images, num_rows*num_cols, 'uint8');

      % Reshape so that it becomes a matrix.
      % Read in column major format -> transpose at the end.
      images(:,:,k) = reshape(uint8(A), num_cols, num_rows).';
  end

  % Read the labels.
  labels = fread(fid_labels, total_images, 'uint8');

  % Close the files.
  fclose(fid_images);
  fclose(fid_labels);
end