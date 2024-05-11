% Load cover image and watermark logo
cover_image = imread('liftingbody.png');
watermark_logo = imread('testpat1.png');

% Display cover image and watermark logo
figure
subplot(1,2,1);
imshow(cover_image);
title('Cover image: 512 x 512');
subplot(1,2,2);
imshow(watermark_logo);
title('Watermark image: 256 x 256');

% Set alpha value for watermarking
alpha = 0.05;

% Define different attacks
attacks = {'No Attack'; 'Gaussian low-pass filter'; 'Median'; 'Gaussian noise';...
    'Salt and pepper noise'; 'Speckle noise'; 'JPEG compression';...
    'JPEG2000 compression'; 'Sharpening attack'; 'Histogram equalization';...
    'Average filter'; 'Motion blur'};
params = [0; 3; 3; 0.001; 0; 0; 50; 12; 0.8; 0; 0; 0];

% Create figure to display attacked watermarked images
figure
for j = 1:length(attacks)
    attack = string(attacks(j));
    param = params(j);
    % Apply watermarking and get PSNR and SSIM
    [watermarked_image, ~] = dwt_hd_svd(cover_image, watermark_logo, alpha, attack, param);
    PSNR = psnr(watermarked_image, cover_image);
    SSIM = ssim(watermarked_image, cover_image);
    % Display watermarked image
    subplot(3,4,j);
    imshow(watermarked_image);
    xlabel(['PSNR=' + string(PSNR); 'SSIM=' + string(SSIM)]);
    title(attack);
end
sgtitle(['DWT-HD-SVD: Attacked watermarked image; Size = ' + string(length(watermark_logo)) + 'x' + string(length(watermark_logo)) + '; \alpha = ' + string(alpha)]);

% Create figure to display extracted watermarks
figure
for j = 1:length(attacks)
    attack = string(attacks(j));
    param = params(j);
    [~, extracted_watermark] = dwt_hd_svd(cover_image, watermark_logo, alpha, attack, param);
    NC = nc(watermark_logo, extracted_watermark);
    % Display extracted watermark
    subplot(3,4,j);
    imshow(extracted_watermark);
    xlabel([attack + 'NC=' + string(NC)]);
end
sgtitle(['DWT-HD-SVD: Extracted watermarks image from the attacked watermarked images; Size = ' + string(length(watermark_logo)) + 'x' + string(length(watermark_logo)) + '; \alpha = ' + string(alpha)]);

% DWT-HD-SVD function
function [watermarked_image, extracted_watermark] = dwt_hd_svd(cover_image, watermark_logo, alpha, attack, param)
% Determine the level of wavelet decomposition based on the image sizes
M = length(cover_image);
N = length(watermark_logo);
R = log2(M/N);

% Apply wavelet decomposition
if R == 1
    [LL, HL, LH, HH] = dwt2(cover_image, 'haar');
    [P, H] = hess(LL);
elseif R == 2
    [LL, HL, LH, HH] = dwt2(cover_image, 'haar');
    [LL2, HL2, LH2, HH2] = dwt2(LL, 'haar');
    [P, H] = hess(LL2);
elseif R == 3
    [LL, HL, LH, HH] = dwt2(cover_image, 'haar');
    [LL2, HL2, LH2, HH2] = dwt2(LL, 'haar');
    [LL3, HL3, LH3, HH3] = dwt2(LL2, 'haar');
    [P, H] = hess(LL3);
end

% Apply singular value decomposition for watermark embedding
[HUw, HSw, HVw] = svd(H, 'econ');
[Uw, Sw, Vw] = svd(double(watermark_logo), 'econ');
HSw_hat = HSw + alpha .* Sw;
H_hat = HUw * HSw_hat * HVw';
LL_hat = P * H_hat * P';

% Reconstruct watermarked image
if R == 1
    watermarked_image = idwt2(LL_hat, HL, LH, HH, 'haar');
elseif R == 2
    LL_hat = idwt2(LL_hat, HL2, LH2, HH2, 'haar');
    watermarked_image = idwt2(LL_hat, HL, LH, HH, 'haar');
elseif R == 3
    LL_hat2 = idwt2(LL_hat, HL3, LH3, HH3, 'haar');
    LL_hat = idwt2(LL_hat2, HL2, LH2, HH2, 'haar');
    watermarked_image = idwt2(LL_hat, HL, LH, HH, 'haar');
end

% Convert watermarked image to uint8
watermarked_image = uint8(watermarked_image);

% Apply selected attack
watermarked_image = Attacks(watermarked_image, attack, param);

% Apply wavelet decomposition for watermark extraction
if R == 1
    [LLw, ~, ~, ~] = dwt2(watermarked_image, 'haar');
    Hw = hess(LLw);
elseif R == 2
    [LLw, ~, ~, ~] = dwt2(watermarked_image, 'haar');
    [LLw2, ~, ~, ~] = dwt2(LLw, 'haar');
    Hw = hess(LLw2);
elseif R == 3
    [LLw, ~, ~, ~] = dwt2(watermarked_image, 'haar');
    [LLw2, ~, ~, ~] = dwt2(LLw, 'haar');
    [LLw3, ~, ~, ~] = dwt2(LLw2, 'haar');
    Hw = hess(LLw3);
end

% Apply singular value decomposition for watermark extraction
[HUw_hat, HSbw_hat, HVw_hat] = svd(Hw);
Sw_hat = (HSbw_hat - HSw) ./ alpha;
w_hat = Uw * Sw_hat * Vw';
extracted_watermark = uint8(w_hat);
end

% Attack function
function [watermarked_image] = Attacks(watermarked_image, attack, param)
switch attack
    case 'No Attack'
    case 'Median'
        watermarked_image = medfilt2(watermarked_image, [param param]);     
    case 'Gaussian noise'
        watermarked_image = imnoise(watermarked_image, 'gaussian', 0, param);
    case 'Salt and pepper noise'
        watermarked_image = imnoise(watermarked_image, 'salt & pepper', 0.001);
    case 'Speckle noise'
        watermarked_image = imnoise(watermarked_image, 'speckle', 0.001);
    case 'Sharpening attack'
        watermarked_image = imsharpen(watermarked_image, 'Amount', param);
    case 'Rotating attack'
        watermarked_image = imrotate(watermarked_image, 2, 'crop');
    case 'Motion blur'
        watermarked_image = imfilter(watermarked_image, fspecial('motion', 7, 4), 'replicate');
    case 'Average filter'
        watermarked_image = imfilter(watermarked_image, fspecial('average', [3 3]), 'replicate');
    case 'JPEG2000 compression'
        imwrite(watermarked_image, 'jpeg2000ImageAttacked.j2k', 'jp2', 'CompressionRatio', param);
        watermarked_image = imread('jpeg2000ImageAttacked.j2k');
    case 'JPEG compression'
        imwrite(watermarked_image, 'jpegImageAttacked.jpg', 'jpg', 'quality', param);
        watermarked_image = imread('jpegImageAttacked.jpg');
    case 'Gaussian low-pass filter'
        watermarked_image = imfilter(watermarked_image, fspecial('gaussian', [3 3], param), 'replicate');
    case 'Histogram equalization'
        watermarked_image = histeq(watermarked_image);
    case 'Rescaling (0.25)'
        watermarked_image = imresize(watermarked_image, 0.25);
    case 'Rescaling (4)'
        watermarked_image = imresize(watermarked_image, 4);
    case 'Crop attack'
        watermarked_image = imcrop(watermarked_image);
    otherwise
        errordlg('Please specify attack!');
end
end

% Normalized Correlation (NC) function
function [NC] = nc(watermark_logo, extracted_watermark)
w = double(watermark_logo);
w_hat = double(extracted_watermark);
N = length(w);
A = 0; B = 0; C = 0;
for i = 1:N
    for j = 1:N
        A = A + w(i,j) * w_hat(i,j);
        B = B + w(i,j) * w(i,j);
        C = C + w_hat(i,j) * w_hat(i,j);
    end
end
B = sqrt(B); C = sqrt(C);
NC = A / (B * C);
end
