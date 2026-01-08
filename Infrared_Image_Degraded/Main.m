close all;
clear;
clc;

ir_folder = '.\gt_ir\'; % Infrared Image
de_folder = '.\Depth\'; % Depth Image

image_tui(ir_folder, de_folder);

%% =====================================================
function image_tui(ir_folder, de_folder)

ir_files = dir(fullfile(ir_folder, '*.png'));
de_files = dir(fullfile(de_folder, '*.png'));

save_path = '.\ir\';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

for k = 1:length(ir_files)

    ir = imread(fullfile(ir_folder, ir_files(k).name));
    de = imread(fullfile(de_folder, de_files(k).name));

    ir = im2double(ir);
    de = im2double(de);

    if size(ir,3) > 1
        ir = rgb2gray(ir);
    end
    if size(de,3) > 1
        de = rgb2gray(de);
    end

    F_ir = fft2(ir);
    F_shift = fftshift(F_ir);

    Magnitude = abs(F_shift);
    Phase = angle(F_shift);

    de_smooth = de;

    mask1 = (de == 0);
    de_smooth(mask1) = de_smooth(mask1) + 0.3;

    mask2 = (de > 0.3) & (de <= 0.5);
    de_smooth(mask2) = de_smooth(mask2) + 0.2;

    de_smooth = min(max(de_smooth, 0), 1);

    de_smooth = imgaussfilt(de_smooth, 0.5);

    de = de_smooth;

    Magnitude_mod = Magnitude .* (0.7 + 0.3 * de);

    F_reconstruct_shift = Magnitude_mod .* exp(1j * Phase);
    F_reconstruct = ifftshift(F_reconstruct_shift);

    ir_fft = ifft2(F_reconstruct);
    ir_fft = real(ir_fft);

    alpha = 0.5;
    ir_out = alpha * ir_fft + (1 - alpha) * (ir .* de);

    imwrite(ir_out, fullfile(save_path, ir_files(k).name));

end

end
