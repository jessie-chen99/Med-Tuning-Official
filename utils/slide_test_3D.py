import math
import torch

def slide_window(ori_img, crop_size, model):
    stride_rate = 1.0/3.0
    stride = int(crop_size * stride_rate)  # 85
    batch, classes, origin_d, origin_h, origin_w = ori_img.size()

    with torch.cuda.device_of(ori_img):
        outputs = ori_img.new().resize_(batch, classes, origin_d, origin_h, origin_w).zero_().cuda()
        count_norm = ori_img.new().resize_(batch, 1, origin_d, origin_h, origin_w).zero_().cuda()
    h_grids = int(math.ceil(1.0 * (origin_h - crop_size) / stride)) + 1
    w_grids = int(math.ceil(1.0 * (origin_w - crop_size) / stride)) + 1
    d_grids = int(math.ceil(1.0 * (origin_d - crop_size) / stride)) + 1

    for idh in range(h_grids):  # 3
        for idw in range(w_grids):
            for idd in range(d_grids):
                h0 = idh * stride
                w0 = idw * stride
                d0 = idd * stride
                h1 = min(h0 + crop_size, origin_h)
                w1 = min(w0 + crop_size, origin_w)
                d1 = min(d0 + crop_size, origin_d)

                #adjustment
                if h1 == origin_h:
                    h0 = h1 - crop_size
                if w1 == origin_w:
                    w0 = w1 - crop_size
                if d1 == origin_d:
                    d0 = d1 - crop_size

                crop_img = crop_image(ori_img, d0, d1, h0, h1, w0, w1)
                output = model_inference(model, crop_img)
                outputs[:, :, d0:d1, h0:h1, w0:w1] += crop_image(output, 0, d1 - d0, 0, h1 - h0, 0, w1 - w0)
                count_norm[:, :, d0:d1, h0:h1, w0:w1] += 1
    assert ((count_norm == 0).sum() == 0)
    outputs = outputs / count_norm
    outputs = outputs[:, :, :origin_d, :origin_h, :origin_w]
    return outputs

def model_inference(model, image):
    
    output = model(image)

    return output

def crop_image(img, d0, d1, h0, h1, w0, w1):
    return img[:, :, d0:d1, h0:h1, w0:w1]
