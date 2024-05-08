from numpy import asarray, resize, percentile, stack, uint8, vectorize
from numpy.lib.stride_tricks import as_strided
import skimage.filters
from PIL import Image as im
import matplotlib.pyplot as plt

Q_DICTIONARY = {
    '0':2,
    '1':4.5,
    '2':8,
    '3':13,
    '4':20,
    '5':30,
}
Q_DICTIONARY_LAST = len(Q_DICTIONARY)-1

def computeMaskedImage(image, heatmap, p_index):
    
    def computeBlur(val):
        if val == 2:
            return float(3.0)
        if val == 4:
            return float(6.0)
        if val == 8:
            return float(12.0)
        if val == 16:
            return float(16.0)
        if val == 32:
            return float(24.0)
        return float(3.0)

    def tile_array(a, b0, b1):
        r, c = a.shape                                    # number of rows/columns
        rs, cs = a.strides                                # row/column strides 
        x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
        return x.reshape(r*b0, c*b1)                      # create new 2D array
    
    arr_image = asarray(image)
    blur_factor = computeBlur(int(len(arr_image)/len(asarray(heatmap))))
    heatmap = tile_array(asarray(heatmap), int(len(arr_image)/len(heatmap)), int(len(arr_image[0])/len(heatmap[0])))
    
    total_len = len(arr_image)*len(arr_image[0])
    ordered_map = resize(heatmap, total_len)

    q = 100 - Q_DICTIONARY[str(p_index)]
    
    p = percentile(ordered_map, q=q, axis=0)

    def f(el, p):
        return 1 if el >= p else 0
    mask_function = vectorize(f, otypes=['float'])
    mask_bool = mask_function(heatmap, p)

    mask = skimage.filters.gaussian(mask_bool, sigma=(blur_factor, blur_factor), truncate=1.5, channel_axis=-1)

    mask3d = stack((mask, mask, mask), axis=2)
    modified_image = arr_image*mask3d

    return im.fromarray(modified_image.astype(uint8)).convert('RGB'), mask3d, Q_DICTIONARY[str(p_index)], mask_bool

def computeMaskedImageWithNumberOfPixels(image, heatmap, pixel_count):
    
    def computeBlur(val):
        if val == 2:
            return float(3.0)
        if val == 4:
            return float(6.0)
        if val == 8:
            return float(12.0)
        if val == 16:
            return float(16.0)
        if val == 32:
            return float(24.0)
        return float(3.0)

    def tile_array(a, b0, b1):
        r, c = a.shape                                    # number of rows/columns
        rs, cs = a.strides                                # row/column strides 
        x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
        return x.reshape(r*b0, c*b1)                      # create new 2D array
    
    arr_image = asarray(image)
    blur_factor = computeBlur(int(len(arr_image)/len(asarray(heatmap))))
    heatmap = tile_array(asarray(heatmap), int(len(arr_image)/len(heatmap)), int(len(arr_image[0])/len(heatmap[0])))
    
    total_len = len(arr_image)*len(arr_image[0])
    ordered_map = resize(heatmap, total_len)

    p_index = 100*float(pixel_count)/total_len

    q = 100 - p_index
    
    p = percentile(ordered_map, q=q, axis=0)

    def f(el, p):
        return 1 if el >= p else 0
    mask_function = vectorize(f, otypes=['float'])
    mask_bool = mask_function(heatmap, p)

    mask = skimage.filters.gaussian(mask_bool, sigma=(blur_factor, blur_factor), truncate=1.5, channel_axis=-1)

    mask3d = stack((mask, mask, mask), axis=2)
    modified_image = arr_image*mask3d

    return im.fromarray(modified_image.astype(uint8)).convert('RGB'), mask3d, p_index, mask_bool