from utils import *
from statistics import median

gaussian_filter_size=5
median_filter_size=3

def get_pixel_at(pixel_grid, i, j):
    '''
    Get pixel values at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.

    Returns:
        ndarray: 1D numpy array representing RGB values.
    '''
    return pixel_grid[i][j]

def get_patch_at(pixel_grid, i, j, size):
    '''
    Get an image patch at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.
        size (int): Patch size.

    Returns:
        ndarray: 3D numpy array representing an image patch.
    '''
    h, w, n = pixel_grid.shape
    len_arm = size//2
    effective_grid = pixel_grid[max(0,i-len_arm):i+len_arm+1,max(0,j-len_arm):j+len_arm+1,:]
    h_eff, w_eff, _ = effective_grid.shape

    if (i-len_arm < 0) or (i+len_arm+1 > h) or (j-len_arm < 0) or (j+len_arm+1 > w):
        padded_grid = np.zeros((size,size,n)).astype(np.uint8)
        x = max(0,len_arm-i)
        y = max(0,len_arm-j)
        padded_grid[x:x+h_eff, y:y+w_eff, :] = effective_grid
        return padded_grid
    else: 
        return effective_grid

def get_gaussian_filter_1d(size):

    dev = size//2
    x = np.linspace(-dev,dev,dev*2+1)
    sigma = size/6
    x = np.exp(-x*x/2/sigma**2)
    return x/x.sum()

'''
def apply_gaussian_filter_patch(after_filter, patch, i, j, gaussian_filter_1d):

    h, *_ = patch.shape
    res = []
    for row in range(h):
        res.append(gaussian_filter_1d.dot(patch[row]))
    res = np.array(res,copy=False)
    after_filter[i][j] = gaussian_filter_1d.dot(res)
'''

def apply_gaussian_filter(pixel_grid, size):
    '''
    Apply gaussian filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''
    print("applying gaussian filter")
    gaussian_filter_1d = get_gaussian_filter_1d(size)
    h, w, n = pixel_grid.shape
    after_filter = np.zeros((h,w,n)).astype(np.uint8)
    for i in range(h):
        for j in range(w):
            patch = get_patch_at(pixel_grid, i, j, size)
            # apply_gaussian_filter_patch(after_filter, _patch, i, j, _gaussian_filter_1d)
            # for performance
            res = []
            for row in range(size):
                res.append(gaussian_filter_1d.dot(patch[row]))
            res = np.array(res,copy=False)
            after_filter[i][j] = gaussian_filter_1d.dot(res)

    return after_filter

'''
def apply_median_filter_patch(after_filter, patch, i, j):

    h, w, n = patch.shape
    lin_pch = np.reshape(patch, (h*w,n))
    rgb = list(zip(*lin_pch))
    after_filter[i][j] = np.array(list(map(median, rgb)),copy=False)
'''

def apply_median_filter(pixel_grid, size):
    '''
    Apply median filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''
    print("applying median filter")
    h, w, n = pixel_grid.shape
    after_filter = np.zeros((h,w,n)).astype(np.uint8)
    num_elem_patch=size**2
    for i in range(h):
        for j in range(w):
            patch = get_patch_at(pixel_grid, i, j, size)
            # apply_median_filter_patch(after_filter, _patch, i, j)
            # for performance
            lin_pch = np.reshape(patch, (num_elem_patch,n))
            r, g, b = list(zip(*lin_pch))
            after_filter[i][j] = np.array([median(r), median(g), median(b)],copy=False)
    return after_filter

def build_gaussian_pyramid(pixel_grid, size, levels=5):
    '''
    Build and return a Gaussian pyramid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.
        levels (int): Number of levels.

    Returns:
        list of ndarray: List of 3D numpy arrays representing Gaussian
        pyramid.
    '''
    pyramid = []
    curr_grid=pixel_grid
    for level in range(levels):
        curr_grid = apply_gaussian_filter(curr_grid, size)
        pyramid.append(curr_grid)
        curr_grid = downsample(curr_grid)
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    '''
    Build and return a Laplacian pyramid.

    Args:
        gaussian_pyramid (list of ndarray): Gaussian pyramid. 

    Returns:
        list of ndarray: List of 3D numpy arrays representing Laplacian
        pyramid
    '''
    num_img = len(gaussian_pyramid)
    #np.int16
    laplacian_pyramid = []
    for i in range(num_img-1):
        laplacian_pyramid.append(gaussian_pyramid[i].astype(np.int16)-upsample(gaussian_pyramid[i+1]))
    return laplacian_pyramid

def apply_mask(img_left, img_right, mask):
    # mask in [0,1]
    return (1-mask)*img_left + mask*img_right

def blend_images_from_pyramids(gaussian_pyramid_left, gaussian_pyramid_right, laplacian_pyramid_left, laplacian_pyramid_right, gaussian_pyramid_mask):
    depth = len(gaussian_pyramid_left)
    laplacian_pyramid_blended = []
    for i in range(depth-1):
        laplacian_pyramid_blended.append(apply_mask(laplacian_pyramid_left[i], laplacian_pyramid_right[i], gaussian_pyramid_mask[i]/255))

    curr_image = apply_mask(upsample(gaussian_pyramid_left[depth-1]), upsample(gaussian_pyramid_right[depth-1]), gaussian_pyramid_mask[depth-2]/255)
    curr_image = np.clip((curr_image + laplacian_pyramid_blended[depth-2]),0,255).astype(np.uint8)
    
    for i in range(depth-2):
        curr_image = np.clip((upsample(curr_image) + laplacian_pyramid_blended[depth-1-2-i]),0,255).astype(np.uint8)
    return curr_image

def blend_images(left_image, right_image):
    '''
    Smoothly blend two images by concatenation.
    
    Tip: This function should build Laplacian pyramids for both images,
    concatenate left half of left_image and right half of right_image
    on all levels, then start reconstructing from the smallest one.

    Args:
        left_image (ndarray): 3D numpy array representing an RGB image.
        right_image (ndarray): 3D numpy array representing an RGB image.

    Returns:
        ndarray: 3D numpy array representing an RGB image after blending.
    '''
    print("building pyramids")
    levels = 5
    size = 3
    gaussian_pyramid_left = build_gaussian_pyramid(left_image, size, levels)
    laplacian_pyramid_left = build_laplacian_pyramid(gaussian_pyramid_left)
    gaussian_pyramid_right = build_gaussian_pyramid(right_image, size, levels)
    laplacian_pyramid_right = build_laplacian_pyramid(gaussian_pyramid_right)

    h, w, n = left_image.shape
    shape_width_half = (h,w//2,n)
    mask = np.hstack((np.zeros(shape_width_half), np.full(shape_width_half,255))).astype(np.uint8)
    gaussian_pyramid_mask = build_gaussian_pyramid(mask, size, levels)

    print('blending images')
    return blend_images_from_pyramids(gaussian_pyramid_left, gaussian_pyramid_right, laplacian_pyramid_left, laplacian_pyramid_right, gaussian_pyramid_mask)
    
if __name__ == "__main__":

    ### Test Gaussian Filter ###
    dog_gaussian_noise = load_image('./images/dog_gaussian_noise.png')
    after_filter = apply_gaussian_filter(dog_gaussian_noise, gaussian_filter_size)
    save_image(after_filter, './dog_gaussian_noise_after.png')

    ### Test Median Filter ###
    dog_salt_and_pepper = load_image('./images/dog_salt_and_pepper.png')
    after_filter = apply_median_filter(dog_salt_and_pepper, median_filter_size)
    save_image(after_filter, './dog_salt_and_pepper_after.png')

    ### Test Image Blending ###
    player1 = load_image('./images/player1.png')
    player2 = load_image('./images/player2.png')
    after_blending = blend_images(player1, player2)
    save_image(after_blending, './player3.png')

    girl1 = load_image('./images/girl1.png')
    girl2 = load_image('./images/girl2.png')
    after_blending = blend_images(girl1, girl2)
    save_image(after_blending, './girl3.png')

    # Simple concatenation for comparison.
    save_image(concat(player1, player2), './player_simple_concat.png')
    save_image(concat(girl1, girl2), './girl_simple_concat.png')

