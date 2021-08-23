################################################  Formatting  ##########################################################

# based on: https://stackoverflow.com/a/62001539
def flatten_dict(input_node: dict, key_: str = '', output_dict: dict = {}):
    if isinstance(input_node, dict):
        for key, val in input_node.items():
            new_key = f"{key_}.{key}" if key_ else f"{key}"
            flatten_dict(val, new_key, output_dict)
    elif isinstance(input_node, list) or isinstance(input_node, tuple):
        for idx, item in enumerate(input_node):
            flatten_dict(item, f"{key_}.[{idx}]", output_dict)
    else:
        output_dict[key_] = input_node
    return output_dict


def dynamic_module_import(modules, globals):
    import importlib
    for module_name in modules:
        if not module_name.endswith(".py") or module_name.endswith("__.py"):
            continue
        module_name = module_name[:-3]
        module_name = module_name.replace("/", ".")
        module = __import__(module_name, fromlist=['*'])
        # importlib.import_module(module_name)
        if hasattr(module, '__all__'):
            all_names = module.__all__
        else:
            all_names = [name for name in dir(module) if not name.startswith('_')]
        globals.update({name: getattr(module, name) for name in all_names})


def adjust_len(a, b):
    # adjusts the len of two sorted lists
    al = len(a)
    bl = len(b)
    if al > bl:
        start = (al - bl) // 2
        end = bl + start
        a = a[start:end]
    if bl > al:
        a, b = adjust_len(b, a)
    return a, b


def circular_list(ls):
    class CircularList(list):
        def __getitem__(self, x):
            import operator
            if isinstance(x, slice):
                return [self[x] for x in self._rangeify(x)]

            index = operator.index(x)
            try:
                return super().__getitem__(index % len(self))
            except ZeroDivisionError:
                raise IndexError('list index out of range')

        def _rangeify(self, slice):
            start, stop, step = slice.start, slice.stop, slice.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            if step is None:
                step = 1
            return range(start, stop, step)
    return CircularList(ls)


def check_audio_in_video(filename):
    # TODO (fabawi): slows down the reading. Consider finding an alternative
    import subprocess
    import re
    mean_volume = subprocess.run("ffmpeg -hide_banner -i " + filename +
                                  " -af volumedetect -vn -f null - 2>&1 | grep mean_volume",
                                 stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    # if mean_volume is not None:
    #     mean_volume = float(re.search(r'mean_volume:(.*?)dB', mean_volume).group(1))
    # else:
    #     mean_volume = -91.0
    #
    # if mean_volume > -90.0:
    #     has_audio = True
    # else:
    #     has_audio = False
    if not mean_volume or '-91.0 dB' in mean_volume:
        has_audio = False
    else:
        has_audio = True
    return has_audio


def extract_width_height_from_video(filename):
    import cv2
    vcap = cv2.VideoCapture(filename)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vcap.release()
    return int(width), int(height)


def extract_thumbnail_from_video(filename, thumb_width=180, thumb_height=108, threshold=1):
    import cv2
    vcap = cv2.VideoCapture(filename)
    res, im_ar = vcap.read()
    while im_ar.mean() < threshold and res:
        res, im_ar = vcap.read()
    im_ar = cv2.resize(im_ar, (thumb_width, thumb_height), 0, 0, cv2.INTER_LINEAR)
    vcap.release()
    return im_ar


def extract_width_height_thumbnail_from_image(filename, thumb_width=180, thumb_height=108):
    import cv2
    im = cv2.imread(filename)
    height, width = im.shape[:2]
    im_ar = cv2.resize(im, (thumb_width, thumb_height), 0, 0, cv2.INTER_LINEAR)
    return int(width), int(height), im_ar


def encode_image(img, raw=False):
    import cv2
    import base64
    ret, jpeg = cv2.imencode('.jpg', img)
    if raw:
        enc_img = jpeg.tobytes()
        return enc_img
    else:
        enc_img = base64.b64encode(jpeg).decode('UTF-8')
        return 'data:image/jpeg;base64,{}'.format(enc_img)


def stack_images(grouped_video_frames_list, grabbed_video_list=None, plot_override=None):
    import numpy as np
    import cv2

    # resize_to_match = lambda img_src, img_tgt: cv2.resize(img_src, (img_tgt.shape[1], img_tgt.shape[0]), 0, 0, cv2.INTER_LINEAR)
    resize_to_match_y = lambda img_src, img_tgt: cv2.resize(img_src, (img_src.shape[1], img_tgt.shape[0]), 0, 0, cv2.INTER_LINEAR)
    resize_to_match_x = lambda img_src, img_tgt: cv2.resize(img_src, (img_tgt.shape[1], img_src.shape[0]), 0, 0, cv2.INTER_LINEAR)

    def stack_image(grouped_video_frames, grabbed_video, plot_override=None):
        if not grabbed_video:
            return None
        rows = []
        plot_frames = grouped_video_frames["PLOT"] if plot_override is None else plot_override
        for row in plot_frames:
            if len(row) > 1:
                rows.append(np.concatenate([
                    resize_to_match_y(grouped_video_frames[row_frame], grouped_video_frames[row[0]]) for row_frame in row], axis=1))
            else:
                rows.append(grouped_video_frames[row[0]])
        if len(rows) > 1:
            return np.concatenate([resize_to_match_x(row, rows[0]) for row in rows], axis=0)
        else:
            return rows[0]

    if isinstance(grouped_video_frames_list, list):
        frames_list = []
        for gv_idx, grouped_video_frames in enumerate(grouped_video_frames_list):
            if grabbed_video_list is not None:
                grabbed_video = grabbed_video_list[gv_idx]
            else:
                grabbed_video = True
            frames_list.append(stack_image(grouped_video_frames,
                                           grabbed_video=grabbed_video,
                                           plot_override=plot_override))
        return frames_list
    else:  # elif isinstance(grouped_video_frames_list, dict)
        return stack_image(grouped_video_frames_list,
                           grabbed_video=True if grabbed_video_list is None else grabbed_video_list,
                           plot_override=plot_override)


def aggregate_frame_ranges(frame_ids):
    """
    Compresses a list of frames to ranges
    :param frame_ids: assumes either a list of frame_ids or a list of lists [frame_id, duration]
    :return: list of frame ranges
    """
    import itertools
    frame_ids_updated = []

    if isinstance(frame_ids[0], list) or isinstance(frame_ids[0], tuple):
        frame_ids_updated.extend(i for frame_id, frame_duration in frame_ids for i in range(frame_id, frame_id + frame_duration))
    if not frame_ids_updated:
        frame_ids_updated = frame_ids
    frame_ids_updated = sorted(list(set(frame_ids_updated)))

    def ranges(frame_ids_updated):
        for a, b in itertools.groupby(enumerate(frame_ids_updated), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    return list(ranges(frame_ids_updated))


###################################################  Math  #############################################################

def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    import numpy as np
    patch_area = float(patch_size[0] * patch_size[1])
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)

    tl_x, tl_y = bbox[0], bbox[1]
    br_x, br_y = bbox[0] + bbox[2], bbox[1] + bbox[3]
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    for x in range(lx, ux + 1):
        for y in range(ly, uy + 1):
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[
                0], patch_tly + patch_size[1]

            aoi_tlx = tl_x if patch_tlx < tl_x else patch_tlx
            aoi_tly = tl_y if patch_tly < tl_y else patch_tly
            aoi_brx = br_x if patch_brx > br_x else patch_brx
            aoi_bry = br_y if patch_bry > br_y else patch_bry

            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max(
                (aoi_bry - aoi_tly), 0) / float(patch_area)

    return aoi_ratio

def multi_hot_coding(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    import numpy as np
    thresh = 0.5
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)
    hot_ind = aoi_ratio > thresh
    while hot_ind.sum() == 0:
        thresh *= 0.8
        hot_ind = aoi_ratio > thresh

    aoi_ratio[hot_ind] = 1
    aoi_ratio[np.logical_not(hot_ind)] = 0

    return aoi_ratio[0]

def pixels_to_bounded_range(xy_pix_max, xy_peak, xy_bounds=(-1, 1)):
    import numpy as np
    xy_pix_max = np.array(xy_pix_max)
    xy_peak = np.array(xy_peak)
    xy_peak = xy_peak / xy_pix_max
    xy_peak = (xy_bounds[1] - xy_bounds[0]) * xy_peak - xy_bounds[1]
    return xy_peak


def cartesian_to_spherical(xyz):
    import numpy as np
    ptr = np.zeros((3,))
    xy = xyz[0] ** 2 + xyz[1] ** 2
    ptr[0] = np.arctan2(xyz[1], xyz[0])
    ptr[1] = np.arctan2(xyz[2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    # ptr[1] = np.arctan2(np.sqrt(xy), xyz[2])  # for elevation angle defined from Z-axis down
    ptr[2] = np.sqrt(xy + xyz[2] ** 2)
    return ptr


def spherical_to_euler(pt):
    import numpy as np
    import math as m
    v1 = np.array([0, 0, -1])
    v2 = np.array([np.sin(pt[1]) * np.cos(pt[0]), np.sin(pt[1]) * np.sin(pt[0]), np.cos(pt[0])])
    Z = np.cross(v1, v2)
    Z /= np.sqrt(Z[0] ** 2 + Z[1] ** 2 + Z[2] ** 2)
    Y = np.cross(Z, v1)
    t = m.atan2(-Z[0], Z[1])
    p = m.asin(Z[0])
    psi = m.atan2(-Y[0], v1[0])
    return np.array([t, p, psi])


def foveal_to_mask(xy, radius, width, height):
    import numpy as np
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - xy[0]) ** 2 + (Y - xy[1]) ** 2)
    mask = dist <= radius
    return mask.astype(np.float32)


def mp_multivariate_gaussian(entries, width, height, xy_std=(10, 10)):
    import numpy as np
    from joblib import Parallel, delayed
    import time

    def multivariate_gaussian(x, y, xy_mean, width, height, xy_std, amplitude=64):
        if np.isnan(xy_mean[0]) == False and np.isnan(xy_mean[1]) == False:
            x0 = xy_mean[0]
            y0 = xy_mean[1]
            # now = time.time()
            result = amplitude * np.exp(
                -((((x - x0) ** 2) / (2 * xy_std[0] ** 2)) + (((y - y0) ** 2) / (2 * xy_std[1] ** 2))))
            # print('gaussian time:', time.time() - now)
            return result
        else:
            return np.zeros((height, width))


    # std_x = np.std(xyfix[:, 0]) / 10
    # std_y = np.std(xyfix[:, 1]) / 10
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    x, y = np.meshgrid(x, y, copy=False, sparse=True)
    results = Parallel(n_jobs=4, prefer="threads")(delayed(multivariate_gaussian)(x, y, (entries[i, 0], entries[i, 1],),
                                                                                  width, height, xy_std,
                                                                                  amplitude = entries[i, 2])
                                                   for i in range(entries.shape[0]))
    result = np.sum(results, axis=0)
    return result


# based on: https://stackoverflow.com/a/39823124/190597 (astrokeat)
def truncated_cone(xyz_orig, xyz_tgt, radius_orig, radius_tgt):
    from scipy.linalg import norm
    import numpy as np
    # vector in direction of axis
    v = xyz_tgt - xyz_orig
    # find magnitude of vector
    mag = norm(v)
    # unit vector in direction of axis
    v = v / mag
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # print n1,'\t',norm(n1)
    # normalize n1
    n1 /= norm(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 100
    t = np.linspace(0, mag, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    r = np.linspace(radius_orig, radius_tgt, n)
    # generate coordinates for surface
    x, y, z = [xyz_orig[i] + v[i] * t + r *
               np.sin(theta) * n1[i] + r * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    z = np.reshape(z, -1)
    x,y,z = x.astype(np.int64), y.astype(np.int64), z.astype(np.int64)
    return x, y, z


def conic_projection(xyz_orig, xyz_tgt, width, height, radius_orig=1, radius_tgt=10):
    import numpy as np
    from scipy.interpolate import griddata
    from scipy import ndimage

    x, y, z = truncated_cone(xyz_orig, xyz_tgt, radius_orig, radius_tgt)
    ###############################
    xi = np.linspace(0, width, width)
    yi = np.linspace(0, height, height)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="nearest")
    p2i = ndimage.gaussian_filter(zi, sigma=6)
    return p2i