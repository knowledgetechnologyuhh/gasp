import cv2
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

from gazenet.utils.helpers import circular_list, mp_multivariate_gaussian, conic_projection


class OpenCV(object):
    def __init__(self, colors=None, color_maps=None):
        if colors is None:
            self.colors = circular_list(
                [(0, 255, 0), (255, 0, 0), (0, 0, 255), (128, 0, 255), (255, 128, 0), (255, 0, 128), (0, 128, 255),
                 (0, 255, 128), (128, 255, 0), (255, 128, 64), (255, 64, 128), (128, 255, 64),
                 (128, 64, 255), (64, 128, 255), (64, 255, 128)])
        else:
            self.colors = colors
        if color_maps is None:
            self.color_maps = {"autumn": cv2.COLORMAP_AUTUMN,
                               "bone": cv2.COLORMAP_BONE,
                               "jet": cv2.COLORMAP_JET,
                               "winter": cv2.COLORMAP_WINTER,
                               "rainbow": cv2.COLORMAP_RAINBOW,
                               "ocean": cv2.COLORMAP_OCEAN,
                               "summer": cv2.COLORMAP_SUMMER,
                               "spring": cv2.COLORMAP_SPRING,
                               "cool": cv2.COLORMAP_COOL,
                               "hsv": cv2.COLORMAP_HSV,
                               "pink": cv2.COLORMAP_PINK,
                               "hot": cv2.COLORMAP_HOT}
        else:
            self.color_maps = color_maps

        self.interpolation = {"nearest": cv2.INTER_NEAREST,
                              "linear": cv2.INTER_LINEAR,
                              "area": cv2.INTER_AREA,
                              "cubic": cv2.INTER_CUBIC,
                              "lanczos": cv2.INTER_LANCZOS4,
                              }

    def __prep_col_image__(self, frame=None, color_id=None):
        if frame is not None:
            frame = frame.copy()
        if color_id is None:
            color = self.colors[0]
        elif isinstance(color_id, tuple):
            color = color_id
        else:
            color = self.colors[color_id]
        return frame, color

    def __prep_map_image__(self, frame=None, color_map=None):
        if frame is not None:
            frame = frame.copy()
        if color_map is None:
            color = self.color_maps["jet"]
        elif isinstance(color_map, str):
            color = self.color_maps[color_map]
        else:
            color = color_map
        return frame, color

    def resize(self, frame, width=None, height=None, interpolation="nearest"):
        if isinstance(interpolation, str):
            interpolation = self.interpolation[interpolation]
        frame = frame.copy()
        frame = cv2.resize(frame, (width, height), interpolation)
        return frame

    def plot_color_map(self, np_frame, color_map=None):
        _, color = self.__prep_map_image__(None, color_map)
        frame = cv2.applyColorMap(np_frame, colormap=color)
        return frame

    def plot_point(self, frame, xy, color_id=None, radius=10, thickness=-1):
        frame, color = self.__prep_col_image__(frame, color_id)
        cv2.circle(frame, xy, radius, color, thickness)
        return frame

    def plot_bbox(self, frame, xy_min, xy_max, color_id=None, thickness=2):
        frame, color = self.__prep_col_image__(frame, color_id)
        cv2.rectangle(frame, xy_min, xy_max, color, thickness)
        return frame

    def plot_text(self, frame, text, xy, color_id=None, thickness=2, font_scale=0.5):
        frame, color = self.__prep_col_image__(frame, color_id)
        cv2.putText(frame, str(text), xy,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return frame

    def plot_arrow(self, frame, xy_orig, xy_tgt, color_id=None, thickness=2):
        frame, color = self.__prep_col_image__(frame, color_id)
        cv2.arrowedLine(frame, xy_orig, xy_tgt, color, thickness)
        return frame

    def plot_axis(self, frame, xy_min, xy_max, xyz, thickness=2):
        frame = frame.copy()
        pitch = xyz[0]
        yaw = -xyz[1]
        roll = xyz[2]

        xy_min = np.array(xy_min)
        xy_max = np.array(xy_max)
        size = np.linalg.norm(xy_max - xy_min)
        xy1 = xy_min + ((xy_max - xy_min) / 2)
        xy1 = xy1.astype("int32")

        tdx = xy1[0]
        tdy = xy1[1]

        # X-Axis pointing to right. drawn in red
        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
        y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
        y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (np.sin(yaw)) + tdx
        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

        cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness)
        cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness)
        cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness)

        return frame

    def plot_fov_mask(self, frame, xy, radius=50, thickness=-1):
        frame = frame.copy()
        mask = np.zeros_like(frame)
        cv2.circle(mask, xy, radius, (255, 255, 255), thickness=thickness)
        mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
        frame = frame * (mask_blur / 255)
        return frame

    def plot_conic_field(self, frame, xyz_orig, xyz_tgt, radius_orig=1, radius_tgt=10, color_map=None):
        frame, color = self.__prep_map_image__(frame, color_map)
        h, w, _ = frame.shape
        xyz_orig = np.array(xyz_orig[:3])
        xyz_tgt = xyz_orig + np.array(xyz_tgt[:3]) * 10
        p2i = conic_projection(xyz_orig, xyz_tgt, width=w, height=h, radius_orig=radius_orig, radius_tgt=radius_tgt)
        frame = self.plot_color_map(255 - np.uint8(((p2i - p2i.min()) / (p2i.max() - p2i.min())) * 255), color_map=color)
        return frame

    def plot_alpha_overlay(self, frame, overlay, xy_min=None, xy_max=None, alpha=0.2, interpolation="nearest"):
        frame = frame.copy()
        overlay = overlay.copy()
        h, w, _ = frame.shape
        ho, wo, co = overlay.shape
        if xy_min is None and xy_max is None:
            if w != wo or h != ho:
                overlay = self.resize(overlay, height=h, width=w, interpolation=interpolation)
        else:
            y1, y2 = max(0, xy_min[1]), min(h, xy_max[1])
            x1, x2 = max(0, xy_min[0]), min(w, xy_max[0])
            # add alpha channel
            tmp = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            _, a = cv2.threshold(tmp, 0, 255, cv2.THRESH_TOZERO)
            b, g, r = cv2.split(overlay)
            rgba = [b, g, r, a]
            overlay = cv2.merge(rgba, 4)
            # resize overlay to fit bbox
            overlay = self.resize(overlay, width=x2 - x1, height=y2 - y1)
            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            overlay_full = np.zeros_like(frame)
            for c in range(0, 3):
                overlay_full[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
            overlay = overlay_full

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    #######################################  Multiple points #################################################

    def plot_fixations_density_map(self, frame, xy_fix, xy_std=(10, 10), alpha=0.2, color_map=None):
        frame, color = self.__prep_map_image__(frame, color_map)
        height, width = frame.shape[:2]
        heatmap = mp_multivariate_gaussian(xy_fix, width=width, height=height, xy_std=xy_std)
        heatmap = np.divide(heatmap, np.amax(heatmap), out=heatmap, where=np.amax(heatmap) != 0)
        heatmap *= 255
        heatmap = heatmap.astype("uint8")

        overlay = self.plot_color_map(heatmap, color_map=color)

        overlay = overlay.astype("uint8")
        frame = self.plot_alpha_overlay(frame, overlay, alpha=alpha)
        return frame

    def plot_fixations_locations(self, frame, xy_fix, radius=10):
        frame = frame.copy()
        for xy in xy_fix:
            try:
                cv2.circle(frame, (int(xy[0]), int(xy[1])), radius, (255,255,255), -1)
            except ValueError:
                pass
        return frame

class MatplotLib(object):
    def __init__(self):
        raise NotImplementedError("Matplotlib plotting not yet supported")
