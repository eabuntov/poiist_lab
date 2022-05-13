import numpy as np
import cv2 as cv


def segments(filename: str) -> []:
    src_image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    src_image = np.max(src_image) - src_image
    image = cv.GaussianBlur(src_image, (7, 7), 0)
    level, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY or cv.THRESH_OTSU)

    def find_ranges(arr):
        start = None if arr[0] == 0 else 0
        bounds = []
        for i in range(1, arr.shape[0]):
            if start is None:
                if arr[i] > 0:
                    start = i
            else:
                if arr[i] == 0:
                    bounds.append([start, i])
                    start = None
        if start is not None:
            bounds.append([start, arr.shape[0]])
        return bounds

    def fit_size(img, final_size=(28, 28), offset=2):
        result = np.zeros(final_size, dtype=img.dtype)
        h, w = img.shape
        if w > h:
            tw = final_size[0] - offset * 2
            th = int(tw * h / w)
            px = offset
            py = (final_size[1] - th) // 2
        else:
            th = final_size[1] - offset * 2
            tw = int(th * w / h)
            px = (final_size[0] - tw) // 2
            py = offset

        result[py:py + th, px:px + tw] = cv.resize(img, (tw, th), cv.INTER_AREA)
        return result

    folded_x = np.sum(image, axis=1)
    line_ranges = find_ranges(folded_x)
    line_fragments = [image[start:end, :] for start, end in line_ranges]

    samples = []
    for line_image in line_fragments:
        folded_y = np.sum(line_image, axis=0)
        ranges = find_ranges(folded_y)
        fragments = []
        for start, end in ranges:
            frag = line_image[:, start:end]
            r = find_ranges(np.sum(frag, axis=1))
            top = r[0][0]
            bottom = r[-1][1]
            fragments.append(frag[top:bottom, :])

        line_height = line_image.shape[0]
        line_samples = [fit_size(s) for s in fragments if s.shape[0] > 0.5 * line_height]
        samples.append(line_samples)
    return samples
