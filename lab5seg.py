import numpy
import cv2
### Альтернативный метод сегментаци изображения на строки

# загрузить изображение в оттенках серого
def segments(filename: str) -> []:
    src_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    ### Подготовка изображения

    # инвертировать (при необходимости)
    src_image = numpy.max(src_image) - src_image
    # сгладить (размер ядра подбирается по размеру символов и уровню помех)
    image = cv2.GaussianBlur(src_image, (7, 7), 0)
    # бинаризовать (с использованием метода Оцу)
    level, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)


    # функция формирования списка диапазонов ненулевых значений в массиве
    def find_ranges(arr):
        # индекс начала текущего диапазона
        start = None if arr[0] == 0 else 0
        bounds = []
        for i in range(1, arr.shape[0]):
            if start is None:
                if arr[i] > 0:
                    # начинаем новый дипазон
                    start = i
            else:
                if arr[i] == 0:
                    # закрываем и запоминаем диапазон
                    bounds.append([start, i])
                    start = None
        if start is not None:
            # закрываем последний открытый диапазон
            bounds.append([start, arr.shape[0]])
        return bounds


    # функция формирования образцов для распознавания
    def fit_size(image, rsize=(28, 28), offset=2):
        # заготовка результата
        result = numpy.zeros( rsize, dtype=image.dtype )
        # исходный размер изображения
        h, w = image.shape
        if w > h:
            # вписываем по ширине
            tw = rsize[0] - offset * 2
            th = int(tw*h/w)
            px = offset
            py = (rsize[1] - th) // 2
        else:
            # вписываем по высоте
            th = rsize[1] - offset * 2
            tw = int(th*w/h)
            px = (rsize[0] - tw) // 2
            py = offset

        # tw, th - итоговый размер с учётом пропорций и рамок
        # px, py - положение в финальном изображении

        # масштабируем
        rimage = cv2.resize(image, (tw, th), cv2.INTER_AREA)
        # помещаем в финальное изображение
        result[py:py+th, px:px+tw] = rimage
        return result


    # свёртка изображения по горизонтали
    folded_x = numpy.sum(image, axis=1)
    # диапазоны вертикальных координат для строк
    line_ranges = find_ranges(folded_x)
    # изображения строк
    line_fragments = [image[start:end, :] for start, end in line_ranges]

    samples = []
    for line_image in line_fragments:
        # свёртка изображения по вертикали
        folded_y = numpy.sum(line_image,  axis=0)
        # диапазоны горизонтальных координат для символов в строке
        ranges = find_ranges(folded_y)

        fragments = []
        for start, end in ranges:
            # изображение символа
            frag = line_image[:, start:end]
            # убираем отступы сверху и снизу
            r = find_ranges(numpy.sum(frag, axis=1))
            # верхний отступ (начало первого диапазона)
            top = r[0][0]
            # нижний отступ (конец последнего диапазона)
            bottom = r[-1][1]
            fragments.append(frag[top:bottom, :])

        # высота строки в пикселях
        line_height = line_image.shape[0]
        # отбрасываем фрагменты недостаточной высоты
        line_samples = [fit_size(s) for s in fragments if s.shape[0] > 0.5*line_height]
        samples.append(line_samples)
    return samples