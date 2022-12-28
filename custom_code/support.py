from PIL import Image
import os


def get_class_name(class_id):
    if class_id < 0 or class_id > 3:
        return 'ERROR'
    classes = ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Potholes']
    return classes[class_id]


def yolo2real(result_files):
    img, txt = result_files
    image1 = Image.open(img)
    width, height = image1.size
    with open(txt, 'r') as f:
        lines = f.readlines()
        newLines = []
        lines.reverse()
        for line in lines:
            lineSpt = line.split()
            rdd_cls = int(lineSpt[0])
            x_c = float(lineSpt[1])
            y_c = float(lineSpt[2])
            x_w = float(lineSpt[3])
            y_h = float(lineSpt[4])

            x0 = int((x_c - x_w / 2) * width)
            if x0 < 0:
                x0 = 0
            y0 = int((y_c - y_h / 2) * height)
            if y0 < 0:
                y0 = 0
            x1 = int((x_c + x_w / 2) * width)
            if x1 > width:
                x1 = width
            y1 = int((y_c + y_h / 2) * height)
            if y1 > height:
                y1 = height

            newLines.append(f'{rdd_cls} {x0} {y0} {x1} {y1} ')

    with open('.' + txt.split('.')[1] + '.json', 'w+') as f2:
        f2.write('[')
        for idx, line in enumerate(newLines):
            lineSpt = line.split()
            class_id = int(lineSpt[0])
            _, x0, y0, x1, y1 = lineSpt
            f2.write('{ ')
            f2.write('"class": ' + str(class_id) + ', ')
            f2.write('"name": "' + get_class_name(class_id) + '", ')
            f2.write('"x0": ' + x0 + ', ')
            f2.write('"y0": ' + y0 + ', ')
            f2.write('"x1": ' + x1 + ', ')
            f2.write('"y1": ' + y1)
            f2.write(' }')
            if idx != len(newLines) - 1:
                f2.write(", ")
        f2.write(']')


def get_latest_result():
    result = os.walk('./runs/detect/')
    _, subdirs, _ = next(result)
    latest_result = sorted(subdirs, reverse=True)[0]
    result_files = []

    for path, subdirs, files in os.walk(f'./runs/detect/{latest_result}/'):
        for name in files:
            result_files.append(os.path.join(path, name))

    return result_files


def prepare_result():
    result_files = get_latest_result()

    yolo2real(result_files)
    return result_files
