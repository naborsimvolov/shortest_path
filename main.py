from PIL import Image, ImageDraw

images = []

import imageio

import cv2  # import OpenCV library
import numpy as np

filenames = ['img/mro0.jpg', 'img/mro1.jpg', 'img/mro2.jpg']

inum = int(input('Type image number: '))

xn = yn = int(input("Set the size of grid: "))

st_x = int(input('Type start_X coordinate:'))
st_y = int(input('Type start_y coordinate:'))
e_x = int(input('Type end_x coordinate:'))
e_y = int(input('Type end_X coordinate:'))

# output_d = str(input('whether to print the state of the varibly "grid" or not?'))
# output_pr = str(input('whether to print the state of the varibly "pr" or not?'))
output_d = 0
output_pr = 0
img_n = cv2.imread(filenames[inum], 1)  # read image 1-rgb, 0-greyscale, -1-rgba
img = img_n[0: 750, 0: 900]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

height, width, channels = img.shape

w = width / xn  # размер окна по x
h = height / yn  # размер окна по y

if inum == 0:
    h1, s1, v1, h2, s2, v2 = 19, 0, 134, 135, 160, 255

elif inum == 1:
    h1, s1, v1, h2, s2, v2 = 53, 42, 79, 202, 135, 148

elif inum == 2:
    h1, s1, v1, h2, s2, v2 = 30, 0, 0, 255, 91, 255

elif inum == 3:
    h1, s1, v1, h2, s2, v2 = 0, 51, 149, 130, 220, 255

h_min = np.array((h1, s1, v1), np.uint8)
h_max = np.array((h2, s2, v2), np.uint8)
mask = cv2.inRange(hsv, h_min, h_max)  # маска

count_img = cv2.countNonZero(mask)  # вычисление белых точек на изображении

okna = xn * yn  # общее кол во окон
okna1 = okna + 1

roi = [[0] * xn for i in range(yn)]  # массив, каждый элемент - одно окно
count_r = [[0] * xn for i in range(yn)]  # массив, содержащий количество белых пикселей на выбранном элементе картинки
pr_r = [[0] * xn for i in range(yn)]  # тоже самое, что count_r, но в процентном соотношении
grid = [[0] * xn for i in range(yn)]  # массив, который будет содержать 1 и 0

for i in range(yn):
    img = cv2.line(img, (0, int(h * (i + 1))), (width, int(h * (i + 1))), (0, 255, 0), 1)
    mask = cv2.line(mask, (0, int(h * (i + 1))), (width, int(h * (i + 1))), (255, 255, 255), 1)
    for j in range(xn):
        roi[i][j] = mask[int(h * i): int(h * (i + 1)), int(w * j): int(w * (j + 1))]
        count_r[i][j] = cv2.countNonZero(roi[i][j])
        pr_r[i][j] = count_r[i][j] / count_img * 100

        if i == 0:
            img = cv2.line(img, (int(w * (j + 1)), 0), (int(w * (j + 1)), height), (0, 255, 0), 1)
            mask = cv2.line(mask, (int(w * (j + 1)), 0), (int(w * (j + 1)), height), (255, 255, 255), 1)

medium = np.mean(count_r)  # среднее арифметическое белых точек
for i in range(yn):
    for j in range(xn):
        if count_r[i][j] <= medium:
            grid[i][j] = 0
        else:
            grid[i][j] = 1

        if output_pr == 'y' or output_pr == 'yes' or int(output_pr) == 1:
            print('percent of barrier in roi', [i], [j], ':', pr_r[i][j], '%')
        if output_d == 'y' or output_d == 'yes' or int(output_d) == 1:
            print('arrange grid', [i], [j], ':', grid[i][j])

a = grid
zoom = 20
borders = 6


def make_step(k):
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] == k:
                if i > 0 and m[i - 1][j] == 0 and a[i - 1][j] == 0:
                    m[i - 1][j] = k + 1
                if j > 0 and m[i][j - 1] == 0 and a[i][j - 1] == 0:
                    m[i][j - 1] = k + 1
                if i < len(m) - 1 and m[i + 1][j] == 0 and a[i + 1][j] == 0:
                    m[i + 1][j] = k + 1
                if j < len(m[i]) - 1 and m[i][j + 1] == 0 and a[i][j + 1] == 0:
                    m[i][j + 1] = k + 1

                if i - 1 > 0 and j - 1 > 0 and m[i - 1][j - 1] == 0 and a[i - 1][j - 1] == 0:
                    m[i - 1][j - 1] = k + 1
                if i - 1 > 0 and j + 1 < len(m[i]) and m[i - 1][j + 1] == 0 and a[i - 1][j + 1] == 0:
                    m[i - 1][j + 1] = k + 1
                if i + 1 < len(m) and j - 1 > 0 and m[i + 1][j - 1] == 0 and a[i + 1][j - 1] == 0:
                    m[i + 1][j - 1] = k + 1
                if i + 1 < len(m) and j + 1 < len(m[i]) and m[i + 1][j + 1] == 0 and a[i + 1][j + 1] == 0:
                    m[i + 1][j + 1] = k + 1


def draw_matrix(a, m, the_path=[]):
    im = Image.new('RGB', (zoom * len(a[0]), zoom * len(a)), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(len(a)):
        for j in range(len(a[i])):
            color = (255, 255, 255)
            r = 0
            if a[i][j] == 1:
                color = (0, 0, 0)
            if i == st_x and j == st_y:
                color = (0, 255, 0)
                r = borders
            if i == e_x and j == e_y:
                color = (0, 255, 0)
                r = borders
            draw.rectangle((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1), fill=color)
            if m[i][j] > 0:
                r = borders
                draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                             fill=(255, 0, 0))
    for u in range(len(the_path) - 1):
        y = the_path[u][0] * zoom + int(zoom / 2)
        x = the_path[u][1] * zoom + int(zoom / 2)
        y1 = the_path[u + 1][0] * zoom + int(zoom / 2)
        x1 = the_path[u + 1][1] * zoom + int(zoom / 2)
        draw.line((x, y, x1, y1), fill=(255, 0, 0), width=5)
    draw.rectangle((0, 0, zoom * len(a[0]), zoom * len(a)), outline=(0, 255, 0), width=2)
    images.append(im)


m = []
for i in range(len(a)):
    m.append([])
    for j in range(len(a[i])):
        m[-1].append(0)
i = st_x
j = st_y
m[i][j] = 1

k = 0
while m[e_x][e_y] == 0:
    k += 1
    make_step(k)
    draw_matrix(a, m)

i = e_x
j = e_y
k = m[i][j]
the_path = [(i, j)]
while k > 1:
    if i > 0 and m[i - 1][j] == k - 1:
        i, j = i - 1, j
        the_path.append((i, j))
        k -= 1
    elif j > 0 and m[i][j - 1] == k - 1:
        i, j = i, j - 1
        the_path.append((i, j))
        k -= 1
    elif i < len(m) - 1 and m[i + 1][j] == k - 1:
        i, j = i + 1, j
        the_path.append((i, j))
        k -= 1
    elif j < len(m[i]) - 1 and m[i][j + 1] == k - 1:
        i, j = i, j + 1
        the_path.append((i, j))
        k -= 1
    elif i - 1 > 0 and j - 1 > 0 and m[i - 1][j - 1] == k - 1:
        i, j = i - 1, j - 1
        the_path.append((i, j))
        k -= 1
    elif i + 1 < len(m) and j - 1 > 0 and m[i + 1][j - 1] == k - 1:
        i, j = i + 1, j - 1
        the_path.append((i, j))
        k -= 1
    elif i - 1 > 0 and j + 1 < len(m[i]) and m[i - 1][j + 1] == k - 1:
        i, j = i - 1, j + 1
        the_path.append((i, j))
        k -= 1
    elif i + 1 < len(m) and j + 1 < len(m[i]) and m[i + 1][j + 1] == k - 1:
        i, j = i + 1, j + 1
        the_path.append((i, j))
        k -= 1

    draw_matrix(a, m, the_path)

for i in range(10):
    if i % 2 == 0:
        draw_matrix(a, m, the_path)
    else:
        draw_matrix(a, m)

images[0].save('path.gif',
               save_all=True, append_images=images[1:],
               optimize=False, duration=1, loop=0)

path = imageio.get_reader('path.gif')
path_cv = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in path]
nums = len(path)

print('total fragments:', xn * yn)
print(the_path)

b = 0
while True:
    cv2.imshow("path", path_cv[b])
    cv2.imshow("image", img)
    cv2.imshow('mask', mask)
    if cv2.waitKey(100) & 0xFF == 27:
        break
    b = (b + 1) % nums
cv2.destroyAllWindows()
