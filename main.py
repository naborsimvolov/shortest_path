#импортитруем необходимые библиотеки

from PIL import Image, ImageDraw
import imageio
import cv2
import numpy as np

#создаем массив изображений и наименований картинок
images = []
filenames = ['img/mro0.jpg', 'img/mro1.jpg', 'img/mro2.jpg', 'img/high0.png']

#пользователь выбирает изображение для работы и задает размеры сетки и координаты конца и начала маршрута
inum = int(input('Type image number: '))
xn = yn = int(input("Set the size of grid: "))
e_x = int(input('Type start_Y coordinate:'))
e_y = int(input('Type start_X coordinate:'))
st_x = int(input('Type end_Y coordinate:'))
st_y = int(input('Type end_X coordinate:'))

#читаем выбранную картинку и переводим в hsv для дальнейшей обработки
img = cv2.imread(filenames[inum], 1)  # read image 1-rgb, 0-greyscale, -1-rgba
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
height, width, channels = img.shape #вычисляем размеры картинки

w = width / xn  # вычисляем размер окна по x и y
h = height / yn

#выбираем цветовой фильтр в зависимости от выбранного пользователем изображения
if inum == 0:
    h1, s1, v1, h2, s2, v2 = 19, 0, 134, 135, 160, 255

elif inum == 1:
    h1, s1, v1, h2, s2, v2 = 53, 42, 79, 202, 135, 148

elif inum == 2:
    h1, s1, v1, h2, s2, v2 = 30, 0, 0, 255, 91, 255

elif inum == 3:
    h1, s1, v1, h2, s2, v2 = 0, 51, 149, 130, 220, 255

elif inum == 4:
    h1, s1, v1, h2, s2, v2 = 15, 15, 15, 255, 255, 255

# накладываем маску на изображение для получения черно-белого изображения, где белым остануться препятствия, а черным - там, где можно ехать
h_min = np.array((h1, s1, v1), np.uint8)
h_max = np.array((h2, s2, v2), np.uint8)
mask = cv2.inRange(hsv, h_min, h_max)

count_img = cv2.countNonZero(mask)  # вычисление белых точек на изображении

okna = xn * yn  # общее кол во окон
okna1 = okna + 1

roi = [[0] * xn for i in range(yn)]  # массив, каждый элемент - одно окно
count_r = [[0] * xn for i in range(yn)]  # массив, содержащий количество белых пикселей на выбранном элементе картинки
pr_r = [[0] * xn for i in range(yn)]  # тоже самое, что count_r, но в процентном соотношении
grid = [[0] * xn for i in range(yn)]  # массив, который будет содержать 1 и 0

#заполняем массив roi, count_r и pr_r
#а также дорисовывем линии на изображении для наглядности
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

medium = np.mean(count_r)  # подсчитывем среднее арифметическое белых точек

#если на сегменте кол-во белых точек(препятствий) больше, чем среднее ар. по всем, то считаем этот фрагмент как препятствие и приписываем значение 1
for i in range(yn):
    for j in range(xn):
        if count_r[i][j] <= medium:
            grid[i][j] = 0
        else:
            grid[i][j] = 1


a = grid
#задаем отрисовки изображения
zoom = 20
borders = 6

#функция шага
#записываем во все соседние клетки начальной клетки число на 1 больше предыдущего, пока не дойдем до конца
#таким образом получается матрица, заполненная числами
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

#отрисовываем матрицу с помощью библиотеки PIL, на основе нашей сетки
def draw_matrix(a, m, the_path=[]):
    im = Image.new('RGB', (zoom * len(a[0]), zoom * len(a)), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for i in range(len(a)):
        for j in range(len(a[i])):
            color = (0, 0, 0)
            r = 0
            if a[i][j] == 1:
                color = (255, 255, 255)
            draw.rectangle((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1), fill=color)
            if m[i][j] > 0:
                r = borders
                draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                             fill=(130, 130, 130))
            if i == st_x and j == st_y:
                r = borders
                draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                             fill=(255, 0, 0))
            if i == e_x and j == e_y:
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

#создаем пока что пустой массив m
m = []
for i in range(len(a)):
    m.append([])
    for j in range(len(a[i])):
        m[-1].append(0)

m[st_x][st_y] = 1

#заполняем массив числами с помощью ранее созданной функции шага пока не дойдем до клетки, обозначенной как финиш
k = 0
while m[e_x][e_y] == 0:
    k += 1
    make_step(k)
    draw_matrix(a, m)

#создаем массив, в который в дальнешем записываем координаты маршрута
i = e_x
j = e_y
k = m[i][j]
the_path = [(i, j)]

#начиная с клетки финиша, заданной пользователем
#смотрим на все клетки вокруг, и если значение клетки меньше на 1, то
#записываем ее в путь и переходим в нее
#и зарисовываем картинку
#и так до того момента, пока не окажемся в клетке старта, заданной пользователем
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


#сохраняем изображение как gif
images[0].save('path.gif',
               save_all=True, append_images=images[1:],
               optimize=False, duration=1, loop=0)

#читаем изображение и преобразуем его в cv
path = imageio.get_reader('path.gif')
path_cv = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in path]
nums = len(path)

print('total fragments:', xn * yn)
print('path length:', len(the_path))

b = 0

while True:
    # выводим изображение
    cv2.imshow("path", path_cv[b])
    cv2.imshow("image", img)
    cv2.imshow('mask', mask)
    #программа работает пока не нажата клавиша esc
    if cv2.waitKey(100) & 0xFF == 27:
        break
    b = (b + 1) % nums
cv2.destroyAllWindows()
