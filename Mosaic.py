# -*- coding: UTF-8 -*-
import time
import re
import tkinter as tk
from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog, ttk, messagebox
from tkinter.filedialog import askdirectory
from PIL import Image
import cv2
from skimage import io
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import warnings


def check_resize(Object_filename, pixel_x, pixel_y, main_directory, picture_name):
    """
    this method is to check whether resize operation is needed
    :return:
    """
    pic = cv2.imread(Object_filename)
    pic_array = pic.shape  # shape of the picture
    assert pic_array[2] == 3
    pic_row = pic_array[0]  # how many rows of the picture
    pic_column = pic_array[1]  # how many columns of the picture
    resize_need = False
    # to make sure the final picture can be divided into a number of pixels without any space left
    y_remainder = pic_row % pixel_y
    x_remainder = pic_column % pixel_x
    if y_remainder != 0:
        pic_row = pic_row + (pixel_y - y_remainder)
        assert pic_row % pixel_y == 0
        resize_need = True
    if x_remainder != 0:
        pic_column = pic_column + (pixel_x - x_remainder)
        assert pic_column % pixel_x == 0
        resize_need = True
    if resize_need:
        pic_resized = cv2.resize(pic, (pic_column, pic_row), Image.ANTIALIAS,
                                 interpolation=cv2.INTER_AREA)  # resize the picture
        os.chdir(main_directory)
        cv2.imwrite('resized_' + picture_name, pic_resized)
    else:
        os.chdir(main_directory)
        cv2.imwrite('resized_' + picture_name, pic)


def main_color(pic_name, pixel_x, pixel_y, k=1):
    """
    The program use the KMeans algorithm, the idea from  Github: Siyuan Li.
    :param k:
    :param pic_name:
    :return: a group of list showing the main RGB color of the picture
    """
    warnings.filterwarnings('ignore')
    pic = cv2.imread(pic_name)  # 读取图片
    image = cv2.resize(pic, (pixel_x, pixel_y), Image.ANTIALIAS, interpolation=cv2.INTER_AREA)  # 将图片尺寸改成需要的大小
    new_name = pic_name + '_resized.jpg'
    cv2.imwrite(new_name, image)
    img = io.imread(new_name)  # 读取图片
    img_ori_shape = img.shape  # 图片的维度形状
    assert img_ori_shape[2] == 3
    img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))  # 转换数组的维度为二维数组
    img_shape = img1.shape  # 更改后图片的维度形状
    assert img_shape[1] == 3
    n_channels = img_shape[1]  # 获取图片的维度
    clf = KMeans(n_clusters=k)  # 构造聚类器
    clf.fit(img1)  # 聚类
    centroids = clf.cluster_centers_  # 获取的聚类中心
    labels = list(clf.labels_)  # 标签
    os.remove(new_name)
    color_info = {}
    for center_index in range(k):
        colorRatio = labels.count(center_index) / len(labels)  # 获取这个颜色中心点的个数占总中心点的ratio
        key = colorRatio
        value = list(centroids[center_index])  # 将对应中心点对应ratio存入字典中
        color_info.__setitem__(key, value)
    color_info_sorted = sorted(color_info.keys(), reverse=True)
    colorInfo = [(k, color_info[k]) for k in color_info_sorted]
    return colorInfo[0][1]


def pixelizer(picture_name, pixel_x, pixel_y):
    """
       this method is to discretisize the picture into a number of pixels and get the RGB value of every pixel
       :return:
       """
    pic_name = 'resized_' + picture_name
    count = 0
    if not os.path.exists(pic_name[0:-4]):  # if the directory does not exist, create a new one
        os.mkdir(pic_name[0:-4])
    else:  # if exist, pass
        shutil.rmtree(pic_name[0:-3])
        os.mkdir(pic_name[0:-3])
    object_pic = cv2.imread(pic_name)
    pic_array = np.array(object_pic)
    height = pic_array.shape[0]
    assert height % pixel_y == 0
    width = pic_array.shape[1]
    assert width % pixel_x == 0
    num_x = width / pixel_x
    num_y = height / pixel_y
    os.chdir(pic_name[0:-3])
    #     divide the picture into pixels
    for x in range(int(num_x)):
        for y in range(int(num_y)):
            count += 1
            x1 = x * pixel_x
            x2 = (x + 1) * pixel_x - 1
            y1 = y * pixel_y
            y2 = (y + 1) * pixel_y - 1
            cropImg = object_pic[y1:y2, x1:x2]
            cv2.imwrite(str(count) + '.bmp', cropImg)  # crop image to get every pixel


def ft(t):
    """
    use this function to help convert the XYZ color to L*ab color
    :param t:
    :return: f(t) result of parameter t
    """
    ft = t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 4 / 29
    return ft


def gamma(x):
    """
    use this function to transfer the rgb value to RGB value that can be further converted to XYZ color
    :param x:
    :return: gamma function result
    """
    gammax = ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92
    return gammax


def rgb2Lab(rgbvalue):
    """
    this method convert the RGB color to L*ab color that can be used to calculate the deltaE value of two color
    :param rgbvalue: rgb color value, a list of three values
    :return: a list contains L*ab color value
    """
    RGB2Lab_Matrix = np.array([[0.412453, 0.357580, 0.180423],
                               [0.212671, 0.715160, 0.072169],
                               [0.019334, 0.119193, 0.950227]])
    R = rgbvalue[0]
    G = rgbvalue[1]
    B = rgbvalue[2]
    gammaR = gamma(R / 255.0)
    gammaG = gamma(G / 255.0)
    gammaB = gamma(B / 255.0)
    RGBvalue = np.array([gammaR, gammaG, gammaB])
    RGBvalue = RGBvalue.reshape(3, 1)
    XYZvalue = np.dot(RGB2Lab_Matrix, RGBvalue)
    assert XYZvalue.shape == (3, 1)
    correction = np.array([[1.0 / 0.950456, 1.0, 1.0 / 1.088754]]).T
    assert correction.shape == (3, 1)
    XYZ = XYZvalue * correction
    assert XYZ.shape == (3, 1)
    YYn = ft(XYZ[1])
    XXn = ft(XYZ[0])
    ZZn = ft(XYZ[2])
    L = 116 * YYn - 16
    a = 500 * (XXn - YYn)
    b = 200 * (YYn - ZZn)
    return [int(L), int(a), int(b)]


def deltaE76(color1, color2):
    """
    :param color1: the L*ab color value of color 1, it is list of three float value
    :param color2: the L*ab color value of color 1, it is list of three float value
    :return: the deltaE value of these two color based on CIE 1976 standard
    """
    assert len(color1) == 3
    assert len(color2) == 3
    deltaE = (((color1[0] - color2[0]) ** 2) + ((color1[1] - color2[1]) ** 2) + (
            (color1[2] - color2[2]) ** 2)) ** 0.5
    return deltaE


def deltaE94(color1, color2):
    """
    :param color1: the L*ab color value of color 1, it is list of three float value
    :param color2: the L*ab color value of color 1, it is list of three float value
    :return: the deltaE value of these two color based on CIE 1994 standard
    """
    assert len(color1) == 3
    assert len(color2) == 3
    C1 = (color1[1] ** 2 + color1[2] ** 2) ** 0.5
    C2 = (color2[1] ** 2 + color2[2] ** 2) ** 0.5
    deltaC = C1 - C2
    deltaa = color1[1] - color2[1]
    deltab = color1[2] - color2[2]
    deltaH = (deltaa ** 2 + deltab ** 2 - deltaC ** 2) ** 0.5
    deltaE1994 = (((color1[0] - color2[0]) ** 2) + ((deltaC / (1 + 0.045 * C1)) ** 2) + (
            (deltaH ** 2) / ((1 + 0.015 * C1) ** 2))) ** 0.5
    if isinstance(deltaE1994, complex):
        deltaE1994 = deltaE1994.real
    return deltaE1994


def create_pixels_color_library(object_pixels_path, pixel_x, pixel_y):
    """
    create an dictionary that contains the Lab color value of every picture in the directory
    :param object_pixels_path:
    :return: dictionary with picture filename as key and Lab color value as item
    """
    os.chdir(object_pixels_path)
    library_object = {}
    start_time = time.time()
    for file in os.listdir(object_pixels_path):
        try:
            if str(file).endswith(('jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'jpeg', 'bmp', 'PNG')):
                key = str(file)
                rgb = main_color(file, pixel_x, pixel_y)
                lab = rgb2Lab(rgb)
                library_object[key] = lab
        except cv2.error:
            # print(
            #     'Problem occure while processing this file, abandoned : ' + str(
            #         file) + '\nplease avoid using Chinese character in the name of the file\n')
            pass
        continue
    end_time = time.time()
    # print('The conversion of pictures cost :' + str(end_time - start_time) + 's\n')
    return library_object


def transparent(file, dest, transparenc):
    """
    make a picture transparent in some degree
    :param file: the picture that is to be processed
    :param dest: the filename of the post processed picture
    :param transparency: how much transparent effect it is
    :return: Nonetype
    """
    img = Image.open(file)
    img = img.convert('RGBA')
    r, g, b, alpha = img.split()
    alpha = alpha.point(lambda i: i > 0 and transparenc)
    img.putalpha(alpha)
    img.save(dest)


def temp_library(main_directory, pixel_x, pixel_y, Library_path):
    """
    create an temporary library store all the resized pictures from library
    :param Libray_name: the library of pictures
    :return: Nonetype
    """
    if not os.path.exists(main_directory + '/temp_' + str(pixel_x) + '_' + str(pixel_y)):
        os.mkdir(main_directory + '/temp_' + str(pixel_x) + '_' + str(pixel_y))
    else:
        return
    for file in os.listdir(Library_path):
        os.chdir(Library_path)
        try:
            if str(file).endswith(('jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'jpeg', 'bmp', 'PNG')):
                # warnings.filterwarnings('ignore')
                pic = cv2.imread(file)  # 读取图片
                image = cv2.resize(pic, (pixel_x, pixel_y), Image.ANTIALIAS,
                                   interpolation=cv2.INTER_AREA)  # 将图片尺寸改成需要的大小
                os.chdir(main_directory)
                cv2.imwrite(
                    main_directory + '/temp_' + str(pixel_x) + '_' + str(pixel_y) + '/' + file,
                    image)
        except cv2.error:
            pass
            # print('Problem occurs while processing this image, try not use Chinese characters, skipped. : ' + str(
            #     file))
        continue


def getpwd():
    pwd = sys.path[0]
    if os.path.isfile(pwd):
        pwd = os.path.dirname(pwd)
    return pwd


def choose_tile_for_each_pixel(main_directory, picture_name, object_dic, library_dic, pixel_x, pixel_y, final_nameee,
                               transparenc, value):
    os.chdir(main_directory)
    base_image = 'resized_' + picture_name
    image_info = cv2.imread(base_image)
    height = image_info.shape[0]
    width = image_info.shape[1]
    num_x = width / pixel_x
    num_y = height / pixel_y
    base_copy = cv2.imread(base_image)
    cv2.imwrite(base_image[0:-4] + 'copy.jpg', base_copy)
    cv2.imwrite(base_image[0:-4] + 'copy2.jpg', base_copy)
    base = Image.open(base_image[0:-4] + 'copy.jpg')
    transparent(base_image[0:-4] + 'copy2.jpg', base_image[0:-4] + 'mask.png', transparenc)

    list1 = []
    first_match = {}
    use_times = dict.fromkeys(library_dic.keys(), 0)
    for keys in object_dic.keys():
        list1.append(int(keys[0: -4]))
    list1.sort()
    try:
        for pixel in list1:
            pixel = str(pixel) + '.bmp'
            deltaE_value = {}
            for tile in library_dic:  # search for every pic in library to find the best match one
                delta = deltaE76(object_dic.get(pixel), library_dic.get(tile))
                deltaE_value[tile] = delta
            deltaE_after_sort = sorted(deltaE_value.items(), key=lambda item: item[1])  # sort the deltaE dictionary
            match_tile = deltaE_after_sort[0][0]  # Identify the first min deltaE's filename
            use_times[match_tile] = use_times[match_tile] + 1  # pic counter
            choice = use_times[
                         match_tile] % 100  # if the picture has been used in last iteration, mark and use second small deltaE
            match_tile = deltaE_after_sort[choice - 1][0]
            use_times[match_tile] = use_times[match_tile] + 1
            first_match[pixel[0:-4]] = match_tile
    except KeyError:
        pass
    os.chdir(main_directory + '/temp_' + str(pixel_x) + '_' + str(pixel_y) + '/')
    i = 0
    j = 0
    for match in first_match:
        Image_png = Image.open(first_match[match])
        if i == 0 and j == 0:
            base.paste(Image_png, (i * pixel_x, j * pixel_y), )  # the coordinate of the left top point
        base.paste(Image_png, (i * pixel_x, j * pixel_y))  # the coordinate of the left top point
        j += 1
        if j == num_y:
            i += 1
            j = 0
    os.chdir(main_directory)
    mask = Image.open(base_image[0:-4] + 'mask.png')
    base.paste(mask, (0, 0), mask)
    base.save(main_directory + '/' + final_nameee, quality=100)
    if value:
        pwd = getpwd()
        os.chdir(pwd)
        sup = cv2.imread('supreme.png')
        sup_height = int(width * 130/(374 * 5))
        re_sup = cv2.resize(sup, (int(width/5), sup_height), Image.ANTIALIAS)
        cv2.imwrite('re_sup.png', re_sup)
        sup = Image.open('re_sup.png')
        os.chdir(main_directory)
        base2 = Image.open(main_directory + '/' + final_nameee)
        base2.paste(sup, (int(2 * width/5), int(height/2 - sup_height/2)))
        base2.save(main_directory + '/supreme_' + final_nameee, quality=100)
    os.remove(base_image[0:-4] + 'mask.png')
    os.remove(base_image[0:-4] + 'copy.jpg')
    os.remove(base_image[0:-4] + 'copy2.jpg')
    pwd = getpwd()
    os.chdir(pwd)
    os.remove('re_sup.png')

app = Tk()
app.title("千图成像 - Powered by love for xy")
fontStyle_welcome = tkFont.Font(family="fangsong", size=20)  # 设置标签字体
fontStyle_text = tkFont.Font(family="Lucida Grande", size=13)  # 设置标签字体
fontStyle_label = tkFont.Font(family="Lucida Grande", size=15)  # 设置标签字体
fontStyle_button = tkFont.Font(family="Lucida Grande", size=10)  # 设置标签字体
fontStyle_go = tkFont.Font(family="Lucida Grande", size=20)  # 设置标签字体
path_maindirectory = StringVar()
path_library = StringVar()
path_pic = StringVar()
final_name = StringVar()
CheckVar1 = IntVar()


def select_path_maindirectory():
    path_select = askdirectory()
    path_maindirectory.set(path_select)


def select_path_library():
    path_select = askdirectory()
    path_library.set(path_select)


def select_path_pic():
    file_path = filedialog.askopenfilename()
    path_pic.set(file_path)


maxValue = 100


class newwind(tk.Toplevel):
    def __init__(self, parent, message, directory, picture_name, pixel_x, pixel_y):
        super().__init__()
        self.title('info')
        self.geometry('500x300')
        self.message = message
        self.parent = parent  # 显式地保留父窗口
        self.directory = directory
        self.picture_name = picture_name
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        btClose = Button(self, text='完成', width=6, height=2, command=self.ok)
        # btClose.pack(side="bottom", pady=5, expand=True)
        btClose.place(relx=0.66, rely=0.9, anchor='center')

        btCancel = Button(self, text='取消', width=6, height=2,
                          command=self.cancel)
        btCancel.place(relx=0.33, rely=0.9, anchor='center')

    def ok(self):
        self.destroy()  # 销毁窗口

    def cancel(self):
        os.chdir(self.directory)
        if not os.path.exists('resized_' + self.picture_name[0:-4]):
            pass
        else:
            shutil.rmtree('resized_' + self.picture_name[0:-4])  # 递归删除文件夹
        if not os.path.exists(self.directory + '/temp_' + str(self.pixel_x) + '_' + str(self.pixel_y)):
            pass
        else:
            os.chdir(self.directory)
            shutil.rmtree(self.directory + '/temp_' + str(self.pixel_x) + '_' + str(self.pixel_y))
        self.destroy()


move_on = True


def check(main_directory, Object_filename, Library_path):
    if not os.path.exists(main_directory):
        isMove = False
        return isMove

    if not os.path.exists(Library_path):
        isMove = False
        return isMove

    if not os.path.exists(Object_filename):
        isMove = False
        return isMove

    isMove = True
    return isMove


def start(main_directory, Object_filename, Library_path, pixel_x, pixel_y, transparency, value):
    starttime = time.time()
    message = '检查中...'
    isMove = check(main_directory, Object_filename, Library_path)
    if isMove is False:
        res = tk.messagebox.showerror('错误', '出错了，请正确输入所有路径参数！')
        if res == 'ok':
            return
    else:
        picture_name = re.findall(r'[^\\/:*?"<>|\r\n]+$', Object_filename)[0]
        final_nameee = 'pixelove_' + re.findall(r'[^\\/:*?"<>|\r\n]+$', Object_filename)[0]
        currentValue = 0
        childwindow = newwind(app, message, main_directory, picture_name, pixel_x, pixel_y)
        progressbar = ttk.Progressbar(childwindow, orient="horizontal", length=300, mode="determinate")
        progressbar.place(relx=0.5, rely=0.1, anchor='center')
        progressbar["value"] = currentValue
        progressbar["maximum"] = maxValue
        label = Label(childwindow, bg='grey', font=fontStyle_button, text=message)
        label.place(relx=0.5, rely=0.5, anchor='center')
        app.wait_visibility(childwindow)
        message = message + '\n正在检查和微调原始图片尺寸...'
        label['text'] = message
        label.update()
        childwindow.update()
        check_resize(Object_filename, pixel_x, pixel_y, main_directory,
                     picture_name)  # check if the original picture need to be resized
        currentValue = 10
        progressbar["value"] = currentValue
        progressbar.update()
        message = message + '\n正在将原始图片分割成像素图片...'
        label['text'] = message
        label.update()
        childwindow.update()
        pixelizer(picture_name, pixel_x,
                  pixel_y)  # discrete the original image and store the pixel pictures get from the original image in dir(resized+filename of the picture)
        currentValue = 20
        progressbar["value"] = currentValue
        progressbar.update()
        message = message + '\n创建缓存文件中...'
        label['text'] = message
        label.update()
        childwindow.update()
        temp_library(main_directory, pixel_x, pixel_y,
                     Library_path)  # resize all images in Library to a directory called 'temp' for later use
        currentValue = 40
        progressbar["value"] = currentValue
        progressbar.update()
        path = main_directory + "/resized_"  # the path that the discreted pictures are stored
        message = message + '\n扫描分割后的像素图片，获取颜色信息中...'
        label['text'] = message
        label.update()
        childwindow.update()
        object_dic = create_pixels_color_library(
            path + picture_name[0:-4], pixel_x, pixel_y)  # get the color dictionary of original picture pixels
        currentValue = 60
        progressbar["value"] = currentValue
        progressbar.update()
        message = message + '\n扫描素材图片库，获取颜色信息中...'
        label['text'] = message
        label.update()
        childwindow.update()
        library_dic = create_pixels_color_library(main_directory + '/temp_' + str(pixel_x) + '_' + str(
            pixel_y), pixel_x, pixel_y)  # get the color dictionary of pictures in temp directory
        currentValue = 80
        progressbar["value"] = currentValue
        progressbar.update()
        message = message + '\n正在生成图片，请静候片刻...'
        label['text'] = message
        label.update()
        childwindow.update()
        choose_tile_for_each_pixel(main_directory, picture_name, object_dic, library_dic, pixel_x, pixel_y,
                                   final_nameee, transparency, value)
        currentValue = 100
        progressbar["value"] = currentValue
        progressbar.update()
        message = message + '\nBingo！ 您的千图成像作品已经在指定目录中\n文件名为 “pixelove_原始文件名”。'
        label['text'] = message
        label.update()
        childwindow.update()
        end = time.time()
        duration = end - starttime
        message = message + '\n总耗时：' + str(duration) + 's'
        label['text'] = message
        label.update()
        childwindow.update()
        childwindow.mainloop()

#
# def check_and_start(main_directory, Object_filename, Library_path, width, height, transparency, value):
#     start(main_directory, Object_filename, Library_path, width, height, transparency, value)
#

# GUI setting
Welcome = Label(app, text='相册里面的每一张照片，都有可能是你喜欢的的照片的一个像素。\n\n千图成像',
                bg='#E0E6E8', relief=RIDGE, font=fontStyle_welcome).pack(padx=10, pady=40, fill='x', expand=True)

Label(app, font=fontStyle_text, text='请选择一个空目录以存放缓存文件和最后的图像，建议新建一个空目录').pack(side='top', anchor='n', expand=True)
Entry(app, textvariable=path_maindirectory, width=50).pack(padx=5, pady=2, side='top', anchor='n', expand=True)
Button(app, text="主目录路径选择", font=fontStyle_button, command=select_path_maindirectory).pack(padx=5, pady=0, side='top',
                                                                                           anchor='n', expand=True)

Label(app, font=fontStyle_text, text='                             ').pack(side='top', anchor='n', pady=5, expand=True)

Label(app, font=fontStyle_text, text='请选择用于合成相片的素材图所在相册，至少包含100张图片').pack(side='top', anchor='n', expand=True)
Entry(app, textvariable=path_library, width=50).pack(padx=5, pady=2, side='top', anchor='n', expand=True)
Button(app, text="库路径选择", font=fontStyle_button, command=select_path_library).pack(padx=5, pady=0, side='top',
                                                                                   anchor='n', expand=True)

Label(app, font=fontStyle_text, text='                             ').pack(side='top', anchor='n', pady=5, expand=True)

Label(app, font=fontStyle_text, text='请选择您要合成的照片').pack(side='top', anchor='n', expand=True)
Entry(app, textvariable=path_pic, width=50).pack(padx=5, pady=2, side='top', anchor='n', expand=True)
Button(app, text="相片路径选择", font=fontStyle_button, command=select_path_pic).pack(padx=5, pady=0, side='top', anchor='n',
                                                                                expand=True)

Label(app, font=fontStyle_text, text='                             ').pack(side='top', anchor='n', pady=5, expand=True)

Label(app, font=fontStyle_text, text='选择原图的透明度以增强效果\n建议从70-90之间开始然后根据效果动态调整').pack(side='top', anchor='n', pady=5,
                                                                                   expand=True)
scale3 = tk.Scale(app, from_=0, to=240, orient='horizonta', tickinterval=30, length=300)
scale3.pack(side='top', anchor='center', expand=True)

Label(app, font=fontStyle_button, text='请选择每个像素照片的大小\n建议长度和宽度从50到80之间开始\n根据效果动态调整大小').pack(side='left', anchor='w',
                                                                                           expand=True, padx=20)
Label(app, font=fontStyle_text, text='像素长:').pack(side='left', anchor='w', expand=True)
scale1 = tk.Scale(app, from_=10, to=150, orient='horizonta', tickinterval=20, length=200)
scale1.pack(side='left', anchor='w', expand=True)
Label(app, font=fontStyle_text, text='像素宽:').pack(side='left', anchor='w', expand=True)
scale2 = tk.Scale(app, from_=10, to=150, orient='horizonta', tickinterval=20, length=200)
scale2.pack(side='left', anchor='w', expand=True)

Label(app, font=fontStyle_text, text='                             ').pack(side='top', anchor='n', pady=5, expand=True)

C1 = Checkbutton(app, text="上流？", font=fontStyle_text, height=2, width=10, variable=CheckVar1, onvalue=1, offvalue=0).place(relx=0.88, rely=0.8, anchor='center')

# start to run the program if pressed the 'gan' button
gan = Button(app, text="淦!", width=5, height=2, font=fontStyle_go,
             command=lambda: start(path_maindirectory.get(), path_pic.get(), path_library.get(), scale1.get(),
                                             scale2.get(), scale3.get(), CheckVar1.get())).pack(padx=5, pady=10,
                                                                               side='top', anchor='center', expand=True)

Label(app, font=fontStyle_text, text='                             ').pack(side='bottom', anchor='s', pady=5,
                                                                           expand=True)

mainloop()
