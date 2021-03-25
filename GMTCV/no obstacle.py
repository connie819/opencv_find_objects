from serial.tools import list_ports
import cv2
from pydobot import Dobot
import tkinter as tk
import random as rd
from matplotlib import cm
from PIL import Image, ImageTk
from tkinter import *
import threading
from math import factorial
import time
import matplotlib.pyplot as plt
import numpy as np
list_x = []
list_y = []
list_z = []
list_r = []
list_j1 = []
list_j2 = []
list_j3 = []
list_j4 = []
state = True
itt=0
要動了=False
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum( comb(n, i) * t**i * (1 - t)**(n - i) * points[i] for i in range(n + 1) )
def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:,0], new_points[:,1],new_points[:,2]
def job():
    global state
    global list_z, list_x, list_y
    while(state):
        #print('do')
        (a,b,c,d,e,f,g,h)=device.pose()
        list_x.append(a)
        list_y.append(b)
        list_z.append(c)
        list_r.append(d)
        time.sleep(0.01)
from tkinter import filedialog
root  = tk.Tk()
root.geometry('900x800')
root.title("操作介面")
""" GUI Function set """
btn = []
btn_context = ['Home','開相機','拍照','Previous','存檔','開始','看路徑']
port = list_ports.comports()[0].device       # COM端口  dobot
device = Dobot(port=port, verbose=False)    # dobot
(x, y, z, r, j1, j2, j3, j4) = device.pose()    # dobot
detect_state=False
if(detect_state==False):
    target=[0,0]
def Home():
    print('Home')
    device.move_to(j1, j2, j3, j4, wait=False)
    var = []
    for j in range(6):
        var.append(ent[j].get())
    print(var)
def Detect():
    t1 = threading.Thread(target=CVJOB)
    t1.start()
def CVJOB():
    global target,detect_state,im,要動了
    detect_state =True
    '''
    def get_gray(img):  # 灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    def get_blurred(img):  # 模糊化
        gray = get_gray(img)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        return blurred
    def get_binary(img):  # 黑白化
        blur = get_blurred(img)
        ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

        return binary

    def get_contours(img):  # 獲取輪廓
        binary = get_binary(img)
        contours, hierachy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return contours
    def draw_contours(img):  # 畫輪廓
        contours = get_contours(img)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        return None
    def all_contour_X_Y(img):  # 要返回所有輪廓的X，Y值
        g_img = get_gray(img)
        binary_img = get_binary(g_img)
        cnt = get_contours(binary_img)
        draw_contours(img, cnt, -1(0, 0, 255), 3)
        return img
    '''
    cap = cv2.VideoCapture(1)
    itt = 0
    while (True):
        #print(target)
        # im = cv2.imread('2object.JPG')
        ret, frame = cap.read()
        im = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        low_threshold = 1
        high_threshold = 10
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        #ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cnt_count = []
        centerX = []
        centerY = []
        #have_countor = 0
        if(要動了==False):
            for cnt in range(len(contours)):
                epsilon = 0.04 * cv2.arcLength(contours[cnt], True)
                approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
                area = cv2.contourArea(contours[cnt])
                if (len(approx) < 5 and len(approx) > 3 and area > 2000 and area < 4000):
                    cv2.drawContours(im, contours[cnt], -1, (255, 255, 255), 3)
                    M = cv2.moments(contours[cnt])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cnt_count.append(cnt)
                        centerX.append(cX)
                        centerY.append(cY)
                        f1 = int(cY * 0.541 -3)
                        f2 = int(cX * 0.571 - 130)
                        target[0]=f1
                        target[1]=f2
                        #print('已更改')
                        text = str( f1) + ',' + str( f2 )
                        cv2.putText(im, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(im, (cX, cY), 10, (1, 227, 254), -1)
        cv2.imshow('123', im)
        cv2.imshow('bin', edges)
        left_point = []  # define for save 4 most point
        right_point = []
        top_point = []
        bottom_point = []
        ##############################################

        mRB = []
        mLB = []
        for Num in range(int(len(cnt_count))):
            cnt = contours[cnt_count[Num]]
            # print(len(cnt_count))
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
            ## get 4 most point
            cv2.circle(im, leftmost, 10, [0, 90, 255], -1)
            cv2.circle(im, topmost, 10, [0, 90, 255], -1)
            cv2.circle(im, rightmost, 10, [0, 90, 255], -1)
            cv2.circle(im, bottommost, 10, [0, 90, 255], -1)
            ## draw 4 most point
            left_point.append(leftmost)
            right_point.append(rightmost)
            top_point.append(topmost)
            bottom_point.append(bottommost)
            ## tuple type change to list type
            npleft = np.array(left_point)
            npright = np.array(right_point)
            # nptop = np.array(top_point)
            npbottom = np.array(bottom_point)
            ## change list to np.array
            leftX = list(npleft[:, 0])
            rightX = list(npright[:, 0])
            # topX = list(nptop [ : ,  0 ] )
            bottomX = list(npbottom[:, 0])
            leftY = list(npleft[:, 1])
            rightY = list(npright[:, 1])
            # topY=list(nptop [ : ,  1 ] )
            bottomY = list(npbottom[:, 1])
            mRB.append((bottomY[Num] - rightY[Num]) / (bottomX[Num] - rightX[Num]))
            mLB.append((bottomY[Num] - leftY[Num]) / (bottomX[Num] - leftX[Num]))
            X_position = leftY[0] / (-1 * mRB[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):  # 按s存檔
            itt = itt + 1;

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print('Detect')
def Next():
    global itt, im
    itt +=1
    #print('Next')
    cv2.imwrite('data\photo' + str(itt) + '.png', im)
    print('儲存:', 'photo' + str(itt) + '.png')
def Previous():
    print("Previous")
def Sucker():
    print("Sucker")
    with open('path_data'+str(itt)+'.txt', 'w') as filehandle:
        for place in range(len(list_x)):
            filehandle.writelines("%s," % int(list_x[place]) + "%s," % int(list_y[place])+"%s\n" % int(list_z[place]))

def Move():
    global target
    t3 = threading.Thread(target=MoveJOB)
    # 執行該子執行緒
    t3.start()
    print("Move")

def MoveJOB():
    global state,要動了,target
    要動了 = True
    t = threading.Thread(target=job)
    # 執行該子執行緒
    t.start()
    points = np.array([[58, -187, -40],
                       [168, -166, 30],  # 無障礙物1
                       [target[0], target[1], 10]])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    print(x,y,z)
    xb, yb, zb = evaluate_bezier(points, 50)
    print(xb[49], yb[49], zb[49])

    for st in range(0,len(xb),2):
        device.move_to(xb[st], yb[st], zb[st]+10, r, wait=False)
    device.move_to(xb[49], yb[49], -63, r, wait=True)
    device.suck(True)
    time.sleep(1)
    device.move_to(xb[49], yb[49], 10, r, wait=True)
    time.sleep(1)
    for st in range(len(xb)-1,0,-2):
        device.move_to(xb[st], yb[st], zb[st]+10, r, wait=False)
    device.move_to(58, -187, -60, r, wait=True)
    device.suck(False)
    device.move_to(58, -187, 30, r, wait=True)
    state = False
    要動了 = False

def Place():
    global list_z,list_x,list_y,itt
    ##動作完畢 開始繪圖
    # 建立 3D 圖形
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-63,200)
    ax.set_xlim(-50*0.6,450*0.6)
    ax.set_ylim(-350*0.6,150*0.6)
    # 產生 3D 座標資料
    z_array = np.array(list_z)
    x_array = np.array(list_x)
    y_array = np.array(list_y)
    # 繪製 3D 曲線
    ax.auto_scale_xyz([0, 500], [0, 500], [0, 500])
    ax.plot(x_array, y_array, z_array, color='gray', label='Arm path')
    asur = ax.scatter(x_array, y_array, z_array, c=z_array, cmap=cm.gist_rainbow, label='via point')
    fig.colorbar(asur, shrink=0.5, aspect=5)
    # 顯示圖例
    ax.legend()
    print(list_x)
    print("**************************")
    print(list_y)
    print("**************************")
    print(list_z)
    print("**************************")

    # 顯示圖形
    plt.show()
    plt.savefig('data\mat'+str(itt)+'.png')  # 儲存圖片
    list_x = []
    list_y = []
    list_z = []
"""把function 放進List"""
fun_list = [Home,Detect,Next,Previous,Sucker,Move,Place]
for j in range(7):
    btn.append(tk.Button(root,text = btn_context[j],
                         command = fun_list[j],
                         width = 12,height = 3,
                         font=('microsoft yahei', 10, 'bold')))
    btn[j].place(x=j*120+20,y=500)
"""處理button 放置"""
"""ABC三點xy共6個entry"""
ent = []

for j in range(6):
    ent.append(tk.Entry(root, show=None,width = 12))
    if(j%2==0):
        ent[j].place(x=50,y=j*30+20)
        print("j=",j,"%2=0")
    else:
        print("j=", j, "%2!=0")
        ent[j].place(x=190,y=(j-1)*30+20)

"""放label"""
lb = []
計數器=1
for j in range(6):
    if(j%2==0):
        lb.append(tk.Label(root, text=str(target[0]), bg='green', font=('Arial', 12), width=5,
                           height=1))
        lb[j].place(x=0,y=j*30+20)
    else:
        lb.append(tk.Label(root, text=str(target[1]), bg='green', font=('Arial', 12), width=5,
                           height=1))
        lb[j].place(x=140,y=(j-1)*30+20)
        計數器+=1
計數器=0


root.mainloop()



device.close()