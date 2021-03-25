# 若要設置自動歸零，則先運行一次手臂依照攝影機座標位置轉動到物件位置但不進行吸取的動作，若手臂未能移動到物件中心位置，
# 則利用手動將物件中心移動到手臂位置，再將手臂轉動移開，利用攝影機將物件位置再次進行視覺辨識出座標，兩者相差大小利用算式去進行運算
from serial.tools import list_ports  # 從 serial.tool 中引進 list_ports(列表端口)
import cv2  # 引進 cv2(opencv-python)(讀取圖檔)
from pydobot import Dobot  # 從 pydobot 中引進 Dobot
import tkinter as tk  # 引進 tkinter 以 tk 表示(GUI使用者介面)
from matplotlib import cm  # 從 matplotlib 中引進 cm(繪圖工具)
import threading  # 引進 threading (當運作子程式時，主程式依舊執行)
from math import factorial  # 從 math 中引進 factorial
import time  # 引進 time
import matplotlib.pyplot as plt  # 引進 matplotlib.pyplot 以 plt 表示
import numpy as np  # 引進 numpy 以 np表示(陣列)
list_x = []
list_y = []
list_z = []
list_r = []
list_j1 = []
list_j2 = []
list_j3 = []
list_j4 = []
state = True   # 宣告state = 真
itt = 0
def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))
def get_bezier_curve(points):  # 得到貝塞爾曲線
    n = len(points) - 1
    return lambda t: sum( comb(n, i) * t**i * (1 - t)**(n - i) * points[i] for i in range(n + 1) )
def evaluate_bezier(points, total): # 評估貝塞爾曲線
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:,0], new_points[:,1],new_points[:,2]
def job(): # 操作介面
    global state             # 全域變數global
    global list_z, list_x, list_y   # 全域變數list_z, list_x, list_y
    while(state):  # 當state=true
        print('do')
        (a,b,c,d,e,f,g,h)=device.pose()
        list_x.append(a)  # 加入a元素進入list_x
        list_y.append(b)  # 加入b元素進入list_y
        list_z.append(c)  # 加入c元素進入list_z
        list_r.append(d)  # 加入d元素進入list_r
        time.sleep(0.01)  # 暫停執行0.01秒

from tkinter import filedialog
root  = tk.Tk()
root.geometry('900x800')   # 介面大小
root.title("操作介面")    # 介面名稱
""" GUI Function set """
btn = []
btn_context = ['Home','開相機','拍照','Previous','存檔','開始','看路徑']
port = list_ports.comports()[0].device
device = Dobot(port=port, verbose=False)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
detect_state=False  #檢測狀態
if(detect_state==False):
    target=[0,0]

def Home():
    print('Home')
    # PTPMode.MOVJ_ANGLE
    device.move_to(j1, j2, j3, j4, wait=False)
    # device.PTPMode.MOVJ_ANGLE(j1+10, j2, j3, j4, wait=True)
    var = []
    for j in range(6):
        var.append(ent[j].get())
    print(var)
def Detect():  # 開啟攝影機
    t1 = threading.Thread(target=CVJOB)
    t1.start()
def CVJOB():  # 視覺辨識
    global target,detect_state,im
    detect_state =True
    # import argparse
    # cv2.imshow('123',binary)
    cap = cv2.VideoCapture(1)    # 建立一個 VideoCapture 物件，物件會連接到一隻網路攝影機，我們可以靠著它的參數來指定要使用那一隻攝影機（0 代表第一隻、1 代表第二隻）。
    itt = 0
    # cv2.waitKey()
    while (True):
        # im = cv2.imread('2object.JPG')
        ret, frame = cap.read()   # 從攝影機擷取一張影像
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
        # print(bool(contours))
        have_countor = 0
        for cnt in range(len(contours)):
            epsilon = 0.04 * cv2.arcLength(contours[cnt], True)
            approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
            # print(len(approx))
            area = cv2.contourArea(contours[cnt])

            if (len(approx) < 5 and len(approx) > 3 and area > 2000 and area < 4000):
                # print(area)
                cv2.drawContours(im, contours[cnt], -1, (255, 255, 255), 3)
                M = cv2.moments(contours[cnt])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])

                    cY = int(M["m01"] / M["m00"])
                    # print(area)
                    cnt_count.append(cnt)
                    centerX.append(cX)
                    centerY.append(cY)
                    # cnt_count = cnt_count + 1
                    f1 = int(cY * 0.541 - 3.1564)
                    f2 = int(cX * 0.571 - 127)
                    target[0]=f1
                    target[1]=f2
                    text = str( f1) + ',' + str( f2 )
                    cv2.putText(im, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(im, (cX, cY), 10, (1, 227, 254), -1)

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

        """
        print(centerX)
        print(centerY)

        print("X=", leftY[0] / (-1 * mRB[0]))
        print('left point ', left_point)
        print('bottom point ', bottom_point)
        print('right point ', right_point)
        print('mRb', mRB)  # 把numpy轉換成list
        print('mLB', mLB)
        """
        cv2.imshow('123', im)
        cv2.imshow('bin', edges)
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
def Sucker():    # 儲存檔案txt檔
    print("Sucker")
    with open('path_data'+str(itt)+'.txt', 'w') as filehandle:
        for place in range(len(list_x)):
            filehandle.writelines("%s," % int(list_x[place]) + "%s," % int(list_y[place])+"%s\n" % int(list_z[place]))

def Move():   # 開始手臂移動
    global target
    t3 = threading.Thread(target=MoveJOB)

    # 執行該子執行緒
    t3.start()
    print("Move")

points= np.array([[58, -187, -40],
                           [156,-38,30],  # 障礙物1
                           # [230,70,20],  # 障礙物2
                           [178,-12,10]])
def MoveJOB():
    global state
    t = threading.Thread(target=job)
    # 執行該子執行緒
    t.start()


    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xb, yb, zb = evaluate_bezier(points, 50)
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




def Place():     # 路徑圖
    global list_z,list_x,list_y,itt
    ## 動作完畢 開始繪圖
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
# 畫障礙物
    '''
    us = np.linspace(0, 2 * np.pi, 50)
    zs = np.linspace(-63, 74-63, 2)

    us, zs = np.meshgrid(us, zs)

    xs = 30 * np.cos(us)+points[1][0]
    ys = 30 * np.sin(us)+points[1][1]
    ax.plot_surface(xs, ys, zs, color='b')
    us1 = np.linspace(0, 2 * np.pi, 50)
    zs1 = np.linspace(-63, 50-63, 2)

    us1, zs1 = np.meshgrid(us1, zs1)

    xs1 = 40 * np.cos(us1)+points[2][0]
    ys1 = 40 * np.sin(us1)+points[2][1]
    ax.plot_surface(xs1, ys1, zs1, color='b')
    '''
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