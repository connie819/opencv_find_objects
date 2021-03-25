from serial.tools import list_ports  # 從 serial.tool 中引進 list_ports(列表端口)
import cv2  # 引進 cv2(opencv-python)(讀取圖檔)
from pydobot import Dobot  # 從 pydobot 中引進 Dobot
import tkinter as tk  # 引進 tkinter 以 tk 表示(GUI使用者介面)
import threading  # 引進 threading
import time  # 引進 time
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

def job():
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

root = tk.Tk()
root.geometry('900x800')   # 介面大小
root.title("操作介面")    # 介面名稱
""" GUI Function set """
btn = []
btn_context = ['Home','開相機','開始']
port = list_ports.comports()[0].device
device = Dobot(port=port, verbose=False)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
detect_state=False  #檢測狀態
if(detect_state == False):
    target = [0, 0]

def Home():
    print('Home')
    device.move_to(40, -130, 10, r, wait=False)

def Detect():  # 開啟攝影機
    t1 = threading.Thread(target=CVJOB)
    t1.start()

def CVJOB():  # 視覺辨識
    global target,detect_state,im
    detect_state =True
    cap = cv2.VideoCapture(1)  # 建立一個VideoCapture物件，物件會連接到一隻網路攝影機，我們可以靠著它的參數來指定要使用那一隻攝影機（0代表第一隻、1代表第二隻）。
    itt = 0
    while(True):
        ret, frame = cap.read()   # 從攝影機擷取一張影像
        im = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        low_threshold = 1
        high_threshold = 10
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        contours, hierachy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cnt_count = []
        centerX = []
        centerY = []
        for cnt in range(len(contours)):
            epsilon = 0.04 * cv2.arcLength(contours[cnt], True)
            approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
            area = cv2.contourArea(contours[cnt])
            print(area)
            if (len(approx) < 5 and len(approx) > 3 and area > 2000 and area < 4000):
                cv2.drawContours(im, contours[cnt], -1, (255, 255, 255), 3)
                M = cv2.moments(contours[cnt])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cnt_count.append(cnt)
                    centerX.append(cX)
                    centerY.append(cY)
                    f1 = int(cY * 0.541 - 3.1564)
                    f2 = int(cX * 0.571 - 127)
                    target[0] = f1
                    target[1] = f2
                    text = str(f1) + ',' + str(f2)
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

def Move():    # 開始手臂移動
    global target
    t3 = threading.Thread(target=MoveJOB)
    t3.start()        # 執行該子執行緒
    print("Move")

points = np.array([[58, -187, -40],[178, -12, 10]])

def MoveJOB():
    global state
    t = threading.Thread(target=job)
    # 執行該子執行緒
    t.start()
    device.move_to(178, -12, -57, r, wait=True)
    device.suck(True)
    time.sleep(1)
    device.move_to(178, -12, 30, r, wait=True)
    device.suck(True)
    device.move_to(58, -187, -40, r, wait=True)
    device.suck(False)
    device.move_to(58, -187, 30, r, wait=True)
    state = False

"""把function 放進List"""
fun_list = [Home,Detect,Move]
for j in range(3):
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