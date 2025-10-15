#########导入需要的库#########
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import mss
import win32con,win32api,win32gui


#########检查GPU是否能够运行#########
device = 'cuda' if torch.cuda.is_available() else 'cpu'         #设备是否是GPU否则是CPU
print(f'Device: {device}')
if device == 'cuda':
    print(f"GPU:{torch.cuda.get_device_name()}")                #如果是GPU则显示你的显卡

#########！！开始！！#########

#########先截图#########
sct = mss.mss()                                                                                                         #截图工具
screen_width, screen_height = 1920,1080                                                                                  #你的cs分辨率
WINDOW_LEFT,WINDOW_TOP,WINDOW_WIDTH,WINDOW_HEIGHT = screen_width//3,screen_height//3,screen_width//3,screen_height//3   #方便显示标记，全屏太大了

monitor = {                                 #建个字典存一下变量
    'left': WINDOW_LEFT,
    'top': WINDOW_TOP,
    'width': WINDOW_WIDTH,
    'height': WINDOW_HEIGHT
}

windows_name = 'screen'                     #定义你的标注的屏幕名字和尺寸
model_size = 640
target_class = 0

model =YOLO('cs16.pt').to(device)             #加载你训练好的模型，我这里改了名字


##########写了个函数用来处理图片#########
def preprocess_image(img_np):                #截图后将你的NUMPY数组转化为torch需要的张量，最后再转化成浮点类型，还要把他放在你的设备上运行CPU/GPU
    img_tensor = torch.from_numpy(img_np).float().to(device)            #HWC  0,1,2   我们需要把他变成Torch用的东西          HWC->CHW->BCHW
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0) /255.0          #CHW  2,0,1   /255是归一化处理
    img_tensor = torch.nn.functional.interpolate(img_tensor,size=(model_size,model_size),mode='bilinear',align_corners=False)
    return img_tensor


#########写个循环#########
while True:
    img = np.array(sct.grab(monitor))[:,:,:3]

    try:
        #预处理图像
        img_tensor = preprocess_image(img)

        #使用GPU推理
        with torch.no_grad():
            result = model(img_tensor)

        #获取结果并标注
        result = result[0]


        #########实现锁人#########
        closet_target = None
        min_distance = float('inf')
        for box in result.boxes:
            if box.cls[0] == target_class and box.conf.item()>0.3:
                x1,y1,x2,y2 = box.xyxy[0]
                scale_x = WINDOW_WIDTH / model_size
                scale_y = WINDOW_HEIGHT / model_size
                x1 = x1 * scale_x
                x2 = x2 * scale_x
                y1 = y1 * scale_y
                y2 = y2 * scale_y
                #相对坐标
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                #转换成绝对坐标
                screen_x = int(WINDOW_LEFT+center_x)
                screen_y = int(WINDOW_TOP+center_y)
                #计算与屏幕中心的距离
                distance = (center_x - WINDOW_WIDTH/2)**2 + (center_y - WINDOW_WIDTH/2)**2
                #选择最近的敌人锁
                if distance < min_distance:
                    min_distance = distance
                    closet_target = (screen_x, screen_y)

                #移动鼠标到敌人身上
            if closet_target:
                win32api.SetCursorPos(closet_target)
                #########实现锁人#########
        annotated_frame = result.plot()

        #显示结果
        cv2.imshow(windows_name,annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit('退出')

        #窗口置顶
        hwnd = win32gui.FindWindow(None,windows_name)
        win32gui.ShowWindow(hwnd,win32con.SW_NORMAL)
        win32gui.SetWindowPos(
            hwnd,win32con.HWND_TOPMOST,0,0,0,0,
            win32con.SWP_NOMOVE|win32con.SWP_NOACTIVATE|win32con.SWP_NOSIZE|win32con.SWP_SHOWWINDOW
        )


    except Exception as e:
        print("有错误：{e}")
        break


