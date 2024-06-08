import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
from ultralytics import YOLO
import numpy as np

def load_model():
    model_path = filedialog.askopenfilename()
    if model_path:
        global model
        model = YOLO(model_path)
        console.insert(tk.END, "模型已成功加载！\n")

def select_image():
    global panelA
    path = filedialog.askopenfilename()

    if len(path) > 0:
        if model:
            results = model([path])  # 使用文件路径而非图像数据
            print("检测到的目标数量:", len(results[0].boxes))  # 检测目标的数量

            if len(results[0].boxes) > 0:
                for result in results:
                    # 使用plot方法绘制检测框和标签到图像上
                    image = result.plot()
            else:
                print("没有检测到任何目标。")
                image = cv2.imread(path)  # 读取原始图片以显示未修改的图片

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
            image = Image.fromarray(image)
            image = ImageOps.fit(image, (650, 490), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(image)

            if panelA is None:
                panelA = tk.Label(image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)
            else:
                panelA.configure(image=image)
                panelA.image = image


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("视频流错误", "无法读取视频流")
            break

        results = model([frame])
        for result in results:
            frame = result.plot()

        cv2.imshow('YOLOv8 Real-time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def exit_program():
    root.quit()

root = tk.Tk()
root.title("路标检测系统")
root.geometry("900x600")  # 设置窗口初始大小

panelA = None
model = None

# 控制台部分
console_frame = tk.Frame(root, height=100)
console_frame.pack(fill=tk.X, side=tk.BOTTOM)

console = tk.Text(console_frame, height=5)
console.pack(side=tk.LEFT, fill=tk.X, expand=True)

# 按钮部分
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.RIGHT)

btn_load_model = tk.Button(button_frame, text="加载模型", command=load_model)
btn_load_model.pack(fill=tk.X)

btn_select_image = tk.Button(button_frame, text="选择图片", command=select_image)
btn_select_image.pack(fill=tk.X)

btn_open_camera = tk.Button(button_frame, text="打开摄像头", command=open_camera)
btn_open_camera.pack(fill=tk.X)

btn_exit = tk.Button(button_frame, text="退出程序", command=exit_program)
btn_exit.pack(fill=tk.X)

root.mainloop()
