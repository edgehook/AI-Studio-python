
import os
import time
import torch
import pandas as pd
import cv2
from deepface import DeepFace

def face_attendance(address):
    # 打开摄像头
    cap = cv2.VideoCapture(address)
    # cap = cv2.VideoCapture(address)
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # while True:
    #     ret, frame = cap.read()

    #     # 将图像转换为灰度图像
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # 检测人脸
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #     for (x, y, w, h) in faces:
    #         # 绘制矩形框表示检测到的人脸
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #         # 提取人脸区域
    #         face = frame[y:y + h, x:x + w]
    #         filename = "demo.jpeg"
    #         result = cv2.imwrite(filename, face)
    #         if result:
    #             print(f"图片 {filename} 保存成功")
    #         else:
    #             print(f"图片 {filename} 保存失败")

    #         # 使用 DeepFace 进行人脸识别（身份验证）
    #         dfs = DeepFace.find(
    #                             img_path=frame, 
    #                             db_path='/home/gangqiangsun/Downloads/deepface',  
    #                             threshold= 0.3,
    #                             model_name="Facenet",
    #                             # detector_backend="retinaface",
    #                             enforce_detection=False
    #                         )
    #         print(dfs)
    #         # 获取识别结果，显示识别的身份信息
    #         for df in dfs:
    #             if not df.empty:
    #                 # 获取匹配到的人脸的文件路径
    #                 matched_paths = df['identity'].tolist()
    #                 # 从文件路径中提取名字
    #                 names = []
    #                 for path in matched_paths:
    #                     # 获取包含图像文件的文件夹名称，即人员名字
    #                     # name = path.split('\\')[-2]  # 对于 Windows 系统
    #                     # 如果是 Linux 或 macOS 系统，使用以下代码
    #                     name = path.split('/')[-2]
    #                     names.append(name)
    #                 print("匹配到的名字:", names)
    #             else:
    #                 print("未匹配到任何人脸")

    #         # 按 'q' 键退出
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    # 释放摄像头资源
    # cap.release()
    # cv2.destroyAllWindows()



    # ////////////////////////////////////////////
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面，请检查摄像头连接。")
            break
        # gray_source_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model(frame)
        detections = results.pandas().xyxy[0]
        for _, detection in detections.iterrows():
            if detection['name'] == 'person':
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                face = frame[y1:y2, x1:x2]
                filename = "demo.jpeg"
                result = cv2.imwrite(filename, face)
                if result:
                    print(f"图片 {filename} 保存成功")
                    if face.size > 0:
                        try:
                            # 使用 DeepFace 进行人脸识别
                            
                            dfs = DeepFace.find(
                                img_path=filename, 
                                db_path='/home/gangqiangsun/Downloads/deepface',  
                                threshold= 0.3,
                                model_name="Facenet",
                                # detector_backend="retinaface",
                                enforce_detection=False
                            )
                            print(dfs)
                            for df in dfs:
                                if not df.empty:
                                    # 获取匹配到的人脸的文件路径
                                    matched_paths = df['identity'].tolist()
                                    # 从文件路径中提取名字
                                    names = []
                                    for path in matched_paths:
                                        # 获取包含图像文件的文件夹名称，即人员名字
                                        # name = path.split('\\')[-2]  # 对于 Windows 系统
                                        # 如果是 Linux 或 macOS 系统，使用以下代码
                                        name = path.split('/')[-2]
                                        names.append(name)
                                    print("匹配到的名字:", names)
                                else:
                                    print("未匹配到任何人脸")
                        except Exception as e:
                            print(f"识别过程中出现错误：{e}")
                            # cv2.putText(frame, "识别出错", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # cv2.imshow('人脸打卡系统', frame)
                        # 按 'q' 键退出打卡系统
                else:
                    print(f"图片 {filename} 保存失败")

                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # run()



    
def run():
    # result = DeepFace.find(img_path="/home/gangqiangsun/Downloads/aibama3.jpeg", db_path="/home/gangqiangsun/Downloads/aobama", model_name='VGG-Face', enforce_detection=False)
    dfs = DeepFace.find(
        img_path = "/home/gangqiangsun/MYPROJECT/PYTHON/ai-studio/studio/demo.jpeg",
        db_path = "/home/gangqiangsun/Downloads/deepface",
        threshold= 0.3,
        model_name="Facenet",
        detector_backend="retinaface"
    )
    for df in dfs:
        if not df.empty:
            # 获取匹配到的人脸的文件路径
            matched_paths = df['identity'].tolist()
            # 从文件路径中提取名字
            names = []
            for path in matched_paths:
                # 获取包含图像文件的文件夹名称，即人员名字
                # name = path.split('\\')[-2]  # 对于 Windows 系统
                # 如果是 Linux 或 macOS 系统，使用以下代码
                name = path.split('/')[-2]
                names.append(name)
            print("匹配到的名字:", names)
        else:
            print("未匹配到任何人脸")

    # objs = DeepFace.analyze(
    #     img_path = "/home/gangqiangsun/Downloads/syy01.jpeg", 
    #     actions = ['age', 'gender', 'emotion'],
    #     detector_backend="retinaface"
    # )
    print("run test")