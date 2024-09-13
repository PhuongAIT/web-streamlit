def run_app1():  
    import cv2
    import streamlit as st
    import numpy as np
    import tensorflow as tf
    from mtcnn import MTCNN 
    from PIL import Image
    import pickle
    from keras.utils import img_to_array
    # import sys
    # sys.path.append('web-streamlit/Diemdanh')
    from Diemdanh import facenet_architecture
    #from facenet_architecture import InceptionResNetV2
    from Diemdanh import create_embedding
    #from create_embedding import get_embedding
    import tempfile
    

    name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
   

    # MTCNN detector
    detector = MTCNN()

    # Biến cờ cho việc dừng webcam và video
    # webcam_active = False
    # video_active = False

    # Tải model và classifier - chỉ thực hiện một lần
    @st.cache_resource
    def load_model_and_classifier():
        facenet = facenet_architecture.InceptionResNetV2()
        path = 'Diemdanh/model/facenet_keras.h5'
        facenet.load_weights(path)
        file_name = 'Diemdanh/model/classify.sav'
        loaded_model = pickle.load(open(file_name, "rb"))
        return facenet, loaded_model

    facenet, loaded_model = load_model_and_classifier()

    # Hàm xử lý từng frame của video/webcam hoặc ảnh
    def process_frame(frame):
        result = detector.detect_faces(frame)
        if len(result) == 0:
            return frame, None, None
        
        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = frame[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        resized_arr = np.asarray(image)
        
        # Tính toán embedding vector
        embed_vector = create_embedding.get_embedding(facenet, resized_arr)
        
        return frame, embed_vector, (x1, y1, x2, y2)

    # Dự đoán class của khuôn mặt
    def predict(model, embed_vector):
        sample = np.expand_dims(embed_vector, axis=0)
        yhat_index = model.predict(sample)
        yhat_prob = np.max(model.predict_proba(sample)[0])
        class_predict = name_list[yhat_index[0]]
        return yhat_prob, class_predict

    # Hàm nhận diện từ webcam
    def recognize_from_webcam():
        global webcam_active
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while webcam_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, embed_vector, box = process_frame(frame_rgb)
            
            if embed_vector is not None:
                prob, pred_class = predict(loaded_model, embed_vector)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame = Image.fromarray(frame)
            stframe.image(frame, caption="Video Stream", use_column_width=True)
        
        cap.release()

    # Hàm nhận diện từ video
    def recognize_from_video(video_path):
        global video_active
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        
        while video_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, embed_vector, box = process_frame(frame_rgb)
            
            if embed_vector is not None:
                prob, pred_class = predict(loaded_model, embed_vector)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame = Image.fromarray(frame)
            stframe.image(frame, caption="Video Stream", use_column_width=True)
        
        cap.release()

    # Hàm nhận diện từ ảnh
    def recognize_from_image(uploaded_file):
        img = Image.open(uploaded_file)
        img_arr = np.array(img)
        
        frame, embed_vector, box = process_frame(img_arr)
        
        if embed_vector is not None:
            prob, pred_class = predict(loaded_model, embed_vector)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        st.image(frame, caption="Processed Image", channels="RGB")

    # Streamlit App
    # def main():
    global webcam_active, video_active
    
    #Sidebar
    #st.sidebar.header("Let's check attendance with us!!")
    st.sidebar.header("WELCOME TO OUR RESEARCH!")
    # Load your image
    image = open('Diemdanh/background/School-Attendance-Automation_1.png', "rb").read()  # Replace "class.jpg" with your image file path

    st.sidebar.header("Let's check attendance with us!!")
    # Display the image in the sidebar below the header
    st.sidebar.image(image, use_column_width=True)

    st.title("Face Recognition From Image, Video, and Webcam with MTCNN and FaceNet")
    st.radio("Selected Task: ",['Attendance'])
    
    option = st.sidebar.radio("Choose Function", ["Image", "Webcam", "Video", "Automatic Attendance Excel"])
    
    if option == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            recognize_from_image(uploaded_file)
    
    elif option == "Webcam":
        if st.button("Start Webcam"):
            webcam_active = True
            recognize_from_webcam()
        
        if st.button("Stop Webcam"):
            webcam_active = False
    
    elif option == "Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
        if video_file is not None:
            temp_video_path = f"temp_video.{video_file.type.split('/')[-1]}"
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())
            
            if st.button("Start Video"):
                video_active = True
                recognize_from_video(temp_video_path)
            
            if st.button("Stop Video"):
                video_active = False
    elif option == "Automatic Attendance Excel":
        st.image('Diemdanh/Exceldiemdanh.png', caption="Automatic check attendance into Excel file")
    else:
        st.error("Please select a valid choice!")
    








































