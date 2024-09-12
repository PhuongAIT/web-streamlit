# def run_app1():
#     from unicodedata import name
#     from tensorflow import keras
#     from keras.preprocessing import image
#     from PIL import Image
#     import numpy as np
#     import base64
#     import warnings
#     warnings.filterwarnings('ignore')
#     from keras.utils import img_to_array, load_img
#     from MTCNN.preprocessing import *
def run_app1():  
    import cv2
    import streamlit as st
    import numpy as np
    import tensorflow as tf
    from mtcnn import MTCNN 
    from PIL import Image
    import pickle
    from keras.utils import img_to_array
    import sys
    sys.path.append('web-streamlit/Diemdanh')
    from Diemdanh import facenet_architecture
    #from facenet_architecture import InceptionResNetV2
    from Diemdanh import create_embedding
    #from create_embedding import get_embedding
    import tempfile
    

    # Load model và classifier
    # facenet = InceptionResNetV2()
    # path = "web-streamlit/Diemdanh/model/facenet_keras_weights.h5"
    # facenet.load_weights(path)

    name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
    # file_name = "web-streamlit/Diemdanh/model/classify.sav"
    # loaded_model = pickle.load(open(file_name, "rb"))

    # MTCNN detector
    detector = MTCNN()

    # Biến cờ cho việc dừng webcam và video
    # webcam_active = False
    # video_active = False

    # Tải model và classifier - chỉ thực hiện một lần
    @st.cache_resource
    def load_model_and_classifier():
        facenet = facenet_architecture.InceptionResNetV2()
        path = "web-streamlit/Diemdanh/model/facenet_keras_weights.h5"
        facenet.load_weights(path)
        file_name = "web-streamlit/Diemdanh/model/classify.sav"
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
    image = open('web-streamlit/Diemdanh/background/School-Attendance-Automation_1.png', "rb").read()  # Replace "class.jpg" with your image file path

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
        st.image("web-streamlit/Diemdanh/Exceldiemdanh.png", caption="Automatic check attendance into Excel file")
    else:
        st.error("Please select a valid choice!")
    

    # if __name__ == '__main__':
    #     main()
    





















# def run_app1():
#     import cv2
#     import streamlit as st
#     import numpy as np
#     import tensorflow as tf
#     from mtcnn import MTCNN
#     from PIL import Image
#     import pickle
#     from keras.utils import img_to_array
#     import sys
#     sys.path.append('D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh')
#     #from Diemdanh 
#     import facenet_architecture
#     #from Diemdanh 
#     import create_embedding
#     import tempfile

#     # Inserting an icon before the title
#     st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)
#     st.markdown(
#         '<div style="display: flex; align-items: center;">'
#         '<h1 style="margin: 0;">Face Recognition with MTCNN and FaceNet</h1>'
#         '<i class="fas fa-user-circle" style="margin-right: 10px;"></i>'
#         '</div>',
#         unsafe_allow_html=True
#     )

#     # Load model và classifier
#     @st.cache_resource
#     def load_model_and_classifier():
#         st.write("Loading model and classifier...")
#         try:
#             facenet = facenet_architecture.InceptionResNetV2()
#             path = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/facenet_keras_weights.h5"
#             facenet.load_weights(path)
#             file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/classify.sav"
#             loaded_model = pickle.load(open(file_name, "rb"))
#             st.write("Model and classifier loaded successfully.")
#             return facenet, loaded_model
#         except Exception as e:
#             st.error(f"Error loading model or classifier: {e}")
#             return None, None

#     facenet, loaded_model = load_model_and_classifier()
#     # if facenet is None or loaded_model is None:
#     #     st.stop()

#     name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
#     # MTCNN detector
#     detector = MTCNN()

#     # Hàm xử lý từng frame của video/webcam hoặc ảnh
#     def process_frame(frame):
#         result = detector.detect_faces(frame)
#         if len(result) == 0:
#             return frame, None, None
        
#         x1, y1, width, height = result[0]['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         face = frame[y1:y2, x1:x2]
#         image = Image.fromarray(face)
#         image = image.resize((160, 160))
#         resized_arr = np.asarray(image)
        
#         # Tính toán embedding vector
#         embed_vector = create_embedding.get_embedding(facenet, resized_arr)
        
#         return frame, embed_vector, (x1, y1, x2, y2)

#     # Dự đoán class của khuôn mặt
#     def predict(model, embed_vector):
#         sample = np.expand_dims(embed_vector, axis=0)
#         yhat_index = model.predict(sample)
#         yhat_prob = np.max(model.predict_proba(sample)[0])
#         class_predict = name_list[yhat_index[0]]
#         return yhat_prob, class_predict

#     # Hàm nhận diện từ webcam
#     def recognize_from_webcam():
#         global webcam_active
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()
        
#         while webcam_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 st.write("Failed to capture image from webcam.")
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ video
#     def recognize_from_video(video_path):
#         global video_active
#         cap = cv2.VideoCapture(video_path)
#         stframe = st.empty()
        
#         while video_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 st.write("Failed to capture frame from video.")
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ ảnh
#     def recognize_from_image(uploaded_file):
#         img = Image.open(uploaded_file)
#         img_arr = np.array(img)
        
#         frame, embed_vector, box = process_frame(img_arr)
        
#         if embed_vector is not None:
#             prob, pred_class = predict(loaded_model, embed_vector)
#             x1, y1, x2, y2 = box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             frame = cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         st.image(frame, caption="Processed Image", channels="RGB")

#     # Streamlit App
#     def main():
#         global webcam_active, video_active

#         st.title("Face Recognition From Image, Video, and Webcam with MTCNN and FaceNet")
        
#         option = st.selectbox("Choose Function", ["Image", "Webcam", "Video"])
#         st.write(f"Selected option: {option}")
        
#         if option == "Image":
#             uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#             if uploaded_file is not None:
#                 recognize_from_image(uploaded_file)
        
#         elif option == "Webcam":
#             if st.button("Start Webcam"):
#                 webcam_active = True
#                 st.write("Starting webcam...")
#                 recognize_from_webcam()
            
#             if st.button("Stop Webcam"):
#                 webcam_active = False
#                 st.write("Stopping webcam...")
        
#         elif option == "Video":
#             video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
#             if video_file is not None:
#                 temp_video_path = f"temp_video.{video_file.type.split('/')[-1]}"
#                 with open(temp_video_path, 'wb') as f:
#                     f.write(video_file.read())
                
#                 if st.button("Start Video"):
#                     video_active = True
#                     st.write("Starting video...")
#                     recognize_from_video(temp_video_path)
                
#                 if st.button("Stop Video"):
#                     video_active = False
#                     st.write("Stopping video...")

#     if __name__ == '__main__':
#         main()


























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # # Load model và classifier
    # facenet = facenet_architecture.InceptionResNetV2()
    # path = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/facenet_keras_weights.h5"
    # facenet.load_weights(path)

    # name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
    # file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/classify.sav"
    # loaded_model = pickle.load(open(file_name, "rb"))

    # # MTCNN detector
    # detector = MTCNN()
    
    # # Hàm xử lý từng frame của video/webcam hoặc ảnh
    # def process_frame(frame):
    #     result = detector.detect_faces(frame)
    #     if len(result) == 0:
    #         return frame, None, None
        
    #     x1, y1, width, height = result[0]['box']
    #     x1, y1 = abs(x1), abs(y1)
    #     x2, y2 = x1 + width, y1 + height
    #     face = frame[y1:y2, x1:x2]
    #     image = Image.fromarray(face)
    #     image = image.resize((160, 160))
    #     resized_arr = np.asarray(image)
        
    #     # Tính toán embedding vector
    #     embed_vector = create_embedding.get_embedding(facenet, resized_arr)
        
    #     return frame, embed_vector, (x1, y1, x2, y2)

    # # Dự đoán class của khuôn mặt
    # def predict(model, embed_vector):
    #     sample = np.expand_dims(embed_vector, axis=0)
    #     yhat_index = model.predict(sample)
    #     yhat_prob = np.max(model.predict_proba(sample)[0])
    #     class_predict = name_list[yhat_index[0]]
    #     return yhat_prob, class_predict

    # # Hàm nhận diện từ webcam/video
    # def recognize_from_video(video_source=0, stop_event=None):
    #     cap = cv2.VideoCapture(video_source)
        
    #     # Tạo một khung chứa video
    #     stframe = st.empty()
        
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
            
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
    #         frame, embed_vector, box = process_frame(frame_rgb)
            
    #         if embed_vector is not None:
    #             prob, pred_class = predict(loaded_model, embed_vector)
    #             x1, y1, x2, y2 = box
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    #         # Cập nhật khung video
    #         frame = Image.fromarray(frame)
    #         stframe.image(frame, caption="Video Stream", use_column_width=True)
            
    #         if stop_event and stop_event():
    #             break
        
    #     cap.release()

    # # Hàm nhận diện từ ảnh
    # def recognize_from_image(uploaded_file):
    #     img = Image.open(uploaded_file)
    #     img_arr = np.array(img)
        
    #     frame, embed_vector, box = process_frame(img_arr)
        
    #     if embed_vector is not None:
    #         prob, pred_class = predict(loaded_model, embed_vector)
    #         x1, y1, x2, y2 = box
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         frame = cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    #     st.image(frame, caption="Processed Image", channels="RGB")

    # # Streamlit App
    # def main():
    #     st.title("Face Recognition From Image, Video, and Webcam with MTCNN and FaceNet")
        
    #     option = st.selectbox("Choose Function", ["Image", "Webcam", "Video"])
        
    #     if option == "Image":
    #         uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    #         if uploaded_file is not None:
    #             recognize_from_image(uploaded_file)
        
    #     elif option == "Webcam":
    #         stop_button = st.button("Stop Webcam")
    #         if st.button("Start Webcam"):
    #             # Stop Webcam when stop_button is pressed
    #             stop_event = lambda: stop_button
    #             recognize_from_video(0, stop_event)
        
    #     elif option == "Video":
    #         video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    #         if video_file is not None:
    #             temp_video_path = f"temp_video.{video_file.type.split('/')[-1]}"
    #             with open(temp_video_path, 'wb') as f:
    #                 f.write(video_file.read())
    #             stop_button = st.button("Stop Video")
    #             if not stop_button:
    #                 stop_event = lambda: stop_button
    #                 recognize_from_video(temp_video_path, stop_event)

    # if __name__ == '__main__':
    #     main()   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    # from unicodedata import name
    # from tensorflow import keras
    # from keras.preprocessing import image
    # from PIL import Image
    # import streamlit as st
    # import numpy as np
    # import base64
    # import warnings
    # warnings.filterwarnings('ignore')
    # import pickle
    # from keras.utils import img_to_array, load_img
    # from preprocessing import *
    # from create_embedding import get_embedding
    # from facenet_architecture import InceptionResNetV2

    # import settings
    # import helper

    # facenet = InceptionResNetV2()
    # path = "D:/NCKH2023-2024/Monitoring Student/Attendance/MTCNN/model/facenet_keras_weights.h5"
    # facenet.load_weights(path)

    # name_list = ['22070018','22070154','22070156','22070167','22070277']
    # file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/MTCNN/model/classify.sav"
    # loaded_model = pickle.load(open(file_name, "rb"))


    # def get_base64(bin_file):
    #     with open(bin_file, 'rb') as f:
    #         data = f.read()
    #     return base64.b64encode(data).decode()


    # # Set background for local web
    # def set_background(png_file):
    #     bin_str = get_base64(png_file)
    #     page_bg_img = '''
    #     <style>
    #     .stApp {
    #     background-image: url("data:image/png;base64,%s");
    #     background-size: cover;
    #     }
    #     </style>
    #     ''' % bin_str
    #     st.markdown(page_bg_img, unsafe_allow_html=True)
    # set_background('background/Bg1.png')

    # def process_input(filename, target_size = (160,160)):
    #     """
    #     Input: Image filename directory
    #     Output: Resized 160x160 image and image file as array
    #     """
    #     img = load_img(filename)
    #     img_arr = np.array(img)
    #     result = detector.detect_faces(img_arr)
    #     if len(result) == 0:
    #         return None, None
    #     else:
    #         x1, y1, width, height = result[0]['box']
    #         x1, y1 = abs(x1), abs(y1)
    #         x2, y2 = x1 + width, y1 + height
    #         face = img_arr[y1:y2, x1:x2]
    #         image = Image.fromarray(face)
    #         image = image.resize(target_size)
    #         resized_arr = np.asarray(image)
    #     return image, resized_arr

    # def embed_input(model, resized_arr):
    #     """
    #     Convert resized_arr to embedded vector through face_net
    #     Input: Resized_arr of image
    #     Output: Embedded vector 
    #     """
    #     embed_vector = get_embedding(facenet, resized_arr)
    #     return embed_vector

    # def predict(model, embed_vector):
    #     """
    #     Input: Embedded vector extracted from image arr
    #     Output: Probability of class, class
    #     """
    #     sample = np.expand_dims(embed_vector, axis = 0)
    #     yhat_index  = model.predict(sample)
    #     yhat_prob = np.max(model.predict_proba(sample)[0])
    #     class_predict = name_list[yhat_index[0]]
    #     return yhat_prob, class_predict



    # def main():
    #     st.markdown("<h2 style='text-align:center; color: yellow;'>Attendance With MTCNN And Facenet</h2>",
    #                 unsafe_allow_html=True)
        

    #     html_class_term = """
    #     <div style="background-color: white ;padding:5px; margin: 20px">
    #     <h5 style="color:black;text-align:center; font-size: 10 px"> There are 5 classes in the dataset: ['22070018','22070154','22070156','22070167','22070277']</h5>
    #     """
    #     st.markdown(html_class_term, unsafe_allow_html=True)


    #     html_temp = """
    #        <div style="background-color: brown ;padding:5px">
    #        <h3 style="color:black;text-align:center; font-size: 15 px"> Click the below button to upload image.</h3>
    #        </div>
    #        """
    #     st.markdown(html_temp, unsafe_allow_html=True)
    #     st.markdown("")


    #     source_radio = st.sidebar.radio(
    #     "Select Source", settings.SOURCES_LIST)
    #     uploaded_file = None
        
    #     if source_radio == settings.IMAGE:
            
    #         uploaded_file = st.sidebar.file_uploader("Choose image file", accept_multiple_files=False)
    #     if uploaded_file is not None:
    #         st.write("File uploaded:", uploaded_file.name)
    #         show_img = load_img(uploaded_file,target_size=(300,300))
    #         st.image(show_img, caption= "Original image uploaded")
    #         save_dir = "image_from_user"
    #         with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
    #             f.write(uploaded_file.getbuffer())


    #     if st.button("Predict"):
    #         saveimg_dir = "image_from_user" + "\{}".format(uploaded_file.name)
    #         image, resized_arr = process_input(saveimg_dir)

    #         if (image == None and resized_arr == None):
    #             print("Can't detect face! Please try another image")
    #         else:
    #             embed_vector = embed_input(loaded_model, resized_arr)
    #             prob, pred_class = predict(loaded_model, embed_vector)
    #             st.success('Predict {} with confidence: {}'.format(pred_class, np.round(prob,4)))
                
                
    #     elif source_radio==settings.VIDEO:
    #           helper.play_stored_video(loaded_model)
    #     elif source_radio == settings.WEBCAM:
    #         helper.play_webcam(loaded_model)
    #     else:
    #         st.error("Please select a valid source type!")
        
    # if __name__=='__main__':
    #     main()

















# COPY DOAN DAU CHAY RIENG


# def run_app1():
#     from unicodedata import name
#     from tensorflow import keras
#     from keras.preprocessing import image
#     from PIL import Image
#     import numpy as np
#     import base64
#     import warnings
#     warnings.filterwarnings('ignore')
#     from keras.utils import img_to_array, load_img
#     #from preprocessing import *
# #def run_app1():  
#     import cv2
#     import streamlit as st
#     import numpy as np
#     import tensorflow as tf
#     from mtcnn import MTCNN 
#     from PIL import Image
#     import pickle
#     from keras.utils import img_to_array
#     import sys
#     sys.path.append('D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh')
#     #from Diemdanh 
#     import facenet_architecture
#     #from facenet_architecture import InceptionResNetV2
#     #from Diemdanh 
#     import create_embedding
#     #from create_embedding import get_embedding
#     import tempfile
    

#     # Load model và classifier
#     # facenet = InceptionResNetV2()
#     # path = "D:/NCKH2023-2024/Monitoring Student/Streamlit/Attentdance/MTCNN/model/facenet_keras_weights.h5"
#     # facenet.load_weights(path)

#     name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
#     # file_name = "D:/NCKH2023-2024/Monitoring Student/Streamlit/Attentdance/MTCNN/model/classify.sav"
#     # loaded_model = pickle.load(open(file_name, "rb"))

#     # MTCNN detector
#     detector = MTCNN()

#     # Biến cờ cho việc dừng webcam và video
#     webcam_active = False
#     video_active = False

#     # Tải model và classifier - chỉ thực hiện một lần
#     @st.cache_resource
#     def load_model_and_classifier():
#         facenet = facenet_architecture.InceptionResNetV2()
#         path = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/facenet_keras_weights.h5"
#         facenet.load_weights(path)
#         file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/classify.sav"
#         loaded_model = pickle.load(open(file_name, "rb"))
#         return facenet, loaded_model

#     facenet, loaded_model = load_model_and_classifier()

#     # Hàm xử lý từng frame của video/webcam hoặc ảnh
#     def process_frame(frame):
#         result = detector.detect_faces(frame)
#         if len(result) == 0:
#             return frame, None, None
        
#         x1, y1, width, height = result[0]['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         face = frame[y1:y2, x1:x2]
#         image = Image.fromarray(face)
#         image = image.resize((160, 160))
#         resized_arr = np.asarray(image)
        
#         # Tính toán embedding vector
#         embed_vector = create_embedding.get_embedding(facenet, resized_arr)
        
#         return frame, embed_vector, (x1, y1, x2, y2)

#     # Dự đoán class của khuôn mặt
#     def predict(model, embed_vector):
#         sample = np.expand_dims(embed_vector, axis=0)
#         yhat_index = model.predict(sample)
#         yhat_prob = np.max(model.predict_proba(sample)[0])
#         class_predict = name_list[yhat_index[0]]
#         return yhat_prob, class_predict

#     # Hàm nhận diện từ webcam
#     def recognize_from_webcam():
#         global webcam_active
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()
        
#         while webcam_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ video
#     def recognize_from_video(video_path):
#         global video_active
#         cap = cv2.VideoCapture(video_path)
#         stframe = st.empty()
        
#         while video_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ ảnh
#     def recognize_from_image(uploaded_file):
#         img = Image.open(uploaded_file)
#         img_arr = np.array(img)
        
#         frame, embed_vector, box = process_frame(img_arr)
        
#         if embed_vector is not None:
#             prob, pred_class = predict(loaded_model, embed_vector)
#             x1, y1, x2, y2 = box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             frame = cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         st.image(frame, caption="Processed Image", channels="RGB")

#     # Streamlit App
#     def main():
#         global webcam_active, video_active

#         st.title("Face Recognition From Image, Video, and Webcam with MTCNN and FaceNet")
        
#         option = st.selectbox("Choose Function", ["Image", "Webcam", "Video"])
        
#         if option == "Image":
#             uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#             if uploaded_file is not None:
#                 recognize_from_image(uploaded_file)
        
#         elif option == "Webcam":
#             if st.button("Start Webcam"):
#                 webcam_active = True
#                 recognize_from_webcam()
            
#             if st.button("Stop Webcam"):
#                 webcam_active = False
        
#         elif option == "Video":
#             video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
#             if video_file is not None:
#                 temp_video_path = f"temp_video.{video_file.type.split('/')[-1]}"
#                 with open(temp_video_path, 'wb') as f:
#                     f.write(video_file.read())
                
#                 if st.button("Start Video"):
#                     video_active = True
#                     recognize_from_video(temp_video_path)
                
#                 if st.button("Stop Video"):
#                     video_active = False

#     if __name__ == '__main__':
#         main()
















# #Copy tu Streamlit/attentdance/MTCNN
# def run_app1():
#     import cv2
#     import streamlit as st
#     import numpy as np
#     import tensorflow as tf
#     from mtcnn import MTCNN
#     from PIL import Image
#     import pickle
#     from keras.utils import img_to_array
#     import sys
#     sys.path.append('D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh')
#     from Diemdanh import facenet_architecture
#     from Diemdanh import create_embedding
#     #from facenet_architecture import InceptionResNetV2
#     #from create_embedding import get_embedding
#     import tempfile

#     # Load model và classifier
#     # facenet = InceptionResNetV2()
#     # path = "D:/NCKH2023-2024/Monitoring Student/Streamlit/Attentdance/MTCNN/model/facenet_keras_weights.h5"
#     # facenet.load_weights(path)

#     name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
#     # file_name = "D:/NCKH2023-2024/Monitoring Student/Streamlit/Attentdance/MTCNN/model/classify.sav"
#     # loaded_model = pickle.load(open(file_name, "rb"))

#     # MTCNN detector
#     detector = MTCNN()

#     # Biến cờ cho việc dừng webcam và video
#     # webcam_active = False
#     # video_active = False

#      # Inserting an icon before the title
#     st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)

#     # Inserting an icon and text on the same line
#     st.markdown(
#         '<div style="display: flex; align-items: center;">'
#         '<h1 style="margin: 0;">Monitoring Student in the Classroom using YOLOv8</h1>'
#         '<i class="fas fa-chalkboard-teacher" style="margin-right: 10px;"></i>'
#         '</div>',
#         unsafe_allow_html=True
#     )

#     #st.balloons()
#     # Sidebar
#     st.sidebar.header("WELCOME TO OUR RESEARCH!")
#     # Load your image
#     image = open('D:/NCKH2023-2024/Monitoring Student/Attendance/YOLO/class.jpg', "rb").read()  # Replace "class.jpg" with your image file path

#     # Display the image in the sidebar below the header
#     st.sidebar.image(image, use_column_width=True)
    
    
#     # Tải model và classifier - chỉ thực hiện một lần
#     @st.cache_resource
#     def load_model_and_classifier():
#         facenet = facenet_architecture.InceptionResNetV2()
#         path = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/facenet_keras_weights.h5"
#         facenet.load_weights(path)
#         file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/classify.sav"
#         loaded_model = pickle.load(open(file_name, "rb"))
#         return facenet, loaded_model

#     facenet, loaded_model = load_model_and_classifier()

#     # Hàm xử lý từng frame của video/webcam hoặc ảnh
#     def process_frame(frame):
#         result = detector.detect_faces(frame)
#         if len(result) == 0:
#             return frame, None, None
        
#         x1, y1, width, height = result[0]['box']
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         face = frame[y1:y2, x1:x2]
#         image = Image.fromarray(face)
#         image = image.resize((160, 160))
#         resized_arr = np.asarray(image)
        
#         # Tính toán embedding vector
#         embed_vector = create_embedding.get_embedding(facenet, resized_arr)
        
#         return frame, embed_vector, (x1, y1, x2, y2)

#     # Dự đoán class của khuôn mặt
#     def predict(model, embed_vector):
#         sample = np.expand_dims(embed_vector, axis=0)
#         yhat_index = model.predict(sample)
#         yhat_prob = np.max(model.predict_proba(sample)[0])
#         class_predict = name_list[yhat_index[0]]
#         return yhat_prob, class_predict

#     # Hàm nhận diện từ webcam
#     def recognize_from_webcam():
#         global webcam_active
#         cap = cv2.VideoCapture(0)
#         stframe = st.empty()
        
#         while webcam_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ video
#     def recognize_from_video(video_path):
#         global video_active
#         cap = cv2.VideoCapture(video_path)
#         stframe = st.empty()
        
#         while video_active and cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame, embed_vector, box = process_frame(frame_rgb)
            
#             if embed_vector is not None:
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#             frame = Image.fromarray(frame)
#             stframe.image(frame, caption="Video Stream", use_column_width=True)
        
#         cap.release()

#     # Hàm nhận diện từ ảnh
#     def recognize_from_image(uploaded_file):
#         img = Image.open(uploaded_file)
#         img_arr = np.array(img)
        
#         frame, embed_vector, box = process_frame(img_arr)
        
#         if embed_vector is not None:
#             prob, pred_class = predict(loaded_model, embed_vector)
#             x1, y1, x2, y2 = box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             frame = cv2.putText(frame, f'{pred_class}: {prob:.2f}', (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
#         st.image(frame, caption="Processed Image", channels="RGB")

#     # Streamlit App
#     def main():
#         global webcam_active, video_active

#         st.title("Face Recognition From Image, Video, and Webcam with MTCNN and FaceNet")
        
#         option = st.selectbox("Choose Function", ["Image", "Webcam", "Video"])
        
#         if option == "Image":
#             uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#             if uploaded_file is not None:
#                 recognize_from_image(uploaded_file)
        
#         elif option == "Webcam":
#             if st.button("Start Webcam"):
#                 webcam_active = True
#                 recognize_from_webcam()
            
#             if st.button("Stop Webcam"):
#                 webcam_active = False
        
#         elif option == "Video":
#             video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
#             if video_file is not None:
#                 temp_video_path = f"temp_video.{video_file.type.split('/')[-1]}"
#                 with open(temp_video_path, 'wb') as f:
#                     f.write(video_file.read())
                
#                 if st.button("Start Video"):
#                     video_active = True
#                     recognize_from_video(temp_video_path)
                
#                 if st.button("Stop Video"):
#                     video_active = False

#     if __name__ == '__main__':
#         main()



# from unicodedata import name
# from tensorflow import keras
# from keras.preprocessing import image
# from PIL import Image
# import streamlit as st
# import numpy as np
# import base64
# import warnings
# warnings.filterwarnings('ignore')
# import pickle
# from keras.utils import img_to_array, load_img
# import sys
# sys.path.append('D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh')
# from Diemdanh.preprocessing import *
# from Diemdanh.create_embedding import get_embedding
# from Diemdanh.facenet_architecture import InceptionResNetV2
# def run_app1():
#     # from unicodedata import name
#     # from tensorflow import keras
#     # from keras.preprocessing import image
#     # from PIL import Image
#     # import streamlit as st
#     # import numpy as np
#     # import base64
#     # import warnings
#     # warnings.filterwarnings('ignore')
#     # import pickle
#     # from keras.utils import img_to_array, load_img
#     # from Diemdanh.preprocessing import *
#     # from Diemdanh.create_embedding import get_embedding
#     # from Diemdanh.facenet_architecture import InceptionResNetV2

#     facenet = InceptionResNetV2()
#     path = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/Facenet_keras_weights.h5"
#     facenet.load_weights(path)

#     name_list = ['22070018', '22070154', '22070156', '22070167', '22070277']
#     file_name = "D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/model/classify.sav"
#     loaded_model = pickle.load(open(file_name, "rb"))


#     def get_base64(bin_file):
#         with open(bin_file, 'rb') as f:
#             data = f.read()
#         return base64.b64encode(data).decode()


#     # Set background for local web
#     def set_background(png_file):
#         bin_str = get_base64(png_file)
#         page_bg_img = '''
#         <style>
#         .stApp {
#         background-image: url("data:image/png;base64,%s");
#         background-size: cover;
#         }
#         </style>
#         ''' % bin_str
#         st.markdown(page_bg_img, unsafe_allow_html=True)
#     set_background('D:/NCKH2023-2024/Monitoring Student/Attendance/Diemdanh/background/Bg1.png')

#     def process_input(filename, target_size = (160,160)):
#         """
#         Input: Image filename directory
#         Output: Resized 160x160 image and image file as array
#         """
#         img = load_img(filename)
#         img_arr = np.array(img)
#         result = detector.detect_faces(img_arr)
#         if len(result) == 0:
#             return None, None
#         else:
#             x1, y1, width, height = result[0]['box']
#             x1, y1 = abs(x1), abs(y1)
#             x2, y2 = x1 + width, y1 + height
#             face = img_arr[y1:y2, x1:x2]
#             image = Image.fromarray(face)
#             image = image.resize(target_size)
#             resized_arr = np.asarray(image)
#         return image, resized_arr

#     def embed_input(model1, resized_arr):
    
#         embed_vector = get_embedding(model1, resized_arr)
#         return embed_vector

#     def predict(model1, embed_vector):
#         """
#         Input: Embedded vector extracted from image arr
#         Output: Probability of class, class
#         """
#         sample = np.expand_dims(embed_vector, axis = 0)
#         yhat_index  = model1.predict(sample)
#         yhat_prob = np.max(model1.predict_proba(sample)[0])
#         class_predict = name_list[yhat_index[0]]
#         return yhat_prob, class_predict



#     def main():
#         st.markdown("<h2 style='text-align:center; color: yellow;'>Face Recogniton With MTCNN And Facenet</h2>",
#                     unsafe_allow_html=True)
        

#         html_class_term = """
#         <div style="background-color: white ;padding:5px; margin: 20px">
#         <h5 style="color:black;text-align:center; font-size: 10 px"> There are 14 classes in the dataset: ['Duc', 'HDuc', 'Hieu', 'Hung', 'Kien', 'Linh', 'Quan', 'Tan', 'Thang'
#                 ,'Truong', 'Tuan', 'Van', 'VietDuc', 'XuanAnh']</h5>
#         """
#         st.markdown(html_class_term, unsafe_allow_html=True)


#         html_temp = """
#         <div style="background-color: brown ;padding:5px">
#         <h3 style="color:black;text-align:center; font-size: 15 px"> Click the below button to upload image.</h3>
#         </div>
#         """
#         st.markdown(html_temp, unsafe_allow_html=True)
#         st.markdown("")


#         uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
#         if uploaded_file is not None:
#             st.write("File uploaded:", uploaded_file.name)
#             show_img = load_img(uploaded_file,target_size=(300,300))
#             st.image(show_img, caption= "Original image uploaded")
#             save_dir = "image_from_user"
#             with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
#                 f.write(uploaded_file.getbuffer())


#         if st.button("Predict"):
#             saveimg_dir = "image_from_user" + "\{}".format(uploaded_file.name)
#             image, resized_arr = process_input(saveimg_dir)

#             if (image == None and resized_arr == None):
#                 print("Can't detect face! Please try another image")
#             else:
#                 embed_vector = embed_input(loaded_model, resized_arr)
#                 prob, pred_class = predict(loaded_model, embed_vector)
#                 st.success('Predict {} with confidence: {}'.format(pred_class, np.round(prob,4)))
                
#     if __name__=='__main__':
#         main()
















