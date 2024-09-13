






   # Python In-built packages
# from pathlib import Path
# import PIL

# # External packages
# import streamlit as st
# Local Modules

import sys
sys.path.append('web-streamlit/YOLO')
from YOLO import settings
from YOLO import helper






def run_app2():
    # Python In-built packages
    from pathlib import Path
    import PIL

    # External packages
    import streamlit as st
    # Local Modules
    
    # import sys
    # sys.path.append('D:/NCKH2023-2024/Monitoring Student/Attendance/YOLO')
    # from YOLO import settings
    # from YOLO import helper
    # import YOLO.settings
    # import YOLO.helper
    
    # import settings
    # import helper
    #import test_videos.py


    # Setting page layout
    # st.set_page_config(
    #     page_title="Monitoring Student System",
    #     page_icon=":rocket:",
    #     layout="wide",
    #     initial_sidebar_state="expanded"
    # )


    # Inserting an icon before the title
    st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)

    # Inserting an icon and text on the same line
    st.markdown(
        '<div style="display: flex; align-items: center;">'
        '<h1 style="margin: 0;">Monitoring Student in the Classroom using YOLOv8</h1>'
        '<i class="fas fa-chalkboard-teacher" style="margin-right: 10px;"></i>'
        '</div>',
        unsafe_allow_html=True
    )

    #st.balloons()
    # Sidebar
    st.sidebar.header("WELCOME TO OUR RESEARCH!")
    # Load your image
    image = open('YOLO/class.jpg', "rb").read()  # Replace "class.jpg" with your image file path

    # Display the image in the sidebar below the header
    st.sidebar.image(image, use_column_width=True)



    # Model Options
    model_type = st.radio("Selected Task: ",['BEHAVIOR'])
    

    if model_type == 'BEHAVIOR':
        #global model_path
        model_path = Path(settings.DETECTION_MODEL)

    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Let's come in the classroom with us!!")

    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)
    source_img = None
    #If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button('Detect Students'):
                    res = model.predict(uploaded_image,
                                        #conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex: 
                        # st.write(ex)
                        st.write("No image is uploaded yet!")
        #test_videos.process(img,results)


    elif source_radio == settings.VIDEO:
        helper.play_stored_video(#confidence,
                                model)
    # test_videos.process(img,results)
    # elif source_radio == settings.WEBCAM:
    #     helper.play_webcam(#confidence, 
    #                     model)
        #test_videos.process(img,results)
        
    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(#confidence, 
                                model)
        #test_videos.process(img,results)
    else:
        st.error("Please select a valid source type!")

    result=st.sidebar.radio("RESULTS: ",['TRAINING','TEST_VIDEO'])
    if result == 'TRAINING':
        st.image('YOLO/result_training.png', caption="This is result of training YOLOv8 models!")
    elif result == 'TEST_VIDEO':
        st.image('YOLO/Bieudohanhvi.jpg', caption="This is result of test video in the class with 5 minutes!")
    else:
        st.error("Please select a valid choice!")









