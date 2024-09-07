import streamlit as st
from Diemdanh.app import run_app1
from YOLO.app import run_app2

 #Setting page layout
st.set_page_config(
        page_title="Monitoring Student System",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def main():
    st.sidebar.title("Please give your choice!")
    option = st.sidebar.radio("Task", ["BEHAVIOR", "ATTENDANCE"])

    if option == "BEHAVIOR":
        run_app2()
    elif option == "ATTENDANCE":
        run_app1()

if __name__ == "__main__":
    main()
