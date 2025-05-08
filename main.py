import streamlit as st  # type: ignore
import tensorflow as tf
import tempfile
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
import numpy as np


@st.cache_resource
def load_model():
    # Load the pre-trained model
    model = tf.keras.models.load_model("garbage-classification.keras")
    return model


def model_prediction(test_image_path):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(
        test_image_path, target_size=(400, 400)
    )

    x = tf.keras.utils.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)

    return np.argmax(prediction)


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Garbage identification"]
)


if app_mode == "Garbage identification":
    st.header("Chào mừng đến với trang Phân loại rác thải")

    # Chọn phương thức nhập ảnh
    st.subheader("🔍 Chọn cách nhập ảnh")
    input_method = st.radio(
        "Bạn muốn sử dụng phương pháp nào?",
        ["📁 Tải ảnh từ máy", "📷 Chụp ảnh từ webcam"],
    )

    test_image = None
    captured_image = None
    temp_file_path = None

    if input_method == "📁 Tải ảnh từ máy":
        test_image = st.file_uploader(
            "Tải ảnh từ thiết bị", type=["jpg", "jpeg", "png"]
        )
    elif input_method == "📷 Chụp ảnh từ webcam":
        captured_image = st.camera_input("Chụp ảnh trực tiếp từ webcam")

    # Ưu tiên ảnh upload nếu có
    image = test_image if test_image is not None else captured_image

    if image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image.read())
            temp_file_path = tmp_file.name

        st.image(temp_file_path, caption="Ảnh bạn đã chọn", use_container_width=True)
        # Predict button
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                result_index = model_prediction(temp_file_path)
                # Reading Labels
                class_name = ['battery',
                    'biological',
                    'cardboard',
                    'clothes',
                    'glass',
                    'metal',
                    'paper',
                    'plastic',
                    'shoes',
                    'trash']

            st.success("Model is Predicting it's a {}".format(class_name[result_index]))

            with st.expander("Đọc thêm"):
                st.write("Dự đoán :")
                # battery
                if result_index == 0:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là pin thuộc nhóm "Rác thải điện tử".*
                        """
                    )
                    st.image(image)
                # biological
                if result_index == 1:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là thực phẩm thừa, vỏ trái cây, rau củ thuộc nhóm "Rác thải hữu cơ".*
                        """
                    )
                    st.image(image)
                # cardboard
                if result_index == 2:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là bìa cứng thuộc nhóm "Rác thải tái chế".*
                        """
                    )
                    st.image(image)
                # clothes
                if result_index == 3:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là vải thuộc nhóm "Rác thải công nghiệp".*
                        """
                    )
                    st.image(image)
                # glass
                if result_index == 4:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là thủy tinh thuộc nhóm "Rác thải tái chế".*
                        """
                    )
                    st.image(image)
                # metal
                if result_index == 5:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là vật dụng kim loại thuộc nhóm "Rác thải tái chế hoặc Rác thải công nghiệp".*
                        """
                    )
                    st.image(image)
                # paper
                if result_index == 6:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là giấy thuộc nhóm "Rác thải tái chế".*
                        """
                    )
                    st.image(image)
                # plastic
                if result_index == 7:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là thủy tinh thuộc nhóm "Rác thải tái chế hoặc Rác thải sinh hoạt".*
                        """
                    )
                    st.image(image)
                # shoes
                if result_index == 8:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là giày thuộc nhóm "Rác thải tái chế hoặc Rác thải sinh hoạt".*
                        """
                    )
                    st.image(image)
                # trash
                if result_index == 9:
                    st.write(
                        """
                        Ảnh chụp cho thấy đây là rác thải hỗn hợp thuộc nhóm "Rác thải sinh hoạt".*
                        """
                    )
                    st.image(image)

