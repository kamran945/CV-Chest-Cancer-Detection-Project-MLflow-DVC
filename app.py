import os


from src.lungCancerDetection.pipeline.prediction import PredictionPipeline


import streamlit as st
from PIL import Image


class ImageProcessingApp:
    def __init__(self):
        self.uploaded_image = None
        self.result_message = ""

    def display_ui(self):
        # Setting up the main layout
        st.title("Image Upload and Processing App")

        # Using Streamlit columns for layout with equal width and adjustable height
        col_left, col_right = st.columns(2, gap="small")

        # Define a minimum height to maintain column alignment
        min_height = 300  # Adjust this value based on your preference

        # Column 1 - Left (File Uploader and Action Buttons)
        with col_left:
            with st.container():
                st.header("Upload Image")
                self.uploaded_image = st.file_uploader(
                    "Choose an image", type=["jpg", "jpeg", "png"]
                )

                # Action Buttons placed below the file uploader
                st.write("")  # Adding a little space
                col_left_btn1, col_left_btn2 = st.columns(2)
                with col_left_btn1:
                    if st.button("Action 1"):
                        self.result_message = self.perform_action1()
                with col_left_btn2:
                    if st.button("Action 2"):
                        self.result_message = self.perform_action2()

            # Spacer to control height
            st.write(" " * min_height)

        # Column 2 - Right (Image Display and Result)
        with col_right:
            with st.container():
                st.header("Display and Results")
                if self.uploaded_image:
                    # Display uploaded image
                    image = Image.open(self.uploaded_image)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                else:
                    st.write("Please upload an image to display it here.")

                # Display result message below the image
                st.write(self.result_message)

            # Spacer to control height
            st.write(" " * min_height)

    def perform_action1(self):
        if self.uploaded_image:
            # Action 1 logic goes here
            os.system("dvc repro")

            return "Training performed successfully!"
        else:
            return "Please upload an image first."

    def perform_action2(self):
        if self.uploaded_image:
            # Action 2 logic goes here

            self.classifier = PredictionPipeline(self.uploaded_image)
            result = self.classifier.predict()
            return f"Prediction: {result}"
        else:
            return "Please upload an image first."


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.display_ui()
