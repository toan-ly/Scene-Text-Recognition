from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")


def format_predictions(predictions_str):
    """Format predictions response into readable JSON"""
    try:
        # Evaluate the string representation of predictions
        predictions = eval(predictions_str)
        if not predictions:
            return "[]"
            
        # Format the JSON-like structure with indentation
        formatted_json = "[\n"
        for bbox, class_name, conf, text in predictions:
            formatted_json += "  {\n"
            formatted_json += f"    \"bbox\": {bbox},\n"
            formatted_json += f"    \"class\": \"{class_name}\",\n"
            formatted_json += f"    \"confidence\": {conf:.2f},\n"
            formatted_json += f"    \"text\": \"{text}\"\n"
            formatted_json += "  },\n"
        formatted_json = formatted_json.rstrip(",\n") + "\n]"
        
        return formatted_json
    except Exception as e:
        return f"Error formatting predictions: {str(e)}"


def process_image_url(url, api_url="http://localhost:8000"):
    """Process image from URL using the OCR API"""
    try:
        response = requests.get(f"{api_url}/ocr", params={"image_url": url})
        response.raise_for_status()

        # Get predictions from headers
        predictions = response.headers.get("X-Predictions", "[]")

        # Display the processed image
        image = Image.open(BytesIO(response.content))
        return image, predictions
    except requests.RequestException as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None


def process_uploaded_file(file, api_url="http://localhost:8000"):
    """Process uploaded image file using the OCR API"""
    try:
        # Validate the uploaded file
        try:
            Image.open(file)
            # Reset file pointer
            file.seek(0)
        except Exception as e:
            st.error("⚠️ Please upload a valid image file")
            return None, None

        # Prepare the file for upload
        files = {"file": ("image.png", file, "image/png")}

        response = requests.post(f"{api_url}/ocr/upload", files=files)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"❌ Server Error: {error_detail}")
            return None, None

        response.raise_for_status()

        # Get predictions from headers
        predictions = response.headers.get("X-Predictions", "[]")

        # Display the processed image
        image = Image.open(BytesIO(response.content))
        return image, predictions
    except requests.RequestException as e:
        st.error(f"🚨 Request error: {str(e)}")
        if hasattr(e.response, "json"):
            try:
                error_detail = e.response.json().get("detail", "")
                st.error(f"Server details: {error_detail}")
            except:
                pass
        return None, None
    except Exception as e:
        st.error(f"🚨 Unexpected error: {str(e)}")
        return None, None


def main():
    st.title("📸 OCR Image Processing")

    st.sidebar.title("⚙️ Settings")
    # API endpoint configuration
    api_url = st.sidebar.text_input("API Endpoint", "http://localhost:8000")

    # Two tabs for URL and file upload modes
    tabs = st.tabs(["🌐 From URL", "📁 Upload Image"])

    # ------ URL ------
    with tabs[0]: 
        st.subheader("🔗 Process Image from URL")
        image_url = st.text_input("Paste image URL here...")

        # Create two columns for images
        col1, col2 = st.columns(2)

        # Original image
        with col1:
            st.markdown("**Original Image**")
            if image_url:
                try:
                    response = requests.get(image_url)
                    original_image = Image.open(BytesIO(response.content))
                    st.image(original_image, use_container_width=True)
                except:
                    st.error("⚠️ Could not load image from URL")

        # Processed image 
        with col2:
            st.markdown("**Processed Image**")
            # Empty placeholder for processed image
            st.empty()

        if image_url and st.button("Process", key="url_button"):
            with st.spinner("🔍 Processing image..."):
                image, predictions = process_image_url(image_url, api_url)
                if image:
                    # Update the processed image in col2
                    with col2:
                        st.image(image, use_container_width=True)

                    # Predictions below
                    st.subheader("📝 Detected Text")
                    st.code(format_predictions(predictions), language="text")

    # ------ File Upload ------
    with tabs[1]:
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose a file (jpg, png)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(uploaded_file, use_container_width=True)
            
            with col2:
                st.markdown("**Processed Image**")
                st.empty()

            if st.button("Process", key="upload_button"):
                with st.spinner("🔍 Processing image..."):
                    image, predictions = process_uploaded_file(uploaded_file, api_url)
                    if image:
                        with col2:
                            st.image(image, use_container_width=True)

                        st.subheader("📝 Detected Text")
                        st.code(format_predictions(predictions), language="text")


if __name__ == "__main__":
    main()