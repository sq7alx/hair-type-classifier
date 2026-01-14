import streamlit as st
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.models.predict import HairSegmenter, HairClassifierSystem, find_latest_run_id, get_inference_transforms

PAGE_TITLE = "Hair Type Classifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_MAP = {
    0: "Type 1 (Straight)",
    1: "Type 2 (Wavy)",
    2: "Type 3 (Curly)"
}

CUSTOM_CSS = """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# model cache
@st.cache_resource
def load_models(device_name):
    device = torch.device(device_name)
    
    try:
        run_id = find_latest_run_id()
    except Exception as e:
        st.error(f"Model not found: {e}")
        return None, None

    with st.spinner(f'Loading models from {run_id}...'):
        segmenter = HairSegmenter(device)
        classifier = HairClassifierSystem(run_id, device)
    
    return segmenter, classifier

def main():
    st.title("Hair Type Classifier Demo")
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Running on: **{device_type.upper()}**")

    st.write("Drop your image for analysis.")

    # model loading
    segmenter, classifier = load_models(device_type)

    if segmenter is None or classifier is None:
        st.stop()

    # file upload
    uploaded_file = st.file_uploader("Upload or drop your image here", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception:
            st.error("Error loading file.")
            st.stop()

        col1, col2, col3 = st.columns([1, 1, 1.5])

        with col1:
            st.write("**Original**")
            st.image(image, caption="Original", width='stretch')

        with col2:
            st.write("**Background removal check**")
            masked_image = segmenter.remove_background(uploaded_file) 
            st.image(masked_image, caption="Detected hair", width='stretch')

        # classification and results
        with col3:
            st.write("**Analysis**")
            transform = get_inference_transforms()
            input_tensor = transform(masked_image).unsqueeze(0)
            
            label, details = classifier.predict(input_tensor)

            # result box
            st.markdown(f"""
            <style>
                .result-box {{
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    padding: 2rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                }}
                .result-highlight {{
                    font-size: 3.5rem;
                    font-weight: 700;
                    background: -webkit-linear-gradient(45deg, #4CAF50, #81C784);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                }}
                .result-text {{
                    color: #e0e0e0;
                    font-size: 1rem;
                    margin-top: 0.5rem;
                }}
            </style>

            <div class="result-box">
                <h3 style="margin:0; color: #aaa; font-size: 1rem;">ANALYSIS RESULT</h3>
                <div class="result-highlight">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("---")
            
            # detailed probabilities
            sorted_probs = sorted(details['all_sub_probs'].items(), key=lambda x: x[1], reverse=True)

            for name, prob in sorted_probs:
                if prob * 100 >= 0:
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        if name == label:
                            st.markdown(f"**{name}**")
                        else:
                            st.markdown(f"{name}")
                    with col_bar:
                        st.progress(prob)
                        st.caption(f"{prob:.2%}")

if __name__ == "__main__":
    main()