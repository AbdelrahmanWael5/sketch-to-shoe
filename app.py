import io
import streamlit as st
from PIL import Image
from inference import ShoeGenerator

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Sketch-to-Shoe",
    page_icon="👟",
    layout="wide"
)

# Hide Streamlit menu/footer
hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return ShoeGenerator("DiT_best_checkpoint.pth")


generator = load_model()

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.title("👟 Sketch-to-Shoe Generation")

st.divider()

# ----------------------------------------------------
# Upload
# ----------------------------------------------------
uploaded = st.file_uploader(
    "Upload an edge image",
    type=["png", "jpg", "jpeg"]
)

output = None

# ----------------------------------------------------
# Main UI
# ----------------------------------------------------
if uploaded is not None:

    edge = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Sketch")
        st.image(edge, use_container_width=True)

    with col2:
        st.subheader("Generated Shoe")
        placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    _, center, _ = st.columns([2, 1, 2])

    with center:
        generate = st.button(
            "🚀 Generate Shoe",
            use_container_width=True
        )

    if generate:

        with st.spinner("Generating image... Please wait."):

            output = generator.generate(edge)

        placeholder.image(
            output,
            use_container_width=True
        )

        st.success("Generation completed!")

        # Download button
        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        st.download_button(
            label="📥 Download Generated Image",
            data=buffer.getvalue(),
            file_name="generated_shoe.png",
            mime="image/png",
            use_container_width=True,
        )