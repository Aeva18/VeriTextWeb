import streamlit as st
import json
from streamlit_lottie import st_lottie
from Backend import get_result
from Backend import session_logs

# Set up page configuration
st.set_page_config(
    page_title="AI Generated Text Detector",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for white background and styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 15px;
        font-size: 16px;
        cursor: pointer;
    }
    .stTextArea > label {
        font-weight: bold;
        font-size: 14px;
    }
    .stTable {
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
if "page" not in st.session_state:
    st.session_state.page = "Predict"

def update_page():
    st.session_state.page = st.session_state.page_radio

st.sidebar.markdown("### Navigation")
st.sidebar.radio(
    "Go to:",
    ["Predict", "Session History"],
    index=["Predict", "Session History"].index(st.session_state.page),
    key="page_radio",
    on_change=update_page,
)

# Predict page
if st.session_state.page == "Predict":
    st.title("💬 VeriText: AI Generated Text Detector")
    st.markdown(
        """
        🚀 **AI Text Detection Utilizing Perplexity and GLTR for Thesis Purpose**  
        Enter any text and see the model's confidence on whether it's AI-generated or human-written.  
        """
    )
    st.divider()

    st.sidebar.header("Instructions")
    st.sidebar.write(
        """
        - Paste the text in the input box.
        - Click on "Analyze Text" to process.
        - Wait for the animation and see the result!
        """
    )

    st.markdown("### Your Input Text")
    user_input = st.text_area(
        "Enter your text below:",
        height=200,
        placeholder="Type or paste your text here...",
    )

    if st.button("Analyze Text"):
        if user_input:
            with st.spinner("Analyzing..."):
                result = get_result(user_input)
                st.success("Analysis complete!")
                st.write(result)
        else:
            st.warning("Please enter some text to analyze.")

# Session Logs page
elif st.session_state.page == "Session History":
    st.title("📜 Session History")
    st.markdown("Here are the history of your sessions...")

    if session_logs:
        st.table(session_logs)
    else:
        st.info("No history available for this session.")
