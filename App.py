import streamlit as st
import json
from streamlit_lottie import st_lottie
from Backend import get_result
from Backend import session_logs


# Set up page configuration
st.set_page_config(
    page_title="VeriText: AI Generated Text Detector",
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
    ["Predict", "Session History", "About VeriText"],
    index=["Predict", "Session History", "About VeriText"].index(st.session_state.page),
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
        - For example, go to chat.openai.com and 
        - Try to generate simple prompt like "Generate essay about cars"
        and paste the result in the input box.
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

# About VeriText page
elif st.session_state.page == "About VeriText":
    st.title("ℹ️ About VeriText")
    st.markdown(
        """
        VeriText leverages state-of-the-art AI techniques to detect whether a text is human-written or AI-generated.  
        
        **Key Methods Used**:
        - **Perplexity**: Measures how "predictable" the text is, based on a language model's confidence.
        - **GLTR Features**: Analyzes token-level probabilities to identify patterns typical of AI-generated text.
        
        These features are extracted from powerful pretrained models like GPT-2 Large and GPT-Neo 125M and combined with ensemble machine learning algorithms, 
        consist of XGBoost, LightGBM, CatBoost, and Random Forest to achieve high accuracy in text classification.

        **Baseline Method: LLMDet**:
        As a baseline, VeriText uses the LLMDet methodology, which calculates proxy perplexity using n-gram statistics. This method:
        - Constructs n-gram dictionaries from the dataset.
        - Estimates token probabilities based on relative frequencies.
        - Uses proxy perplexity as features for a machine learning classifier that same as main method, which is ensemble method consist of XGBoost, LightGBM, CatBoost, and Random Forest for fairness in comparison.
        
        While LLMDet is computationally efficient due to its reliance on n-gram probabilities, it lacks the nuanced understanding of language context provided by pretrained language models. VeriText overcomes this limitation by leveraging pretrained models such as GPT-2 Large and GPT-Neo 125M, which are trained on vast corpora of text data. These models allow us to calculate perplexity based on richer, contextualized token predictions rather than simple n-gram frequencies.

        Furthermore, VeriText incorporates **GLTR (Giant Language Model Test Room)** features, which analyze token-level rankings and probabilities to provide additional insights. GLTR helps the model understand the context of the data more effectively by identifying whether token choices are consistent with human-like patterns or AI-generated outputs. By combining these advanced features with pretrained models, VeriText achieves a deeper understanding of linguistic nuances, resulting in superior performance compared to LLMDet.
        """
    )
    
    # Display an example image
    st.markdown("### How It Works")
    st.image("Assets/VeriTextMethod.png", caption="VeriText Method Overview", use_column_width=True)
    st.image("Assets/LLMDetMethod.png", caption="LLMDet Method Overview", use_column_width=True)


    st.markdown('### Data Collection')
    st.markdown("""  
        This research utilizes a combined dataset of essays generated by AI and humans, totaling 788,922 entries, with 394,350 used due to resource limitations. 
        - The AI dataset (55.8%) includes data from DAIGT, MAGE, and various LLMs such as ChatGPT, GPT-4, Claude, and PaLM, comprising over 220,000 entries. 
        - The human dataset (44.2%) consists of essays from students (ELLIPSE, IELTS), opinion essays (EssayForum & IvyPanda), and a persuasion corpus based on instructions, totaling over 174,000 entries. 
        This diverse combination ensures robust detection of AI-generated text.
    """)

    st.markdown('### Result')
    data = {
        "Methods": ["LLMDet", "gpt2-large Perplexity & GLTR", "gpt-neo-125m Perplexity & GLTR", "gpt2-large + gpt-neo-125m Perplexity & GLTR"],
        "Accuracy (%)": ["68%", "88%", "82%", "89%"],
        "Precision (%)": ["69%", "88%", "82%", "89%"],
        "Recall (%)": ["68%", "88%", "82%", "89%"],
        "F1 Score (%)": ["67%", "88%", "82%", "89%"],
    }
    
    # Create a table from example data
    st.table(data)

    st.markdown(
        """
        The best model achieved, which is the combination of perplexity & GLTR that extracted with gpt2-large and gpt-neo-125m, with accuracy, precision, recall, and F1 score of 89%, used for final model that used in this website
        """
    )