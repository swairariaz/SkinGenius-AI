"""
SKIN GENIUS - STREAMLIT UI v1.2
================================
Purpose:
- User-friendly interface for skincare analysis.
- Supports text/upload interactions.

Features:
1. Dark theme with custom CSS.
2. Session history sidebar.
3. Dynamic response formatting (emojis/Markdown).
4. Follow-up question handling.

Tech Stack:
- Streamlit
- Requests (API calls)
- PIL (image uploads)
- UUID (session tracking)

Workflow:
1. User inputs query → Calls /analyze.
2. Displays recipe + diagnosis.
3. Follow-ups → Calls /followup.
"""

import uuid
import streamlit as st
import requests
import json
import os
from datetime import datetime
import base64
from PIL import Image

# Page configuration with sAI-inspired title and icon
st.set_page_config(
    page_title="SkinGenius by sAI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black theme and skincare background
st.markdown(
    """
    <style>
    .main {
        background: url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1350&q=80') no-repeat center center fixed;
        background-size: cover;
        background-color: #1a1a1a; /* Black base */
        color: #e0e0e0; /* Light text for contrast */
        padding: 20px;
        border-radius: 10px;
        position: relative;
        overflow: hidden;
    }
    .main::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7); /* Black overlay for partial visualization */
        z-index: 0;
    }
    .content {
        position: relative;
        z-index: 1;
    }
    .stButton>button {
        background-color: #00cc66; /* Soft green accent inspired by sAI */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00b359;
        transform: translateY(-2px);
    }
    .stTextInput>div>input, .stFileUploader>div>input {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stTextInput>div>label, .stFileUploader>label {
        color: #00cc66;
        font-weight: bold;
    }
    .response-box {
        background: #2a2a2a;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    .sidebar .sidebar-content {
        background: #1a1a1a;
        color: #e0e0e0;
    }
    .sidebar .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .footer {
        text-align: center;
        padding: 10px;
        color: #999;
        font-size: 0.8em;
    }
    .sidebar-caption {
        text-align: center;
        color: #e0e0e0;
        font-style: italic;
        font-size: 0.9em;
        padding: 10px;
        margin-top: 20px;
        border-top: 1px solid #444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# API endpoints
ANALYZE_URL = "http://localhost:8000/analyze"
FOLLOWUP_URL = "http://localhost:8000/followup"
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "memory")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'waiting' not in st.session_state:
    st.session_state.waiting = False


# Load existing memory from individual session files
def load_memory():
    memory = {}
    if os.path.exists(MEMORY_DIR):
        for filename in os.listdir(MEMORY_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(MEMORY_DIR, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        if isinstance(session_data, dict) and 'session_id' in session_data:
                            memory[session_data['session_id']] = session_data
                except json.JSONDecodeError as e:
                    st.error(f"Error loading session file {filename}: {str(e)}")
    return memory


# Save memory (handled by backend, no need to save here)
def save_memory(memory):
    pass  # Backend handles saving to individual files


# Send request to backend
def send_request(endpoint, data):
    try:
        response = requests.post(endpoint, files=data if 'image' in data else data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        return None


# Display conversation with natural formatting and enhanced emojis
def display_conversation():
    for msg in st.session_state.conversation:
        with st.container():
            st.markdown(f"**You:** {msg['user_input']}", unsafe_allow_html=True)
            with st.expander("**SkinGenius Response** 🌿", expanded=True):
                response = msg['response']
                if 'diagnosis' in response:
                    diagnosis = response['diagnosis']
                    st.markdown(
                        f"✨ **Diagnosis**: Your skin type is {diagnosis.get('skin_type', 'unknown').capitalize()}! 💖 "
                        f"I detect {', '.join(diagnosis.get('conditions', {}).get('semantic_matches', []) or diagnosis.get('conditions', {}).get('exact_matches', []) or ['no specific conditions'])}. "
                        f"{diagnosis.get('emotional_context', {}).get('skincare_context', '')} 🧴",
                        unsafe_allow_html=True
                    )
                    recipe = response['recipes']
                    st.markdown(
                        f"🌸 **Recipe: {recipe['name']}** ✨\n"
                        f"- 💕 **Ingredients**: {', '.join(recipe['ingredients'])}\n"
                        f"- 🛁 **Steps**: {'; '.join(recipe['steps'])}\n"
                        f"- ⚠️ **Safety**: {recipe['safety_warning']} 😊\n"
                        f"- 🌙 **Best Time**: {recipe['usage_time'].capitalize()}",
                        unsafe_allow_html=True
                    )
                    if 'safety_override' in recipe:
                        st.markdown(f"- 📝 **Note**: {recipe['safety_override']} ❗", unsafe_allow_html=True)
                elif 'answer' in response:
                    st.markdown(f"💡 **Answer**: {response['answer']} 🌿", unsafe_allow_html=True)
                    if 'recipe' in response and response['recipe']:
                        recipe = response['recipe']
                        st.markdown(
                            f"🌺 **Adapted Recipe: {recipe['name']}** ✨\n"
                            f"- 💕 **Ingredients**: {', '.join(recipe['ingredients'])}\n"
                            f"- 🛁 **Steps**: {'; '.join(recipe['steps'])}\n"
                            f"- ⚠️ **Safety**: {recipe['safety_warning']} 😊\n"
                            f"- 🌙 **Best Time**: {recipe['usage_time'].capitalize()}",
                            unsafe_allow_html=True
                        )
                    if 'alternatives' in response and response['alternatives']:
                        st.markdown(f"- 🌟 **Alternatives**: {', '.join(response['alternatives'])} 💖",
                                    unsafe_allow_html=True)
                    if 'followup' in response:
                        st.markdown(f"- 💬 **Follow-Up**: {response['followup']} 🌸", unsafe_allow_html=True)
                else:
                    st.markdown("**Response**: Unable to parse response. Please try again. 😔", unsafe_allow_html=True)


# Sidebar for navigation
with st.sidebar:
    st.image("https://xai-public.s3.amazonaws.com/xai-logo.png", use_container_width=True)
    st.title("SkinGenius by sAI")
    if st.button("Start New Session"):
        st.session_state.session_id = None
        st.session_state.conversation = []
        st.session_state.waiting = False
        st.rerun()

    st.subheader("Session History")
    memory = load_memory()
    for sid, data in memory.items():
        conversation = data.get('conversation', [])
        timestamp = 'N/A'
        if conversation and isinstance(conversation, list):
            for msg in conversation:
                if isinstance(msg, dict) and 'timestamp' in msg:
                    timestamp = msg['timestamp']
                    break
        if st.button(f"Session {sid[:8]}... ({timestamp})"):
            st.session_state.session_id = sid
            st.session_state.conversation = conversation
            st.session_state.waiting = True
            st.rerun()

    # Aesthetic caption at the bottom of the sidebar
    st.markdown(
        '<div class="sidebar-caption">🌸 Remember: Your skin is unique and beautiful at every stage of its journey</div>',
        unsafe_allow_html=True
    )

# Main content
st.markdown('<div class="main"><div class="content">', unsafe_allow_html=True)
st.title("SkinGenius AI - Your Personal Dermatologist 🌿")
st.write("Describe your skin concern and optionally upload an image for personalized advice.")

# Input section
if not st.session_state.waiting:
    user_input = st.text_input("What’s your skin issue? (e.g., 'I have acne')", key="user_input")
    uploaded_file = st.file_uploader("Upload a skin image (optional, your privacy is respected)",
                                     type=["jpg", "jpeg", "png"], key="image_upload")

    if st.button("Get Skincare Advice"):
        if user_input or uploaded_file:
            data = {
                'text': (None, user_input),
                'session_id': (None, st.session_state.session_id or str(uuid.uuid4()))
            }
            if uploaded_file:
                data['image'] = (uploaded_file.name, uploaded_file.getvalue())

            response = send_request(ANALYZE_URL, data)
            if response and response.get("status") == "success":
                st.session_state.session_id = response["session_id"]
                st.session_state.conversation.append({
                    "user_input": user_input,
                    "response": response["analysis"],
                    "timestamp": datetime.now().isoformat()
                })
                st.write("Debug: API Response:", response)  # Temporary debug
                st.session_state.waiting = True
                st.rerun()
        else:
            st.warning("Please provide a description or image.")
else:
    st.write("**Your Personalized Skincare Plan:**")
    display_conversation()
    follow_up = st.text_input("Ask a follow-up (e.g., 'I don’t have these ingredients' or 'Can I go in sunlight?')",
                              key="follow_up")

    if st.button("Send Follow-Up"):
        if follow_up:
            data = {
                'question': (None, follow_up),
                'session_id': (None, st.session_state.session_id)
            }
            response = send_request(FOLLOWUP_URL, data)
            if response and response.get("status") == "success":
                st.session_state.conversation.append({
                    "user_input": follow_up,
                    "response": response["response"],
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.waiting = True
                st.rerun()
            else:
                st.error("Failed to process follow-up. Please try again.")
        else:
            st.warning("Please enter a follow-up question.")

st.markdown('</div></div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div class="footer">
        © 2025 SkinGenius by sAI | Powered by cutting-edge AI
    </div>
    """,
    unsafe_allow_html=True
)