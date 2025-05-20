import os
import logging
import time
import eventlet
eventlet.monkey_patch()  # Ensure non-blocking I/O with eventlet

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_session import Session

import redis
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Use a consistent secret key (from environment variable) for production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")

# Configure server-side sessions using Redis
app.config["SESSION_TYPE"] = "redis"
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
app.config["SESSION_REDIS"] = redis.StrictRedis(host=redis_host, port=redis_port)
app.config["SESSION_PERMANENT"] = False
Session(app)

# Set maximum allowed file upload payload to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Initialize SocketIO with Eventlet for asynchronous operations
socketio = SocketIO(app, async_mode='eventlet')

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch API key and initialize Google Gen AI client and model
api_key = os.environ.get("API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-pro-exp-03-25"

# Define and ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helpers to manage conversation history stored in the session
def init_conversation_history():
    if 'conversation_history' not in session:
        session['conversation_history'] = []

def add_message_to_history(role, text):
    history = session.get('conversation_history', [])
    history.append({'role': role, 'text': text})
    session['conversation_history'] = history

def build_prompt(user_input):
    history = session.get('conversation_history', [])
    prompt = ""
    # Label messages appropriately
    for message in history:
        role_label = "User" if message['role'] == "user" else "Diagnostic AI"
        prompt += f"\n{role_label}: {message['text']}"
    prompt += f"\nUser: {user_input}"
    return prompt

# Route to serve the main page
@app.route('/')
def index():
    init_conversation_history()
    return render_template('index.html')

# Endpoint to reset conversation history
@app.route('/reset_conversation', methods=['GET'])
def reset_conversation():
    session.pop('conversation_history', None)
    return redirect(url_for('index'))

# Endpoint to handle image uploads securely
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({"message": "Image uploaded successfully", "image_path": filepath})
    else:
        return jsonify({"error": "Unsupported file type"}), 400

# Primary socket event for sending and receiving messages
@socketio.on('send_message')
def handle_chat_message(json_data):
    user_input = json_data.get('user_input', '').strip()
    if not user_input:
        emit('chat_response', {"error": "Please enter a valid question."})
        return

    # Build prompt from the conversation history and new input
    full_prompt = build_prompt(user_input)
    logger.info(f"Built prompt (SocketIO): {full_prompt}")

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)]
        )
    ]

    # On new conversation, include the disclaimer; omit it if continuing the dialogue
    history = session.get('conversation_history', [])
    if not history:
        system_text = (
            "You are an advanced diagnostic AI assistant identified as Dr Expert. "
            "Your role is to provide short but accurate, evidence-based diagnostic assessments and "
            "offer proper medical advice and assistance to patients based on their input. "
            "DISCLAIMER: The information provided is not a substitute for professional medical consultation."
        )
    else:
        system_text = (
            "You are an advanced diagnostic AI assistant identified as Dr Expert."
            "Your role is to provide short but accurate, evidence-based diagnostic assessments and "
            "offer proper medical advice and assistance to patients based on their input."
        )

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text=system_text)]
    )

    max_retries = 3
    retry_delay = 2  # seconds
    full_response = None

    # Retry loop in case of transient errors
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            full_response = ""
            for chunk in response:
                full_response += chunk.text
            break
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                eventlet.sleep(retry_delay)
            else:
                emit('chat_response', {"error": f"An error occurred: {str(e)}. Please try again."})
                return

    if full_response is None:
        emit('chat_response', {"error": "No response from the AI service."})
        return

    # Update session's conversation history
    add_message_to_history('user', user_input)
    add_message_to_history('assistant', full_response)

    emit('chat_response', {"response": full_response})

# Socket event handler for conversation summarization
@socketio.on('summarize_conversation')
def handle_summarize():
    history = session.get('conversation_history', [])
    if not history:
        emit('summary_response', {"error": "No conversation history to summarize."})
        return

    # Concatenate message texts from the conversation history
    summary_text = " ".join([msg['text'] for msg in history])
    summary_prompt = "Please summarize the following conversation concisely: " + summary_text

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=summary_prompt)]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[types.Part.from_text(text="Summarize the conversation.")]
    )

    max_retries = 3
    retry_delay = 2
    summary_response = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            summary_response = ""
            for chunk in response:
                summary_response += chunk.text
            break
        except Exception as e:
            logger.error(f"Summarize attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                eventlet.sleep(retry_delay)
            else:
                emit('summary_response', {"error": f"Summarization error: {str(e)}."})
                return

    if summary_response:
        # Optionally replace conversation history with the summary message
        new_history = [{"role": "assistant", "text": "Summary: " + summary_response}]
        session['conversation_history'] = new_history
        emit('summary_response', {"summary": summary_response})

# Socket event handler for receiving feedback on AI responses
@socketio.on('feedback')
def handle_feedback(data):
    message_index = data.get('message_index')
    rating = data.get('rating')
    logger.info(f"Received feedback for message index {message_index}: {rating}")
    # Here you would normally store the feedback for analysis.
    emit('feedback_response', {"status": "Feedback recorded."})

if __name__ == "__main__":
    socketio.run(app, debug=True)
