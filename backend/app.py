from flask import Flask, render_template, request, jsonify
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq # Corrected import path for Groq

import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# --- Groq & LangChain Setup ---
# Load API key from .env or system environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment!")

# Initialize the Groq LLM with the correct class
llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# Initialize Conversation Memory
memory = ConversationBufferMemory()

# Define Prompt Template: 'emotion' is embedded within the 'input' variable
prompt_template = PromptTemplate(
    input_variables=["history", "input"], # Removed 'emotion' as a separate input variable
    template="""
    You are a compassionate mental health support chatbot trained to provide thoughtful and helpful responses. 
    You have access to the user's current facial emotion. Use this emotion context to tailor your response appropriately, especially if the user is expressing distress or negative emotions.

    Chat History:
    {history}

    User: {input}
    Bot:""" # The 'input' variable will now contain both the emotion and the user's message
)

# Initialize Conversation Chain with the emotion-aware prompt
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template
)

# --- Flask Routes ---

@app.route("/")
def index():
    # Renders the main chatbot interface (index.html)
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot interaction."""
    data = request.json
    user_message = data.get("message") # Renamed to avoid conflict with prompt's 'input'
    detected_emotion = data.get("emotion", "neutral/unknown") 

    if not user_message:
        return jsonify({"error": "No input received"}), 400

    try:
        # Combine emotion and user message into a single input string for the LLM
        # This is the key change to resolve the ValidationError
        combined_input_for_llm = f"User's emotion: {detected_emotion}. User's message: {user_message}"
        
        # Pass only the 'input' variable to the ConversationChain
        response = conversation.run(input=combined_input_for_llm) 
        return jsonify({"response": response})
    except Exception as e:
        print(f"[ERROR] Chatbot interaction failed: {e}")
        return jsonify({"response": "Sorry, I am having trouble processing that right now."})

if __name__ == "__main__":
    app.run(debug=True)
