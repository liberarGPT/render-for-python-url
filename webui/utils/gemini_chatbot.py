import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time

load_dotenv() # Load environment variables from .env file

class GeminiChatbot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiChatbot, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[GEMINI_CHATBOT] WARNING: GEMINI_API_KEY not found. Chatbot will work in demo mode.")
            self.model = None
            self.chat_sessions: Dict[str, Any] = {}
            self._initialized = True
            return
        
        try:
            genai.configure(api_key=api_key)
            
            # Try different model names in order of preference (faster models first)
            model_names = [
                'gemini-2.0-flash-exp',  # Fastest
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-flash-latest',
                'gemini-1.5-flash'
            ]
            self.model = None
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    print(f"[GEMINI_CHATBOT] Successfully initialized with model: {model_name}")
                    break
                except Exception as model_error:
                    print(f"[GEMINI_CHATBOT] Failed to initialize with {model_name}: {model_error}")
                    continue
            
            if self.model is None:
                raise Exception("No compatible Gemini model found")
                
            self.chat_sessions: Dict[str, Any] = {} # Stores chat.GenerativeModel.start_chat() objects per session_id
            self._initialized = True
            print("[GEMINI_CHATBOT] GeminiChatbot initialized successfully.")
            
        except Exception as e:
            print(f"[GEMINI_CHATBOT] WARNING: Failed to initialize Gemini: {e}. Chatbot will work in demo mode.")
            self.model = None
            self.chat_sessions: Dict[str, Any] = {}
            self._initialized = True

    def get_or_create_chat_session(self, session_id: str) -> Any:
        """
        Retrieves an existing chat session or creates a new one for a given session_id.
        """
        if self.model is None:
            # Return a dummy object for demo mode
            return type('DummyChat', (), {'send_message': lambda msg: type('DummyResponse', (), {'text': 'Demo mode - no API key configured'})()})()
        
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = self.model.start_chat(history=[])
            print(f"[GEMINI_CHATBOT] Created new chat session for ID: {session_id}")
        return self.chat_sessions[session_id]

    def send_message(self, session_id: str, user_message: str, context: Optional[str] = None) -> str:
        """
        Sends a message to the Gemini model within a specific chat session,
        optionally including additional context.
        """
        if self.model is None:
            return "I'm currently running in demo mode. To use the full chatbot functionality, please configure your GEMINI_API_KEY in the .env file."
        
        chat = self.get_or_create_chat_session(session_id)
        
        full_message = user_message
        if context:
            full_message = f"Context about the app interface: {context}\n\nUser query: {user_message}"
            
        try:
            # Add timeout and faster generation settings
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1024,  # Increased for more detailed responses
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
            
            start_time = time.time()
            response = chat.send_message(
                full_message,
                generation_config=generation_config
            )
            end_time = time.time()
            
            print(f"[GEMINI_CHATBOT] Response time: {end_time - start_time:.2f}s")
            return response.text
        except Exception as e:
            print(f"[GEMINI_CHATBOT] Error sending message to Gemini: {e}")
            return f"I'm sorry, I encountered an error: {e}"

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the conversation history for a given session_id.
        """
        if self.model is None:
            return []  # No history in demo mode
        
        chat = self.get_or_create_chat_session(session_id)
        history = []
        for message in chat.history:
            role = "user" if message.role == "user" else "model"
            history.append({"role": role, "text": message.parts[0].text})
        return history

    def clear_history(self, session_id: str):
        """
        Clears the conversation history for a given session_id.
        """
        if self.model is None:
            return  # Nothing to clear in demo mode
        
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            print(f"[GEMINI_CHATBOT] Cleared chat session for ID: {session_id}")

# Initialize the chatbot instance lazily
gemini_chatbot = None

def get_gemini_chatbot():
    """Get the Gemini chatbot instance, initializing it only when needed."""
    global gemini_chatbot
    if gemini_chatbot is None:
        gemini_chatbot = GeminiChatbot()
    return gemini_chatbot