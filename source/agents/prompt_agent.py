import requests
from abc import ABC, abstractmethod
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Assuming you store your Groq API key in .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/v1/complete")  # Example URL

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_groq(self, messages, temperature=0.7, max_tokens=150):
        retries = 0
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "groq-large",  # Assuming Groq uses model names like OpenAI
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Groq API:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")
                
                # Make the API request to Groq
                response = requests.post(GROQ_API_URL, headers=headers, json=payload)
                response.raise_for_status()  # Will raise an exception for non-2xx status codes
                
                response_json = response.json()
                reply = response_json['choices'][0]['message']['content']  # Assuming the structure is similar to OpenAI
                
                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")
                return reply
            
            except requests.exceptions.RequestException as e:
                retries += 1
                logger.error(f"[{self.name}] Error during Groq API call: {e}. Retry {retries}/{self.max_retries}")
        
        raise Exception(f"[{self.name}] Failed to get response from Groq after {self.max_retries} retries.")
