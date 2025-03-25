import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class AppConfig:
    def __init__(self):
        os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# Function to load configurations
def load_config():
    return AppConfig()
