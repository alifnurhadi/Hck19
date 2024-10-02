import os
import time
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
import playsound
import re
from PIL import Image
from formated_response import format_response
from gtts import gTTS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
import pygame



load_dotenv()

# Make sure to replace this with your actual OpenAI API key securely
# api_key = os.getenv('OPENAI_API_KEY')

# # Load documents from CSV
# csv_loader = CSVLoader("mix555.csv")
# documents = csv_loader.load()
# # Load environment variables

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Set up the language model
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, api_key=openai_api_key)

# Load and process the CSV data
datapath = 'D:\hacktiv8_alif\dok_dimana_dok\data_clean2.csv'
loader = CSVLoader(datapath)
documents = loader.load()

# Create a vector store
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
vectorstore = FAISS.from_documents(documents, embeddings)

# Load context from external file
with open('context.txt', 'r', encoding='utf-8') as file:
    context = file.read()

# Create the chatbot template
with open('chatbot_template.txt', 'r', encoding='utf-8') as file:
    chatbot_template = file.read()

chatbot_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=chatbot_template
)

# Create the LLMChain
chat_chain = LLMChain(
    llm=llm,
    prompt=chatbot_prompt
)

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        time.sleep(1)
        
        audio = r.listen(source ,timeout=7, phrase_time_limit=7)
    
        st.write("Processing...")

    try:
        text = r.recognize_google(audio, language='id-ID')
        return str(text)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def clean_text(text):
    text = re.sub(r'[*#-]', '', text)
    return text

def say(text):

    text = clean_text(text)    
    tts = gTTS(text=text, lang='id')
    tts.save("output.mp3")
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")
    finally:
        pygame.mixer.quit()
        os.remove('output.mp3')

def chat_with_bot(query):
    try:
        # Create the retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Fetch relevant documents
        relevant_docs = retriever.get_relevant_documents(query)

        # Prepare the inputs for the chat chain
        inputs = {
            "context": context + "\n" + "\n".join([doc.page_content for doc in relevant_docs]),
            "query": query
        }

        # Run the chat chain with the relevant documents
        result = chat_chain.invoke(inputs)

        formatted_response = format_response({
            'query': query,
            'text': result['text'],
            'context': inputs['context']
        })
        return formatted_response
        
        # return f'Pertanyaan: {query}\n\nJawaban: {result}'
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {str(e)}"

st.title("Cari Dokter")
st.write('### dok, dimana dok ?')

images = Image.open('logo.jpeg')
st.image(images, caption='dok, dimana dok')

if st.button("Start Recording"):
    text = speech_to_text()
    
    if text:
        try:
            response = chat_with_bot(text)
            st.write("You said: ", text)
            st.write(response)
            say(response)
            st.write('Complete')

        except Exception as e:
            st.error(f"Error during response generation: {str(e)}")
    else:
        st.error("No text detected.")

elif st.button('Start Writing'):
    user_input = st.text_input('Atau, masukkan pertanyaan Anda di sini:')
    if user_input:
        response = chat_with_bot(user_input)
        st.write(response)
        say(response)
        st.write('complete')
