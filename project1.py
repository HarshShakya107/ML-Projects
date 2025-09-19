import pandas as pd
import streamlit as st
import numpy as np
import time 
from gtts import gTTS
import tempfile
import PyPDF2
from googletrans import Translator
from transformers import pipeline

st.title("TEXT TO SPEECH")

uploaded_file=st.file_uploader("Upload your text file:",type=["text"])
uploaded_pdf=st.file_uploader("Upload your text file:",type=["pdf"])
languages={
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-cn"
}
file_text=""
languagess=st.selectbox("Select language:",list(languages.keys()))
language=languages[languagess]

summarizer =pipeline("summarization",model="sshleifer/distilbart-cnn-12-6",device=-1)
qa_pipeline =pipeline("question-answering",model="distilbert-base-cased-distilled-squad",device=-1)
translator=Translator()
def stream_data_with_voice(text,lang):
    if lang != "en":
        translated=translator.translate(text,dest=lang).text
    else:
        translated=text
    
    for word in translated.split(" "):
        yield word + " "
        time.sleep(0.03)
    
    tts=gTTS(text=translated,lang=lang)
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp3") as tmpfile:
         tts.save(tmpfile.name)
         st.audio(tmpfile.name,format="audio/mp3")

if uploaded_file is not None:
    file_text=uploaded_file.read().decode("utf-8")
    
    if st.button("Stream Data with audio:"):
       st.write(stream_data_with_voice(file_text,language))
 
if uploaded_pdf is not None:
    reader=PyPDF2.PdfReader(uploaded_pdf)
    
    for page in reader.pages:
        file_text+=page.extract_text() + "\n"

if file_text:
    task=st.radio("Choose Task:",["Read as it is", "Summarize", "Auto Q&A"])  

    if st.button("Run"):
        if task=="Read as it is":
            st.write(stream_data_with_voice(file_text,language))

        elif task == "Summarize":
            summary=summarizer(file_text[:1000],max_length=150,min_length=50,do_sample=False)
            result=summary[0]['summary_text'] 
            st.subheader("Summary")
            st.write(result)
            st.write(stream_data_with_voice(result,language))
        
        elif task == "Auto Q&A":
            st.subheader("Auto Genrated Q&A")
            questions=[
                "What is the main idea of the text?",
                "Who is mentioned in the text?",
                "What is the conclusion?"
            ]
            qna_text = ""
            for q in questions:
                ans = qa_pipeline(question=q, context=file_text[:1000])
                qna_text += f"Q: {q}\nA: {ans['answer']}\n\n"
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {ans['answer']}")
            
            st.write(stream_data_with_voice(qna_text, language))
