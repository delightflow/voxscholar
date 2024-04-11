import streamlit as st
import google.cloud.speech as speech
from bs4 import BeautifulSoup
import requests
import whisper

model = whisper.load_model("base")

def transcribe_speech(audio_data):
    audio = whisper.load_audio(audio_data)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # return the transcribed text
    return result.text
  
  

# Function to scrape research papers
def scrape_research_papers(query):
    # Implement web scraping logic here
    pass

def main():
    st.title("Research Paper Scraper with Speech-to-Text")

    # Speech input
    audio_data = st.file_uploader("Upload audio file or record voice input", type=["wav", "mp3"])
    if audio_data is not None:
        transcribed_text = transcribe_speech(audio_data.read())
        st.write("Transcribed text:", transcribed_text)

        # Scrape research papers based on transcribed text
        research_papers = scrape_research_papers(transcribed_text)

        # Display research paper information
        for paper in research_papers:
            st.write(paper.title)
            st.write(paper.authors)
            st.write(paper.abstract)

if __name__ == "__main__":
    main()