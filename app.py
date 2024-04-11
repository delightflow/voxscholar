import streamlit as st
import whisper
import base64
import streamlit.components.v1 as components

# Load the Whisper model
model = whisper.load_model("base")

# Function to transcribe speech using Whisper
def transcribe_speech(audio_data):
    audio = whisper.load_audio(audio_data)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    return result.text

def main():
    st.title("Research Paper Scraper with Speech-to-Text")

    # Render the audio recording component
    components.html(
        """
        <script src="/voxscholars/static/record.js"></script>
        <div id="audio-recorder"></div>
        """,
        height=200,
    )

    # Receive the base64-encoded audio data from the JavaScript component
    audio_data = components.get_component_value()

    if audio_data:
        # Decode the base64 audio data
        audio_bytes = base64.b64decode(audio_data)

        # Transcribe the speech using Whisper
        transcribed_text = transcribe_speech(audio_bytes)
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