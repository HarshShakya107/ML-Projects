# Text to Speech Streamlit App

Ye app **text ya PDF files ko read, summarize aur auto Q&A** ke liye use hoti hai, aur saath hi **audio output** generate karti hai using gTTS.

---

## Features

1. Upload a **.txt** file or **.pdf** file.
2. Choose a **language** (English, Hindi, Spanish, French, German, Japanese, Chinese).
3. Tasks available:
   - **Read as it is**: Text ko screen pe dikhaye aur audio generate kare.
   - **Summarize**: Text ka summary generate kare (Hugging Face `distilbart-cnn-12-6` model use).
   - **Auto Q&A**: Automatic questions generate kare aur answers dikhaye.
4. Audio streaming using **gTTS**.

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
