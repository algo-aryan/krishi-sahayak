# 🌾 Krishi Sahayak

**Krishi Sahayak** is a multimodal conversational assistant tailored for Indian farmers. It allows users to interact via audio or image inputs over WhatsApp, and uses AI to interpret queries, translate between Indian languages and English, and respond meaningfully with localized agricultural advice.

---

## 📦 Features

- 🎙️ Audio-to-Text + Translation + Response + Text-to-Speech
- 🖼️ Image-to-Text (OCR) + Translation + Response
- 📲 WhatsApp-based interaction using Twilio
- 🌐 Supports multilingual queries and responses
- 🤖 LLM integration with context-aware prompting

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `audio_pipeline.py` | Handles audio input: STT → Translation → LLM → Back Translation → TTS |
| `image_pipeline.py` | Extracts text from images: OCR → Translation → LLM response |
| `config.py` | Stores environment variables |
| `db.py` | MongoDB connection setup |
| `llm.py` | Sends prompt to OpenAI LLM and gets response |
| `stt.py` | Uses Whisper to convert audio to text |
| `tts.py` | Converts response text to speech |
| `translator.py` | Translates text between Indian languages and English |
| `whatsapp_webhook.py` | Flask webhook for Twilio WhatsApp integration |

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/algo-aryan/krishi-sahayak.git
cd krishi-sahayak
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
openai
pytesseract
pillow
googletrans==4.0.0rc1
gtts
pydub
pymongo
Flask
twilio
python-dotenv
requests
```

### 3. Set up Environment Variables

Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string
```

### 4. Install Tesseract

- **macOS:** `brew install tesseract`
- **Ubuntu:** `sudo apt install tesseract-ocr`
- **Windows:** [Download here](https://github.com/tesseract-ocr/tesseract/wiki)

### 5. Run the Flask Server

```bash
python whatsapp_webhook.py
```

---

## 🌐 WhatsApp Integration (Twilio)

1. Create a Twilio account
2. Enable a WhatsApp number from the console
3. Point the webhook to your public server:

Use ngrok during development:
```bash
ngrok http 5000
```
Copy the HTTPS forwarding URL and paste it in the Twilio WhatsApp sandbox webhook config.

---

## 🧠 Prompt Design

All user inputs are translated to English, processed by LLM using agriculture-related prompts, and translated back to the original language.

Example Prompt:
> "A farmer asked this: 'translated query'. Reply as a helpful agriculture advisor in [original language]."

---

## 🧪 Testing

- Send an audio or image to your Twilio WhatsApp number.
- Receive a contextual response (audio + text for voice inputs).
- Logs will be shown in terminal.

---

## 📌 To-Do / Improvements

- Add error handling for all pipeline stages
- Add video/image doc support
- Improve multilingual detection & fallback strategy
- Add user tracking/logging in MongoDB

---

## 🤝 Contributors

- Aryan Bansal ([@algo-aryan](https://github.com/algo-aryan))

---

## 📄 License

MIT License