# 🌾 Krishi Sahayak

**Krishi Sahayak** is an intelligent, multimodal chatbot system designed for Indian farmers. It enables communication via WhatsApp using voice and image inputs in native Indian languages. Inputs are processed, translated, and answered using AI — with responses in text or audio, in the user's own language.

---

## 🚜 Use Case

Small and marginal farmers in India often struggle with access to timely, accurate agricultural advice in their native language. **Krishi Sahayak** addresses this gap by offering:

- ✅ Voice-based interaction over WhatsApp  
- ✅ Image support (e.g., crop damage, pesticide labels)  
- ✅ Regional language translation and synthesis  
- ✅ Agricultural insights via AI (powered by OpenAI)

---

## 🏗️ Architecture Overview

```
User (WhatsApp)
└── Twilio Webhook (Flask)
    ├── Audio Input → audio_pipeline.py
    │   ├── Whisper ASR (using Sarvam AI Saarika)
    │   ├── Translator (to English using indicTrans2)
    │   ├── LLM Prompt (Gemini API)
    │   ├── Translator (back to local lang using indicTrans2)
    │   └── TTS (using Sarvam AI BulBul)
    ├── Image Input → image_pipeline.py
    │   ├── OCR (Tesseract)
    │   ├── Translator
    │   ├── LLM Prompt
    │   └── Text Reply
    └── MongoDB Logging → db.py
```

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `audio_pipeline.py` | Voice input pipeline: STT → Translate → LLM → TTS |
| `image_pipeline.py` | Image processing pipeline using OCR and LLM |
| `config.py` | Manages environment variables securely |
| `db.py` | MongoDB connection and logging |
| `llm.py` | Handles prompt formatting and interaction with Gemini AI |
| `stt.py` | Converts voice to text |
| `tts.py` | Converts text to audio |
| `translator.py` | Language translation |
| `whatsapp_webhook.py` | Flask app for Twilio WhatsApp webhook |

---

## 🧑‍💻 Tech Stack

| Layer        | Technology           |
|--------------|----------------------|
| AI Model     | Gemini-2.5-flash |
| STT          | Sarvam AI Saarika-v2.5 |        |
| TTS          | Sarvam AI Bulbul-v2.5 |
| OCR          | Tesseract             |
| Translation  | AI4BHARAT'S IndicTrans2 API  |
| Backend      | Flask                 |
| Messaging    | Twilio WhatsApp API   |
| Database     | MongoDB (pymongo)     |
| Deployment   | Localhost / Ngrok / Cloud |

---

## ⚙️ Setup Instructions

<details>
<summary><strong>1. Clone the Repository</strong></summary>

```bash
git clone https://github.com/algo-aryan/krishi-sahayak.git
cd krishi-sahayak
```
</details>

<details>
<summary><strong>2. Install Dependencies</strong></summary>

```bash
pip install -r requirements.txt
```

Required packages:

```
Flask==3.0.0
python-dotenv==1.0.0
requests==2.31.0
twilio==8.12.0
pymongo==4.6.1
google-generativeai==0.3.2
transformers==4.36.2
torch==2.1.2
torchaudio==2.1.2
indictrans2-ai4bharat==0.1.0
ffmpeg-python==0.2.0
gunicorn==21.2.0

```
</details>

<details>
<summary><strong>3. Setup Environment Variables</strong></summary>

Create a `.env` file in the root:

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_NUMBER=whatsapp:+14155238886

SARVAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

MONGODB_URI=mongodb://localhost:27017/aiman

FLASK_ENV=development
FLASK_DEBUG=true

```
</details>

<details>
<summary><strong>4. Install Tesseract</strong></summary>

- **macOS:** `brew install tesseract`  
- **Ubuntu:** `sudo apt install tesseract-ocr`  
- **Windows:** [Download Tesseract](https://github.com/tesseract-ocr/tesseract)
</details>

<details>
<summary><strong>5. Run the Flask Webhook</strong></summary>

```bash
python whatsapp_webhook.py
```
</details>

<details>
<summary><strong>6. Expose with Ngrok (for WhatsApp testing)</strong></summary>

```bash
ngrok http 5000
```

Copy the generated URL and set it as the **Webhook URL** in Twilio’s WhatsApp Sandbox.
</details>

---

## 🔐 Access Control

- No login is required for users  
- Twilio credentials required for WhatsApp integration  
- All secrets must be stored in `.env`

---

## 🧪 Testing Guidelines

| Input Type | Expected Output     |
|------------|----------------------|
| 🎙️ Voice   | Audio + Text reply   |
| 🖼️ Image   | Text reply           |
| 🧑‍🌾 Lang   | Native language reply |

---

## ⚠️ Error Handling

| Module       | Failure Strategy                     |
|--------------|--------------------------------------|
| STT          | Fallback to manual transcription     |
| OCR          | Informs user if image is unreadable  |
| Translator   | Retry once, then default to English  |
| LLM          | Sends default message on timeout     |
| TTS          | Sends text reply if audio fails      |

---

## 🧠 Prompt Strategy

All LLM prompts follow this structure:

```
A farmer asked: '<translated_input>'.
Reply to him in <original_language> like a friendly agriculture expert.
```

---

## 📌 To-Do / Future Enhancements

- ✅ Better support for regional dialects  
- ✅ Add fallback prompt layers  
- ⏳ Hindi voice synthesis (Coqui / iSpeech)  
- ⏳ UI dashboard for interaction logs  
- ⏳ Location-based (pin-code) suggestions  

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## 🤝 Contributors

Made with ❤️ for Bharat Kisan by:

- **Aryan Bansal** ([@algo-aryan](https://github.com/algo-aryan))
- **Arnav Bansal** ([@algo-aryan](https://github.com/algo-aryan))
- **Arnsh Goel** ([@algo-aryan](https://github.com/algo-aryan))
- **Arnav Goyal** ([@algo-aryan](https://github.com/algo-aryan))