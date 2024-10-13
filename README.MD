# Medical Chatbot Using CSV & PDF Files

## Project Overview
This project aims to develop a chatbot that can provide accurate medical information and advice based on the "Gale Encyclopedia of Medicine." PDF & Medical Transcriptions CSV
The chatbot is designed to assist users by answering questions related to medical conditions, treatments, symptoms, and preventive healthcare.

## Features
- **Medical Inquiry Handling:** Answers a wide range of medical questions from disease definitions to treatment options.
- **Symptom Checker:** Users can describe symptoms to receive potential diagnoses and advice.
- **Health Tips:** Provides preventive health tips and wellness information.

## Datasets
The chatbot utilizes information extracted from the "Gale Encyclopedia of Medicine" PDF FILE & Cleaned Medical Transcriptions CSV FILE, ensuring the responses are well-informed and reliable.

## Technologies Used
- **Python**: Primary programming language.
- Transformers: sentence-transformers/all-MiniLM-L6-v2
- Base model for chatbot: llama-2-7b-chat.ggmlv3.q8_0.bin Open source model from Huggingface uploaded by The Bloke, Make sure to download it locally to run the chatbot
- WEB-APP UI: Chainlit

## Installation
Install the Requirements.txt file by simply typing: pip install -r requirements.txt

## Run the ChatBot
chainlit run main.py

