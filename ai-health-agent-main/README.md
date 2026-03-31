# AI Health Agent (BMI + Risk + Recommendations)

This project is an AI-powered health monitoring and recommendation app built around the provided `bmi.xlsx` dataset.

## Features

- BMI calculator (metric) and BMI-class categorization
- Dataset-driven obesity risk prediction (trained from `bmi.xlsx`)
- Personalized diet and fitness recommendations based on BMI + risk
- Dataset insights:
  - BMI distribution
  - correlation (Age vs BMI)
  - prevalence comparison (overweight/obesity rates in dataset)
- Accounts + persistence (SQLite): profiles, check-ins, chat history
- Chatbot interface:
  - rule-based fallback
  - optional LLM mode if `OPENAI_API_KEY` is set

## Setup

```bash
cd "c:\Users\LOQ\Downloads\SAPO Casestudy"
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Optional: enable LLM chat

Set an environment variable, then run the app:

```bash
setx OPENAI_API_KEY "your_key_here"
```

Optional model override:

```bash
setx OPENAI_MODEL "gpt-4o-mini"
```

## Dataset

The app expects `bmi.xlsx` to exist in the project folder and uses the `bmi` sheet.
