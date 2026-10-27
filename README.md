# GlucoSync — Congressional App Challenge MVP

A Python + Streamlit web-app prototype for meal-aware diabetes decision support.

## Features
- Meal photo upload
- Optional vision-API food suggestions
- Editable ingredient, portion, carbohydrate, protein, fat, and fiber estimates
- Session-only recipe library
- Synthetic glucose timeline and sample meal history
- Clear labels separating synthetic demo data from real medical data

## Run locally

1. Install Python 3.10+.
2. Unzip this folder and open a terminal in the project directory.
3. Create a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run:

   ```bash
   streamlit run app.py
   ```

## Enable photo analysis

Set an API key in your terminal before launching the app. Do not paste API keys into source code or commit them to GitHub.

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

The app defaults to `gpt-4.1-mini` for image analysis. If your account does not have access to that model, set `OPENAI_VISION_MODEL` to a vision-capable model available to your account. API usage may incur costs. Without a key, the app still runs using the sample meal.

## Important limitations
- This is an educational prototype, not a medical device or validated clinical decision-support tool.
- It does not calculate insulin doses, recommend corrections, alter pump settings, or connect to a real CGM/pump.
- All glucose readings and sample meal-response data are synthetic.
- Photo-derived portion sizes and nutrient estimates can be wrong. Users must review ingredients and portions.
- Do not enter personal health information or identifiable patient data.
- Before any public deployment, review security, privacy, accessibility, and relevant medical-device requirements.

## Suggested demo script
1. Open Meal Studio and load the sample meal.
2. Edit the portion or a nutrient value to show user control.
3. Save the meal.
4. Open Recipe Library and show the saved meal.
5. Open My Patterns and explain the synthetic glucose timeline.
6. If the vision API is configured, upload a meal photo and review the model's suggestions.
