import base64
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from io import BytesIO
from PIL import Image
from transformers import pipeline

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(
    page_title="GlucoSync",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEMO_FOODS = [
    {"food": "Cooked rice", "portion_g": 150, "carbs_g": 42, "protein_g": 4, "fat_g": 0.4, "fiber_g": 0.6},
    {"food": "Grilled chicken", "portion_g": 100, "carbs_g": 0, "protein_g": 31, "fat_g": 3.6, "fiber_g": 0},
    {"food": "Mixed vegetables", "portion_g": 100, "carbs_g": 8, "protein_g": 2, "fat_g": 0.3, "fiber_g": 3},
]
DEMO_GLUCOSE = pd.DataFrame({
    "Minutes from meal": [-30, 0, 30, 60, 90, 120, 150, 180],
    "Glucose (mg/dL)": [108, 112, 132, 158, 151, 169, 151, 139],
})
DEMO_MEALS = [
    {"Meal": "Rice, chicken & vegetables", "Carbs (g)": 50, "Protein (g)": 37, "Fat (g)": 5, "Note": "Synthetic example"},
    {"Meal": "Bean and cheese wrap", "Carbs (g)": 38, "Protein (g)": 16, "Fat (g)": 12, "Note": "Synthetic example"},
    {"Meal": "Oatmeal with berries", "Carbs (g)": 34, "Protein (g)": 8, "Fat (g)": 6, "Note": "Synthetic example"},
]

DEMO_PATIENTS = {
    "Jane Doe": {
        "id": "DEMO-001",
        "age": 28,
        "weight_lb": 145,
        "glucose": 168,
        "target_glucose": 110,
        "carb_ratio": "Not configured",
        "correction_factor": "Not configured",
        "therapy": "Multiple daily injections",
        "note": "Fictional example profile",
    },
    "John Doe": {
        "id": "DEMO-002",
        "age": 35,
        "weight_lb": 178,
        "glucose": 142,
        "target_glucose": 110,
        "carb_ratio": "Not configured",
        "correction_factor": "Not configured",
        "therapy": "Insulin pump",
        "note": "Fictional example profile",
    },
    "Jamie Doe": {
        "id": "DEMO-003",
        "age": 22,
        "weight_lb": 132,
        "glucose": 204,
        "target_glucose": 110,
        "carb_ratio": "Not configured",
        "correction_factor": "Not configured",
        "therapy": "Multiple daily injections",
        "note": "Fictional example profile",
    },
}


st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    [data-testid="stSidebar"] { background: #eef3fa; }
    .hero {
        padding: 1.25rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #285e82);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
    .hero p { color: #e5f2ff; margin-bottom: 0; }
    .small-note { color: #526273; font-size: 0.88rem; }
    div[data-testid="stMetric"] {
        background: white; padding: 0.9rem; border-radius: 12px;
        border: 1px solid #e3eaf2;
    }
</style>
""", unsafe_allow_html=True)

if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "food_rows" not in st.session_state:
    st.session_state.food_rows = pd.DataFrame(DEMO_FOODS)
if "analysis_note" not in st.session_state:
    st.session_state.analysis_note = "Demo meal loaded. Upload a photo to try image analysis."

st.sidebar.markdown("## 💧 GlucoSync")
st.sidebar.caption("Meal-aware diabetes support · prototype")

page = st.sidebar.radio(
    "Navigate",
    ["Patient Demo", "Meal Studio", "My Patterns", "Recipe Library"]
)
st.sidebar.divider()
st.sidebar.info(
    "DEMO MODE\n\nThis prototype uses synthetic glucose and insulin-related examples. "
    "It is not connected to a real CGM or pump and does not recommend insulin doses."
)

@st.cache_resource
def load_food_classifier():
    return pipeline(
        "image-classification",
        model="nateraw/food"
    )


def analyze_photo(image_bytes, mime_type):
    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        classifier = load_food_classifier()
        predictions = classifier(image, top_k=5)

        nutrition = pd.read_csv("food_nutrition.csv")
        nutrition["food_key"] = (
            nutrition["food"]
            .str.lower()
            .str.replace("_", " ", regex=False)
            .str.strip()
        )

        prediction_text = ", ".join(
            f'{p["label"].replace("_", " ")} '
            f'({p["score"]:.0%})'
            for p in predictions
        )

        for prediction in predictions:
            label = prediction["label"].replace(
                "_", " "
            ).lower().strip()

            matches = nutrition[
                nutrition["food_key"] == label
            ]

            if not matches.empty:
                item = matches.iloc[0]

                row = {
                    "food": item["food"],
                    "portion_g": float(item["serving_g"]),
                    "carbs_g": float(item["carbs_g"]),
                    "protein_g": float(item["protein_g"]),
                    "fat_g": float(item["fat_g"]),
                    "fiber_g": float(item["fiber_g"]),
                }

                return [row], (
                    f"Food-101 model suggestion: {label}. "
                    f"Top predictions: {prediction_text}. "
                    "Confirm the food, portion, and nutrients "
                    "before saving."
                )

        return None, (
            "Food recognition completed. Top predictions: "
            f"{prediction_text}. No matching nutrition entry "
            "was found. Keep the sample meal or edit the "
            "food and nutrient values manually."
        )

    except Exception as exc:
        return None, (
            f"Image analysis failed: {type(exc).__name__}. "
            "Check the Terminal error and try again."
        )

def nutrition_totals(df):
    return {
        "carbs": float(df["carbs_g"].sum()),
        "protein": float(df["protein_g"].sum()),
        "fat": float(df["fat_g"].sum()),
        "fiber": float(df["fiber_g"].sum()),
    }

def show_header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )
if page == "Patient Demo":
    show_header(
        "Patient Demo",
        "A simulated patient dashboard for the GlucoSync prototype."
    )

    patient_name = st.selectbox(
        "Choose a demo patient",
        list(DEMO_PATIENTS.keys()),
        index=0,
        key="demo_patient_selector",
    )
    patient = DEMO_PATIENTS[patient_name]

    st.markdown(f"### Welcome, {patient_name}")
    st.caption(
        f"Profile ID: {patient['id']} · {patient['note']}"
    )

    st.divider()
    st.subheader("Patient overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Age", f"{patient['age']} years")
    col2.metric("Weight", f"{patient['weight_lb']} lb")
    col3.metric(
        "Example glucose",
        f"{patient['glucose']} mg/dL"
    )

    col4, col5 = st.columns(2)
    col4.metric(
        "Illustrative target",
        f"{patient['target_glucose']} mg/dL"
    )
    col5.metric("Therapy type", patient["therapy"])

    st.divider()
    st.subheader("Meal context")

    st.write(
        "Use Meal Studio to recognize a food photo, review the "
        "estimated nutrients, and save a meal to the Recipe Library."
    )

    if st.button("Open Meal Studio", type="primary"):
        st.session_state["demo_navigation_request"] = "Meal Studio"
        st.rerun()

    if st.session_state.get("demo_navigation_request") == "Meal Studio":
        st.info(
            "Select Meal Studio from the sidebar to continue "
            "with the meal photo workflow."
        )
        st.session_state["demo_navigation_request"] = None

    st.divider()
    st.subheader("Insulin-method overview")

    st.write(
        "GlucoSync organizes the information that a future, "
        "clinically validated decision-support workflow may need."
    )

    st.markdown("#### 1. Total daily insulin concept")
    st.write(
        "A reference estimate may be discussed in terms of body "
        "weight and a clinician-established treatment plan. "
        "This demo uses standard mock factors for educational simulation."
    )

    st.markdown("#### 2. Basal and mealtime insulin")
    st.write(
        "Basal insulin and mealtime insulin serve different roles. "
        "Their selection and distribution must follow an "
        "individualized treatment plan."
    )

    st.markdown("#### 3. Meal carbohydrate context")
    st.write(
        "The meal-analysis workflow estimates carbohydrates from "
        "visible foods. The estimate must be checked against the "
        "actual serving, ingredients, and nutrition information."
    )

    st.markdown("#### 4. Glucose correction context")
    st.write(
        "A real correction decision requires an established "
        "individual correction factor, target, recent insulin "
        "history, and other clinical considerations."
    )

    st.divider()
    st.subheader("Calculation workflow preview")

    st.write(
        "The following steps demonstrate how information would "
        "move through a future decision-support interface."
    )

   
    current_glucose = float(patient.get('glucose', 150))
    target_glucose = float(patient.get('target_glucose', 100))
    
   
    insulin_to_carb_ratio = 10.0  
    correction_factor = 50.0     

    
    st.markdown("### Interactive Simulation Sandbox")
    st.caption("Adjust the values below to see how the prototype simulates standard clinical math equations.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
       
        default_carbs = st.session_state.get("saved_meal_carbs", 45)
        
        mock_carbs = st.number_input(
            "Simulated Meal Carbs (grams)", 
            min_value=0, 
            max_value=150, 
            value=int(default_carbs), 
            step=5
        )
      
        if "saved_meal_carbs" in st.session_state:
            st.caption("✨ Automatically pulled from your Meal Studio photo analysis!")
            
    with sim_col2:
        mock_glucose = st.number_input("Simulated Current Glucose (mg/dL)", min_value=50, max_value=400, value=int(current_glucose), step=10)


   
    carb_dose = mock_carbs / insulin_to_carb_ratio
    
    if mock_glucose > target_glucose:
        correction_dose = (mock_glucose - target_glucose) / correction_factor
    else:
        correction_dose = 0.0
        
    total_simulated_dose = round(carb_dose + correction_dose, 1)

    workflow = [
        ("01", "Patient context", f"Profile '{patient_name}' loaded successfully."),
        ("02", "Meal analysis", f"Estimated Carbohydrates: {mock_carbs}g (1u per {int(insulin_to_carb_ratio)}g) → **{round(carb_dose, 1)} Units**"),
        ("03", "Glucose context", f"Current: {mock_glucose} mg/dL (Target: {int(target_glucose)} mg/dL)"),
        ("04", "Clinical parameters", f"Correction Factor: 1u drops {int(correction_factor)} mg/dL → **{round(correction_dose, 1)} Units**"),
        ("05", "Calculation output", f"Hypothetical Bolus Recommendation: **{total_simulated_dose} Units**"),
    ]

    for number, title, description in workflow:
        with st.container(border=True):
            left, right = st.columns([1, 4])
            left.markdown(f"### {number}")
            right.markdown(f"**{title}**")
            right.markdown(description)

    st.divider()
    st.subheader("Demo activity")

    st.write(
        f"Current demo profile: **{patient_name}**"
    )
    st.write(
        "Next: open Meal Studio, upload a food photo, review "
        "the estimated nutrition, and save the meal."
    )

elif page == "Meal Studio":
    show_header("Meal Studio", "A clearer picture of what is on your plate.")
    st.write("Upload a photo or start with the built-in sample meal. Review and edit the estimates before saving.")
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        uploaded = st.file_uploader("Upload a meal photo", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded meal photo", use_container_width=True)
            if st.button("✨ Analyze meal photo", type="primary", use_container_width=True):
                with st.spinner("Looking for visible foods…"):
                    rows, note = analyze_photo(uploaded.getvalue(), uploaded.type or "image/jpeg")
                if rows:
                    st.session_state.food_rows = pd.DataFrame(rows)
                    st.session_state.analysis_note = note
                    st.success("Food suggestions generated. Please review all values.")
                else:
                    st.session_state.analysis_note = note
                    st.warning(note)
        if st.button("Load sample meal", use_container_width=True):
            st.session_state.food_rows = pd.DataFrame(DEMO_FOODS)
            st.session_state.analysis_note = "Synthetic sample meal loaded for demonstration."
        st.caption("AI output is a draft, not a measurement. Hidden ingredients and exact portions may be missed.")

    with right:
        st.subheader("Meal details")
        meal_name = st.text_input("Meal name", value="Rice, chicken & vegetables", key="meal_name")
        st.caption(st.session_state.analysis_note)
        edited = st.data_editor(
            st.session_state.food_rows,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "food": st.column_config.TextColumn("Food / ingredient", required=True),
                "portion_g": st.column_config.NumberColumn("Portion (g)", min_value=0, step=5, format="%.1f g"),
                "carbs_g": st.column_config.NumberColumn("Carbs (g)", min_value=0, step=1, format="%.1f"),
                "protein_g": st.column_config.NumberColumn("Protein (g)", min_value=0, step=1, format="%.1f"),
                "fat_g": st.column_config.NumberColumn("Fat (g)", min_value=0, step=1, format="%.1f"),
                "fiber_g": st.column_config.NumberColumn("Fiber (g)", min_value=0, step=1, format="%.1f"),
            },
        )
        totals = nutrition_totals(edited)
        m1, m2 = st.columns(2)
        m1.metric("Carbohydrates", f'{totals["carbs"]:.1f} g')
        m2.metric("Protein", f'{totals["protein"]:.1f} g')
        m3, m4 = st.columns(2)
        m3.metric("Fat", f'{totals["fat"]:.1f} g')
        m4.metric("Fiber", f'{totals["fiber"]:.1f} g')

        st.markdown("#### What this estimate can and cannot tell you")
        st.write(
            "This nutrient summary can support meal logging and later pattern review. "
            "It cannot determine how your glucose will respond or calculate a safe insulin dose."
        )
        if st.button("Save meal to Recipe Library", type="primary", use_container_width=True):
            st.session_state["saved_meal_carbs"] = totals["carbs"]
            
            record = {
                "name": meal_name.strip() or "Untitled meal",
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "items": edited.to_dict(orient="records"),
                "totals": totals,
                "note": "Nutrition estimates should be checked against ingredients and package labels."
            }
            st.session_state.recipes.append(record)
            st.success("Saved to this session's Recipe Library and active profile calculation!")


    st.divider()
    st.subheader("Your next step")
    st.info("After checking the ingredients and portions, save the meal. The Patterns page demonstrates how meal logs can be viewed alongside synthetic CGM data.")

elif page == "My Patterns":
    show_header("My Patterns", "Explore meal timing and glucose trends — with synthetic data.")
    st.warning(
        "All readings below are invented for the demo. This is not a live CGM feed, clinical prediction, "
        "or treatment recommendation."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Demo reading", "139 mg/dL", "synthetic")
    col2.metric("Meal log", "3 examples", "synthetic")
    col3.metric("Device connection", "Not connected")
    st.subheader("Example glucose timeline")
    df = DEMO_GLUCOSE.copy()
    fig = px.line(
        df, x="Minutes from meal", y="Glucose (mg/dL)", markers=True,
        title="Illustrative glucose pattern after a meal",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Minutes relative to meal", yaxis_title="Glucose (mg/dL)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The curve is hand-authored synthetic data, not a model trained on patients.")
    st.subheader("Sample meal log")
    st.dataframe(pd.DataFrame(DEMO_MEALS), use_container_width=True, hide_index=True)
    st.subheader("How personalization could work later")
    st.markdown("""
    - Compare user-confirmed meal logs with their historical CGM patterns.
    - Group similar meals and show the range of observed responses rather than promising an exact forecast.
    - Display data gaps and uncertainty clearly.
    - Use supported device integrations only after permissions, documentation, and safety requirements are addressed.
    """)
    st.info("No insulin dose, correction recommendation, or pump-setting change is generated anywhere in this prototype.")

else:
    show_header("Recipe Library", "Keep meal notes and reviewed ingredient estimates together.")
    if not st.session_state.recipes:
        st.info("No saved recipes in this session yet. Open Meal Studio to save your first meal.")
    else:
        for idx, recipe in enumerate(reversed(st.session_state.recipes)):
            with st.expander(f'{recipe["name"]} · saved {recipe["saved_at"]}'):
                totals = recipe["totals"]
                a, b, c, d = st.columns(4)
                a.metric("Carbs", f'{totals["carbs"]:.1f} g')
                b.metric("Protein", f'{totals["protein"]:.1f} g')
                c.metric("Fat", f'{totals["fat"]:.1f} g')
                d.metric("Fiber", f'{totals["fiber"]:.1f} g')
                st.dataframe(pd.DataFrame(recipe["items"]), use_container_width=True, hide_index=True)
                st.caption(recipe["note"])
    st.caption("Saved recipes are stored only in this browser session and are cleared when the session resets.")

st.sidebar.divider()
st.sidebar.caption("Student prototype · Not for clinical use")
st.sidebar.caption("Avoid entering personal health information or identifiable patient data.")
