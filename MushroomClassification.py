import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Mushroom Edibility Predictor",
    page_icon="🍄",
    layout="centered"
)

friendly_labels = {
    "cap-shape": {
        "Bell": "b", "Conical": "c", "Convex": "x",
        "Flat": "f", "Knobbed": "k", "Sunken": "s"
    },
    "cap-surface": {
        "Fibrous": "f", "Grooves": "g",
        "Scaly": "y", "Smooth": "s"
    },
    "gill-attachment": {
        "Attached": "a", 
        "Free": "f", "Notched": "n"
    },
    "gill-size": {
        "Broad": "b", "Narrow": "n"
    },
    "spore-print-color": {
        "Black": "k", "Brown": "n", "Buff": "b",
        "Chocolate": "h", "Green": "r", "Orange": "o",
        "Purple": "u", "White": "w", "Yellow": "y"
    },
    "veil-color": {
        "Brown": "n", "Orange": "o",
        "White": "w", "Yellow": "y"
    },
    "population": {
        "Abundant": "a", "Clustered": "c", "Numerous": "n",
        "Scattered": "s", "Several": "v", "Solitary": "y"
    },
    "habitat": {
        "Grasses": "g", "Leaves": "l", "Meadows": "m",
        "Paths": "p", "Urban": "u", "Waste": "w", "Woods": "d"
    }
}

# Train Model
@st.cache_resource
def train_model():
    data = pd.read_csv("mushrooms.csv")

    encoding_maps = {}

    for col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

        encoding_maps[col] = {
            label: int(code)
            for label, code in zip(le.classes_, le.transform(le.classes_))
        }

    selected_features = [
        'gill-size',
        'population',
        'habitat',
        'cap-surface',
        'spore-print-color',
        'veil-color',
        'gill-attachment',
        'cap-shape'
    ]

    X = data[selected_features]
    y = data['class']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model, encoding_maps, selected_features

model, encoding_maps, features = train_model()

# Start of UI
st.title("🍄 Mushroom Edibility Predictor")

st.info(
    "👋 Welcome! Select the mushroom characteristics below. "
    "The system will predict whether the mushroom is safe to eat."
)

st.divider()
st.subheader("🔍 Mushroom Characteristics")

user_input = {}
for feature in features:
    options = list(friendly_labels[feature].keys())

    selected_option = st.selectbox(
        feature.replace("-", " ").title(),
        options,
        help=f"Choose the {feature.replace('-', ' ')} of the mushroom"
    )

    user_input[feature] = friendly_labels[feature][selected_option]

if st.button("🍄 Predict Edibility", use_container_width=True):

    encoded_input = []
    for feature in features:
        encoded_input.append(
            encoding_maps[feature][user_input[feature]]
        )

    input_array = np.array(encoded_input).reshape(1, -1)

    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    st.divider()

    if prediction[0] == 0:
        st.balloons()
        st.success("### ✅ This mushroom is **EDIBLE** 🍽️")
        st.write(f"🟢 Confidence: **{probability[0][0]*100:.2f}%**")
    else:
        st.error("### ❌ This mushroom is **POISONOUS** ☠️")
        st.write(f"🔴 Confidence: **{probability[0][1]*100:.2f}%**")

# DISCLAIMER 
st.warning(
    "⚠️ **For Educational purpose only.** "
    "Do NOT use this prediction to decide real mushroom consumption."
)