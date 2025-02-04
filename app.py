import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load the trained model and preprocessing tools
model = tf.keras.models.load_model('fertilizer_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to make predictions
def predict_fertilizer(input_values):
    example = np.array([input_values])
    prediction = model.predict(example)
    predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

    suggestions = {
        '10-10-20 blend': [
            "IFFCO NPK 10:10:20",
            "Coromandel NPK 10:10:20",
            "RCF Suphala 10:10:20",
            "Nagarjuna NPK Fertilizer 10:10:20",
            "Zuari NPK Mixture 10:10:20",
            "Kribhco NPK 10:10:20",
            "Tata Chemicals NPK 10:10:20",
            "Godrej Agrovet NPK 10:10:20",
            "Deepak Fertilizers NPK 10:10:20",
            "Southern Petrochemical Industries Corporation (SPIC) 10:10:20"
        ],
        'Calcium Ammonium Nitrate (CAN)': [
            "YaraLiva™ TROPICOTE™",
            "YaraBela™",
            "Nutramon",
            "Hi-Yield Ammonium Calcium Nitrate",
            "EuroChem CAN 27",
            "Haifa Cal Prime",
            "The Andersons PureCal",
            "Athena PRO CORE",
            "CAN-17",
            "YaraLiva™ NITRABOR™"
        ],
        'Superphosphate': [
            "Single Superphosphate (SSP)",
            "Triple Superphosphate (TSP)",
            "Monoammonium Phosphate (MAP)",
            "Diammonium Phosphate (DAP)",
            "Basic Slag",
            "Calcium Superphosphate",
            "Superphosphate of Lime",
            "Ammonium Polyphosphate",
            "Granulated Superphosphate",
            "High-Analysis Superphosphate"
        ],
        'Potassium sulfate': [
            "Sulfate of Potash (SOP)",
            "Potassium Sulfate Granules",
            "Potassium Sulfate Fertilizer (0-0-50)",
            "K2SO4 (Potassium Sulfate) Crystals",
            "Langbeinite (K2Mg2(SO4)3)",
            "Potassium Magnesium Sulfate",
            "Potassium Sulfate Prills",
            "Liquid Potassium Sulfate",
            "Potassium Sulfate Solution",
            "Granulated Potassium Sulfate"
        ]
    }

    if predicted_class in suggestions:
        return predicted_class, suggestions[predicted_class]
    else:
        return predicted_class, []

# Set the background color
page_bg_color = '''
<style>
[data-testid="stAppViewContainer"] {
    background-color:rgb(53, 97, 53);
}
</style>
'''

st.markdown(page_bg_color, unsafe_allow_html=True)

# Streamlit UI
st.title("Fertilizer Recommendation System")
st.write("Enter the soil parameters below:")

input_values = []
columns = ['Ca (Calcium)', 'Mg (Magnesium)', 'K (Potassium)', 'S (Sulfur)', 'N (Nitrogen)', 'Lime', 'C (Carbon)', 'P (Phosphorus)', 'Moisture']

for col in columns:
    value = st.text_input(f"{col}", "")
    if value:
        try:
            value = float(value)
            if col in ['P (Phosphorus)', 'K (Potassium)', 'S (Sulfur)', 'Mg (Magnesium)']:
                if value >= 0:
                    input_values.append(value)
                else:
                    st.error("Please enter valid number")
            elif col in ['Moisture', 'N (Nitrogen)']:
                input_values.append(value)
            else:
                if value > 0:
                    input_values.append(value)
                else:
                    st.error("Please enter valid number")
        except ValueError:
            st.error("Please enter valid number")
    else:
        st.error("Enter the value")

if st.button('Recommend'):
    if len(input_values) == len(columns):
        input_values = scaler.transform([input_values])
        fertilizer, suggestions = predict_fertilizer(input_values[0])
        st.success(f"Suggested Fertilizer: {fertilizer}")
        st.write("Here are some recommended options:")
        for suggestion in suggestions:
            st.write(f" - {suggestion}")
    else:
        st.error("Please fill in all the values with valid numbers.")

# Stop the Streamlit app gracefully
st.stop()