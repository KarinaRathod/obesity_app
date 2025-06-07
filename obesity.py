import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import numpy as np

# --- Configuration for theming and custom CSS ---
# Custom CSS for full page background and sidebar styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F2F6; /* Light grey background for the main app */
    }
    /* Target the sidebar container by its class name (may change with Streamlit updates) */
    .st-emotion-cache-16txt4v {
        background-color: #FFFFFF; /* White sidebar background */
        border-right: 1px solid #E0E0E0; /* Subtle border for separation */
        box-shadow: 2px 0 5px rgba(0,0,0,0.05); /* Soft shadow for depth */
    }
    /* Target the main content area by its class name (may change with Streamlit updates) */
    .st-emotion-cache-10q0tf1 {
        background-color: #F0F2F6; /* Light grey background for main content */
    }
    /* Style for the main title */
    h1 {
        color: #2C3E50; /* Dark blue-grey for main headings */
    }
    /* Style for subheaders */
    h2, h3, h4 {
        color: #34495E; /* Slightly lighter blue-grey for subheadings */
    }
    /* General styling for Streamlit's status messages (success, info, warning, error) */
    /* Note: The class.st-emotion-cache-10trc1u might change with Streamlit updates */
    .st-emotion-cache-10trc1u {
        border-radius: 0.5rem; /* Rounded corners for status boxes */
        padding: 1rem; /* Padding inside status boxes */
        margin-bottom: 1rem; /* Space below status boxes */
    }
    /* Specific style for success messages */
    .st-emotion-cache-10trc1u.st-emotion-cache-10trc1u-success {
        background-color: #D4EDDA; /* Light green for success */
        color: #155724; /* Dark green text */
        border-color: #C3E6CB;
    }
    /* Specific style for info messages */
    .st-emotion-cache-10trc1u.st-emotion-cache-10trc1u-info {
        background-color: #CCE5FF; /* Light blue for info */
        color: #004085; /* Dark blue text */
        border-color: #B8DAFF;
    }
    /* Specific style for warning messages */
    .st-emotion-cache-10trc1u.st-emotion-cache-10trc1u-warning {
        background-color: #FFF3CD; /* Light yellow for warning */
        color: #856404; /* Dark yellow text */
        border-color: #FFECB5;
    }
    /* Specific style for error messages */
    .st-emotion-cache-10trc1u.st-emotion-cache-10trc1u-error {
        background-color: #F8D7DA; /* Light red for error */
        color: #721C24; /* Dark red text */
        border-color: #F5C6CB;
    }
    /* Custom style for the predicted class text to make it stand out */
    .predicted-class-text {
        font-size: 2.5em; /* Larger font size */
        font-weight: bold; /* Bold text */
        color: #28A745; /* A vibrant green for positive emphasis */
        text-align: center; /* Center align the text */
        margin-top: 20px; /* Space above */
        margin-bottom: 20px; /* Space below */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_model_and_preprocessors():
    """
    Loads the trained machine learning model, label encoders, and scaler.
    This function is cached to prevent reloading on every Streamlit rerun.
    """
    try:
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

        # Define features (X) and target (y)
        X = df.drop('NObeyesdad', axis=1)
        y = df['NObeyesdad']

        # Identify categorical columns
        categorical_cols = X.select_dtypes(include='object').columns

        # Initialize and fit LabelEncoders for categorical features
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # At this point, X is entirely numerical.
        # Now, fit the StandardScaler on the entire numerical X DataFrame.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) # This scales ALL columns in X

        # Convert scaled array back to DataFrame to retain column names for features_order
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Encode the target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        label_encoders['NObeyesdad'] = le_target

        # Split the scaled data into training and testing sets
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_df, y_encoded, test_size=0.2, random_state=42)

        # Train the Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # The features_order should be the column names of X after encoding, which were also scaled.
        features_order = X_scaled_df.columns.tolist() # This is the correct features_order

        return model, label_encoders, scaler, features_order

    except FileNotFoundError:
        st.error("Error: 'ObesityDataSet_raw_and_data_sinthetic.csv' not found. Please ensure the dataset is in the same directory as this Streamlit app.")
        st.stop() # Stop the app execution if essential files are missing

model, label_encoders, scaler, features_order = load_model_and_preprocessors()

# Define the target classes for decoding the numerical predictions back to human-readable labels
target_classes = label_encoders['NObeyesdad'].classes_

# --- Health Tips Dictionary ---
health_tips = {
    'Insufficient_Weight': """
        **Focus on healthy weight gain:**
        * **Balanced Diet:** Increase calorie intake from nutrient-dense foods like whole grains, lean proteins, healthy fats (avocado, nuts), and dairy.
        * **Frequent Meals:** Eat smaller, more frequent meals throughout the day.
        * **Strength Training:** Incorporate strength training exercises to build muscle mass.
        * **Consult a Professional:** If you're struggling to gain weight, consider consulting a dietitian or doctor.
    """,
    'Normal_Weight': """
        **Great job maintaining a healthy weight!**
        * **Continue Healthy Habits:** Keep up your balanced diet, regular physical activity, and adequate hydration.
        * **Variety is Key:** Ensure a diverse intake of fruits, vegetables, and whole foods.
        * **Listen to Your Body:** Pay attention to hunger and fullness cues.
        * **Stress Management:** Manage stress effectively, as it can impact overall health.
    """,
    'Overweight': """
        **Focus on gradual, sustainable changes:**
        * **Portion Control:** Be mindful of portion sizes to manage calorie intake.
        * **Increase Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic activity per week.
        * **Hydration:** Drink plenty of water throughout the day.
        * **Limit Processed Foods:** Reduce consumption of sugary drinks, fast food, and highly processed snacks.
        * **Professional Guidance:** Consider speaking with a healthcare provider or a registered dietitian for personalized advice.
    """,
    'Obesity_Type_I': """
        **It's time to prioritize your health with structured efforts:**
        * **Dietary Changes:** Focus on a balanced diet rich in vegetables, fruits, lean proteins, and whole grains. Limit saturated fats, added sugars, and refined carbohydrates.
        * **Regular Exercise:** Gradually increase your physical activity. Start with brisk walking and aim for consistency.
        * **Behavioral Changes:** Address eating habits, stress, and sleep patterns that might contribute to weight gain.
        * **Medical Consultation:** Strongly recommended to consult a doctor or a specialist for a comprehensive health plan, which may include dietary plans, exercise routines, and medical supervision.
    """,
    'Obesity_Type_II': """
        **Serious and consistent intervention is crucial for your health:**
        * **Comprehensive Medical Review:** Seek immediate consultation with a healthcare team (doctor, dietitian, exercise physiologist) to develop a personalized weight management plan.
        * **Structured Diet Plan:** Adhere to a professionally guided low-calorie, nutrient-dense diet.
        * **Supervised Exercise:** Engage in physical activity under professional guidance to ensure safety and effectiveness.
        * **Support Systems:** Consider joining support groups or seeking psychological counseling to address emotional eating or other contributing factors.
        * **Potential Medical Interventions:** Discuss options such as medication or bariatric surgery with your doctor if appropriate.
    """,
    'Obesity_Type_III': """
        **This is a critical health situation requiring urgent and intensive medical intervention:**
        * **Immediate Medical Care:** It is imperative to work closely with a multidisciplinary medical team, including bariatric specialists, endocrinologists, and dietitians.
        * **Aggressive Lifestyle Modification:** Implement significant changes to diet and physical activity under strict medical supervision.
        * **Medical Management:** Explore all medical treatment options, which may include highly specialized dietary plans, medication, or metabolic/bariatric surgery.
        * **Psychological Support:** Seek comprehensive psychological support to manage the emotional and mental aspects of severe obesity.
        * **Regular Monitoring:** Consistent follow-up and monitoring by healthcare professionals are essential.
    """,
    'Public_Transportation': """
        **Consider incorporating more activity:**
        * If possible, walk or bike to your public transport stop.
        * Stand during your commute instead of sitting.
        * Get off a stop earlier and walk the rest of the way.
    """,
    'Walking': """
        **Excellent! Walking is a great foundation for health.**
        * Vary your walking routes to keep it interesting.
        * Increase your pace or incorporate inclines for a greater challenge.
        * Consider adding short bursts of jogging or bodyweight exercises.
    """,
    'Automobile': """
        **Look for opportunities to move more:**
        * Park further away from your destination.
        * Take the stairs instead of elevators.
        * Break up long periods of sitting with short walks.
        * Consider carpooling less or biking/walking for short errands.
    """,
    'Motorbike': """
        **Integrate more physical activity into your day:**
        * Find time for structured exercise, like walking, jogging, or sports.
        * Use your motorbike less for very short distances where walking or biking is feasible.
        * Focus on daily movement goals beyond just commuting.
    """,
    'Bike': """
        **Fantastic! Biking is an excellent form of exercise.**
        * Explore new biking trails or routes.
        * Increase the intensity or duration of your rides.
        * Consider adding other forms of exercise like strength training for overall fitness.
    """
}


# --- Streamlit App Title and Introduction ---
st.title("Obesity Risk Prediction 📊🧔‍♂️🍔⚖️🧍‍♂️")
st.markdown("""
    Welcome to the **Obesity Risk Prediction** application. This tool helps you assess your potential risk of obesity
    based on various lifestyle and demographic factors. Please enter your details in the sidebar to get a prediction.
    
    ---
    
    *Disclaimer: This application is for informational purposes only and should not be used as a substitute for professional medical advice.*
""")

# --- Input Form in Sidebar ---
# All primary input widgets are placed in the sidebar for a clean main content area [3, 9, 4, 10]
with st.sidebar:
    st.header("Enter Your Details")
    st.markdown("Fill in the details below to get your personalized obesity risk prediction.")

    # Use st.form to batch user inputs, preventing constant reruns [2]
    with st.form("obesity_prediction_form"):
        st.subheader("Personal Information")
        # st.radio for binary/few options, st.number_input for precise numbers [1]
        gender = st.radio("Gender", ['Female', 'Male'], help="Select your biological gender.", index=0)
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=25, help="Your age in years.", format="%d")
        height = st.number_input("Height (meters)", min_value=0.50, max_value=2.50, value=1.70, step=0.01, format="%.2f", help="Your height in meters (e.g., 1.75).")
        weight = st.number_input("Weight (kilograms)", min_value=10.0, max_value=300.0, value=70.0, step=0.1, format="%.1f", help="Your weight in kilograms (e.g., 70.5).")

        st.subheader("Lifestyle & Habits")
        family_history_with_overweight = st.radio("Family history of overweight?", ['yes', 'no'], help="Do close family members have a history of overweight or obesity?", index=0)
        favc = st.radio("Frequent consumption of high caloric food?", ['yes', 'no'], help="Do you frequently consume high-caloric foods (e.g., fast food, sugary drinks)?", index=1)
        # st.slider for selecting from a range [1]
        fcvc = st.slider("Frequency of consumption of vegetables", 1, 3, value=2, help="1: Never, 2: Sometimes, 3: Always")
        ncp = st.slider("Number of main meals per day", 1, 4, value=3, help="How many main meals do you typically have per day?")
        # st.selectbox for single selection from a list [1]
        caec = st.selectbox("Consumption of food between meals", ['no', 'Sometimes', 'Frequently', 'Always'], help="How often do you eat between meals?", index=1)
        smoke = st.radio("Smoker?", ['yes', 'no'], help="Are you a smoker?", index=1)
        ch2o = st.slider("Consumption of water daily (liters)", 1, 3, value=2, help="1: Less than 1L, 2: 1-2L, 3: More than 2L")
        scc = st.radio("Calories consumption monitoring?", ['yes', 'no'], help="Do you monitor your calorie intake?", index=1)
        faf = st.slider("Physical activity frequency (days/week)", 0, 3, value=1, help="0: Never, 1: 1-2 days/week, 2: 2-4 days/week, 3: 4-5 days/week")
        tue = st.slider("Time using technology devices (hours/day)", 0, 2, value=1, help="0: 0-2 hours, 1: 3-5 hours, 2: More than 5 hours")
        calc = st.selectbox("Consumption of alcohol", ['no', 'Sometimes', 'Frequently', 'Always'], help="How often do you consume alcohol?", index=0)
        mtrans = st.selectbox("Transportation used", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'], help="What is your primary mode of transportation?", index=0)

        st.markdown("---") # Visual separator
        submit_button = st.form_submit_button("Predict Obesity Risk") # The button to submit the form [1, 2]

# --- Main Content Area (Tabs) ---
# Organize content into distinct tabs for better navigation and reduced clutter [3, 4]
tab1, tab2 = st.tabs(["Prediction & Health Tips", "About"])

with tab1:
    if submit_button:
        # Display a spinner while the prediction is being calculated [5]
        with st.spinner("Analyzing your data and calculating risk..."):
            # Create a DataFrame from user inputs, ensuring correct column order
            input_data_dict = {
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': family_history_with_overweight,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }
            input_df = pd.DataFrame([input_data_dict]) # Removed columns=features_order here for initial DataFrame creation

            # Store original input for display later (before encoding/scaling)
            original_input_for_display = input_df.copy()

            # Apply label encoding to categorical features in the input DataFrame
            categorical_cols_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
            for col in categorical_cols_to_encode:
                if col in label_encoders and col in input_df.columns:
                    try:
                        input_df.loc[:, col] = label_encoders[col].transform(input_df[col])
                    except ValueError as e:
                        st.error(f"Error encoding '{col}': {e}. Please check your input.")
                        st.stop()


            # Ensure all columns are numeric
            for col in input_df.columns:
                input_df.loc[:, col] = pd.to_numeric(input_df[col], errors='coerce')
            
            # CRITICAL FIX: Reorder input_df columns to match the order used during training (features_order)
            # This ensures the scaler receives columns in the expected sequence and with correct names.
            input_df_ordered = input_df[features_order]

            # Apply StandardScaler to the prepared input DataFrame
            scaled_input = scaler.transform(input_df_ordered)
            
            # Make prediction and get probabilities
            prediction_encoded = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)

            # Decode the numerical prediction back to its original class name
            predicted_class = label_encoders['NObeyesdad'].inverse_transform(prediction_encoded)[0]

            st.success("Prediction Complete! 🎉") # Success message [5]

            st.subheader("Your Predicted Obesity Risk Category:")
            # Display the main prediction prominently using custom CSS for emphasis [1, 5]
            st.markdown(f"<p class='predicted-class-text'>{predicted_class.replace('_', ' ')}</p>", unsafe_allow_html=True)

            st.subheader("Prediction Probabilities:")
            # Create a DataFrame for probabilities to be used in Plotly chart
            proba_df = pd.DataFrame({
                'Category': target_classes,
                'Probability': prediction_proba[0] # Access the first (and only) row of probabilities
            }).sort_values(by='Probability', ascending=False)

            # Visualize prediction probabilities using a bar chart [1, 6, 7]
            fig = px.bar(proba_df, x='Category', y='Probability',
                         title='Probability Distribution Across Categories',
                         labels={'Probability': 'Probability', 'Category': 'Obesity Category'},
                         color='Probability', # Color bars based on probability value
                         color_continuous_scale=px.colors.sequential.Viridis, # Choose a color scale
                         template="plotly_white") # Use a clean white background template for the chart
            fig.update_layout(xaxis_title="Obesity Category", yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True) # Display the interactive Plotly chart

            st.subheader("Personalized Health Tips:")
            st.info(f"""
                Based on your predicted category, here are some health tips tailored for you:
                {health_tips.get(predicted_class, "No specific tips available for this category.")}
                
                Also, considering your primary mode of transportation is **{mtrans.replace('_', ' ')}**:
                {health_tips.get(mtrans, "No specific tips available for this transportation method.")}
            """)


            st.subheader("Understanding Your Input Data")
            # Use an expander to hide raw input data, reducing clutter [8, 4]
            with st.expander("Click to View Your Entered Data"):
                st.dataframe(original_input_for_display) # Display the original, un-processed input data [1, 5]

            st.info("""
                **Interpretation:** The bar chart above illustrates the model's confidence in classifying you into different obesity categories.
                The category with the highest bar represents the most probable outcome based on your inputs.
                
                **Next Steps:** If your predicted category indicates overweight or obesity, it is highly recommended
                to consult a healthcare professional for personalized advice and a comprehensive health assessment.
                Remember, this tool provides a statistical estimation and is not a substitute for medical diagnosis.
            """) # Informational message [1, 5]

    else:
        # Initial message when the app loads or before submission
        st.info("Please enter your details in the sidebar and click 'Predict Obesity Risk' to get started. Your privacy is important; no data is stored.")

with tab2:
    st.header("About the Model")
    st.markdown("""
        This application leverages a **Logistic Regression** machine learning model to predict obesity risk.
        The model was trained on a comprehensive synthetic dataset (`ObesityDataSet_raw_and_data_sinthetic.csv`)
        that includes various demographic and lifestyle factors.
    """)

    # Use expanders to provide detailed information without overwhelming the user [8, 4]
    with st.expander("Model Details and Methodology"):
        st.markdown("""
            The predictive model undergoes several key steps to process data and make accurate predictions:
            
            **1. Data Preprocessing:**
            - **Categorical Encoding:** Features like 'Gender', 'Family History', and 'Transportation' are converted into numerical formats using `LabelEncoder`.
            - **Feature Scaling:** After categorical features are encoded, all numerical features (both original and newly encoded categorical ones) are scaled using `StandardScaler`. This ensures that all features contribute equally to the model's learning process.
            
            **2. Model Training:**
            - The preprocessed and scaled data is split into training and testing sets (80% for training, 20% for testing).
            - A **Logistic Regression** model is then trained on the training data. This algorithm is chosen for its interpretability and effectiveness in classification tasks, providing probability scores for each possible outcome.
            
            **3. Model Persistence:**
            - The trained model, along with the `LabelEncoder` objects and the `StandardScaler`, are managed within the Streamlit application's `@st.cache_resource` decorator, eliminating the need for external `.pkl` files.
        """)

    with st.expander("Limitations and Important Disclaimers"):
        st.warning("""
            **Important Limitations:**
            - **Synthetic Data:** The model is trained on a *synthetic* dataset. While designed to simulate real-world scenarios, it may not fully capture the intricate complexities and rare patterns present in actual human health data.
            - **Generalization:** The model's predictions are based on patterns observed in its training data. Its accuracy might vary when applied to individuals with characteristics significantly different from those in the dataset.
            - **Correlation vs. Causation:** This model identifies statistical correlations between input factors and obesity categories. It does not establish a causal relationship.
            - **Unforeseen Factors:** The model cannot account for all possible factors influencing obesity, such as specific genetic predispositions, underlying medical conditions, or environmental influences not included in the input features.
            - **Dynamic Health:** Human health is dynamic. This model provides a snapshot based on current inputs and does not track changes over time or respond to interventions.
            
            **Crucial Disclaimer:**
            This application is developed for **educational and informational purposes only**. It is **not** a medical device, a diagnostic tool, or a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this application. Reliance on any information provided by this application is solely at your own risk.
        """)