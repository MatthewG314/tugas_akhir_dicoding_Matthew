import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
day_data = pd.read_csv("day.csv")
hourly_data = pd.read_csv("hour.csv") 

# Sidebar : Count/Prediction
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type:",
    options=["Rental Count Analysis", "Prediction Analysis"]
)

# Select Year
year_options = st.sidebar.multiselect(
    "Select Year(s) to Analyze:",
    options=[2011, 2012],
    default=[2011, 2012]
)

# Filter data
filtered_day_data = day_data[day_data['yr'].isin([year - 2011 for year in year_options])]
filtered_hourly_data = hourly_data[hourly_data['yr'].isin([year - 2011 for year in year_options])]

# Main Dashboard: Title
st.title("Bike Sharing Data Analysis Dashboard")

#----------------------------------------------------------------------------------------------------------------
#RENTAL COUNT

if analysis_type == "Rental Count Analysis":
    st.header("Rental Count Analysis")
    st.write("### Adjust Temperature and Humidity")

    # Temperature and Humidity Sliders
    temp_range = st.slider(
        "Select Temperature Range",
        min_value=float(filtered_day_data['temp'].min()), 
        max_value=float(filtered_day_data['temp'].max()), 
        value=(float(filtered_day_data['temp'].min()), float(filtered_day_data['temp'].max()))
    )

    humidity_range = st.slider(
        "Select Humidity Range",
        min_value=float(filtered_day_data['hum'].min()), 
        max_value=float(filtered_day_data['hum'].max()), 
        value=(float(filtered_day_data['hum'].min()), float(filtered_day_data['hum'].max()))
    )

    # Temperature and humidity ranges
    filtered_data = filtered_day_data[
        (filtered_day_data['temp'] >= temp_range[0]) &
        (filtered_day_data['temp'] <= temp_range[1]) &
        (filtered_day_data['hum'] >= humidity_range[0]) &
        (filtered_day_data['hum'] <= humidity_range[1])
    ]

    col1, col2 = st.columns(2)
    with col1:
    # Rental Count by Season
        st.subheader("Average Rental Count by Season")
        season_data = filtered_data.groupby("season")["cnt"].mean().reset_index()
        season_data["season"] = season_data["season"].map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})

        fig1, ax1 = plt.subplots()
        sns.barplot(data=season_data, x="season", y="cnt", palette="viridis", ax=ax1)
        ax1.set_title("Average Rental Count by Season")
        ax1.set_xlabel("Season")
        ax1.set_ylabel("Average Count")
        st.pyplot(fig1)

    with col2:
    # Effect of Weather Conditions on Rental Count
        st.subheader("Effect of Weather Conditions on Rental Count")
        weather_data = filtered_data.copy()
        weather_data["weathersit"] = weather_data["weathersit"].map({1: "Clear/Partly Cloudy", 2: "Mist/Cloudy", 3: "Light Snow/Light Rain", 4: "Heavy Rain/Heavy Snow"})

        fig3, ax3 = plt.subplots()
        sns.boxplot(data=weather_data, x="weathersit", y="cnt", palette="coolwarm", ax=ax3)
        ax3.set_title("Effect of Weather Conditions on Rental Count")
        ax3.set_xlabel("Weather Condition")
        ax3.set_ylabel("Rental Count")
        st.pyplot(fig3)

    # Outlier for Rental Count by Weather Conditions
        outliers = []
        for condition in weather_data["weathersit"].unique():
            condition_data = weather_data[weather_data["weathersit"] == condition]
            Q1 = condition_data["cnt"].quantile(0.25)
            Q3 = condition_data["cnt"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
            condition_outliers = condition_data[(condition_data["cnt"] < lower_bound) | (condition_data["cnt"] > upper_bound)]
            outliers.append(condition_outliers)
        outlier_data = pd.concat(outliers)

    # Display the outlier data
        if not outlier_data.empty:
            st.write("Detected Outliers in Rental Count:")
            st.write(outlier_data[["dteday", "weathersit", "cnt"]])
        else:
            st.write("No outliers detected for the selected conditions.")

    # Rental Count by Hour
    st.subheader("Average Rental Count by Hour")
    hourly_avg = filtered_hourly_data.groupby("hr")["cnt"].mean().reset_index()

    fig2, ax2 = plt.subplots()
    sns.lineplot(data=hourly_avg, x="hr", y="cnt", marker='o', ax=ax2)
    ax2.set_title("Average Rental Count by Hour of the Day")
    ax2.set_xlabel("Hour of the Day")
    ax2.set_ylabel("Average Count")
    st.pyplot(fig2)
#-----------------------------------------------------------------------------------------------------------------------------------------
#PREDICTION ANALYSIS

elif analysis_type == "Prediction Analysis":
    prediction_type = st.sidebar.selectbox(
        "Select Prediction Type:",
        options=["Daily Prediction", "Hourly Prediction"]
    )

    # Prediction type : Daily predicition or Hourly prediction
    if prediction_type == "Daily Prediction":
        data = filtered_day_data
        feature_options = ["temp", "hum", "windspeed", "season", "weathersit", "holiday", "workingday"]
    else:
        data = filtered_hourly_data
        feature_options = ["hr", "temp", "atemp", "hum", "windspeed", "season", "weathersit", "holiday", "workingday"]

    st.subheader("Bike Rental Prediction Analysis")

    st.write("### Condition")
    selected_features = st.multiselect(
        "Select Conditions to Include in the Prediction Model:",
        options=feature_options,
        default=feature_options  # Set all condition for input
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one condition.")
    else:
        X = data[selected_features]
        y = data["cnt"]

        # Polynomial prediction
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        poly = PolynomialFeatures(degree=3) 
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)
        
        reg_model = LinearRegression()
        reg_model.fit(X_poly_train, y_train)

        y_pred_test = reg_model.predict(X_poly_test)

        # Create input sliders for making new predictions
        st.subheader("Input conditions")
        col1, col2 = st.columns([2, 1])  # Create two columns for inputs and results

        # Collect user inputs for each selected feature
        input_values = []
        with col1:
            for feature in selected_features:
                if feature == "hr" and prediction_type == "Hourly Prediction":
                    hr_input = ("Hour of the Day", 0, 23, 12)  # Default hour to noon
                    input_values.append(hr_input)
                elif feature == "temp":
                    temp_input = st.slider("Temperature (Normalized)", float(data['temp'].min()), float(data['temp'].max()), 0.5)
                    input_values.append(temp_input)
                elif feature == "atemp" and prediction_type == "Hourly Prediction":
                    atemp_input = st.slider("Feeling Temperature (Normalized)", float(data['atemp'].min()), float(data['atemp'].max()), 0.5)
                    input_values.append(atemp_input)
                elif feature == "hum":
                    humidity_input = st.slider("Humidity (Normalized)", float(data['hum'].min()), float(data['hum'].max()), 0.5)
                    input_values.append(humidity_input)
                elif feature == "windspeed":
                    windspeed_input = st.slider("Windspeed (Normalized)", float(data['windspeed'].min()), float(data['windspeed'].max()), 0.1)
                    input_values.append(windspeed_input)
                elif feature == "season":
                    season_input = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"])
                    season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
                    input_values.append(season_map[season_input])
                elif feature == "weathersit":
                    weather_input = st.selectbox("Weather Situation", options=["Clear", "Mist", "Light Snow/Rain", "Heavy Rain/Snow"])
                    weather_map = {"Clear": 1, "Mist": 2, "Light Snow/Rain": 3, "Heavy Rain/Snow": 4}
                    input_values.append(weather_map[weather_input])
                elif feature == "holiday":
                    holiday_input = st.selectbox("Holiday", options=["No", "Yes"])
                    holiday_map = {"No": 0, "Yes": 1}
                    input_values.append(holiday_map[holiday_input])
                elif feature == "workingday":
                    workingday_input = st.selectbox("Day Type", options=["Holiday", "Working Day"])
                    workingday_map = {"Holiday": 0, "Working Day": 1}
                    input_values.append(workingday_map[workingday_input])

 
        if prediction_type == "Hourly Prediction":
            hours = range(24)

            input_samples = pd.DataFrame([input_values] * 24, columns=selected_features)
            input_samples["hr"] = hours  

            if "hr" not in X.columns:
                input_samples_poly = poly.transform(input_samples)
            else:
                X_train["hr"] = filtered_hourly_data["hr"]
                input_samples_poly = poly.transform(input_samples)

            # Predict rental count for each hour
            predicted_counts = reg_model.predict(input_samples_poly)
            predicted_counts = np.maximum(predicted_counts, 0) 
            predicted_counts = np.ceil(predicted_counts)

            # Display hourly predictions in a table format
            with col2:
                hourly_prediction_df = pd.DataFrame({"Hour": hours, "Predicted Rental": predicted_counts})
                st.write("### Hourly Predictions")
                st.write(hourly_prediction_df)

            st.subheader("Hourly Predicted Rental Count")
            fig, ax = plt.subplots()
            sns.lineplot(data=hourly_prediction_df, x="Hour", y="Predicted Rental", marker='o', ax=ax)
            ax.set_title("Hourly Predicted Rental Count")
            ax.set_xlabel("Hour of the Day")
            ax.set_ylabel("Predicted Rental Count")
            st.pyplot(fig)

        # For Daily Prediction
        else:  
            input_data = [input_values]  
            input_data_poly = poly.transform(input_data) 

            predicted_count = reg_model.predict(input_data_poly)[0]
            predicted_count = max(predicted_count, 0)  
            predicted_count = np.ceil(predicted_count) 

            # Display the predicted rental count
            with col2:
                st.write(f"**Predicted Daily Rental Count:** {predicted_count:.0f}")

