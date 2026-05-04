import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
import os
import smtplib
from email.message import EmailMessage

# Set Streamlit page config
st.set_page_config(
    page_title="Procrastination Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #2c3e50;
    }
    .subtitle-text {
        color: #7f8c8d;
    }
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 1px solid rgba(0,0,0,.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ML Pipeline & Data Generation
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    # Generate synthetic data for training
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    study_time = np.random.uniform(0, 10, n_samples)
    failures = np.random.randint(0, 5, n_samples)
    absences = np.random.randint(0, 30, n_samples)
    going_out = np.random.randint(1, 6, n_samples)
    screen_time = np.random.uniform(1, 14, n_samples)
    sleep_duration = np.random.uniform(4, 10, n_samples)
    internet_usage = np.random.randint(1, 6, n_samples)
    higher_edu = np.random.choice([0, 1], n_samples) # 0: No, 1: Yes
    
    X = pd.DataFrame({
        'Study_Time': study_time,
        'Failures': failures,
        'Absences': absences,
        'Going_Out': going_out,
        'Screen_Time': screen_time,
        'Sleep_Duration': sleep_duration,
        'Internet_Usage': internet_usage,
        'Higher_Education': higher_edu
    })
    
    # Target variable (synthetic relationship)
    # Higher risk if low study time, high failures, absences, going out, screen time, low sleep
    risk_score = (
        -0.8 * study_time + 
        1.2 * failures + 
        0.1 * absences + 
        0.5 * going_out + 
        0.6 * screen_time - 
        0.4 * sleep_duration + 
        0.3 * internet_usage - 
        1.5 * higher_edu
    )
    
    # Convert score to probabilities and then binary outcome
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.6).astype(int) # High procrastination risk
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    averages = X.mean().to_dict()
    
    return model, scaler, averages

model, scaler, average_values = train_model()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def predict_risk(inputs):
    df = pd.DataFrame([inputs])
    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0][1] # Probability of high risk
    return prob

def get_suggestions(inputs):
    suggestions = []
    if inputs['Study_Time'] < 3:
        suggestions.append("📚 **Study Time is low.** Try the Pomodoro technique (25 min focus, 5 min break) to gradually increase your daily study hours without feeling overwhelmed.")
    if inputs['Screen_Time'] > 6:
        suggestions.append("💻 **Screen Time is high.** Consider installing website blockers during study hours and implement a 'no-screens 1 hour before bed' rule.")
    if inputs['Sleep_Duration'] < 6:
        suggestions.append("😴 **Sleep Duration is suboptimal.** Aim for 7-8 hours. A consistent sleep schedule improves cognitive function and reduces the urge to procrastinate.")
    if inputs['Going_Out'] > 3:
        suggestions.append("🎉 **Frequent Outings detected.** While socializing is important, try to schedule your outings *after* you've completed your critical tasks for the day as a reward.")
    if inputs['Absences'] > 5:
        suggestions.append("🏫 **High Absences.** Missing classes often leads to falling behind, which triggers procrastination due to anxiety. Try to attend classes regularly.")
        
    if not suggestions:
        suggestions.append("🌟 **Great habits!** Keep up the balanced lifestyle, you're on the right track.")
        
    return suggestions

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------

st.markdown("<h1 class='title-text'>🧠 Procrastination Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Analyze your habits, predict your procrastination risk, and get personalized actionable insights.</p>", unsafe_allow_html=True)

# Sidebar for Inputs
with st.sidebar:
    st.header("📊 Your Profile & Habits")
    st.write("Please fill in your current habits below:")
    
    study_time = st.number_input("Study Time (hours/day)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    sleep_duration = st.number_input("Sleep Duration (hours/day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    
    st.divider()
    
    failures = st.number_input("Past Failures (count)", min_value=0, max_value=20, value=0)
    absences = st.number_input("School Absences (count)", min_value=0, max_value=100, value=2)
    
    st.divider()
    
    going_out = st.slider("Going Out Frequency (1=Low, 5=High)", 1, 5, 3)
    internet_usage = st.slider("Internet Usage (1=Low, 5=High)", 1, 5, 4)
    
    higher_edu_input = st.selectbox("Wants to pursue Higher Education?", ["Yes", "No"])
    higher_edu = 1 if higher_edu_input == "Yes" else 0

user_inputs = {
    'Study_Time': study_time,
    'Failures': failures,
    'Absences': absences,
    'Going_Out': going_out,
    'Screen_Time': screen_time,
    'Sleep_Duration': sleep_duration,
    'Internet_Usage': internet_usage,
    'Higher_Education': higher_edu
}

# ---------------------------------------------------------
# Predictions & Metrics
# ---------------------------------------------------------
risk_probability = predict_risk(user_inputs)
risk_percentage = risk_probability * 100

st.divider()
st.subheader("🎯 Procrastination Risk Assessment")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    if risk_percentage < 33:
        status_color = "green"
        status_text = "Low Risk"
        st.success(f"Status: {status_text}")
    elif risk_percentage < 66:
        status_color = "#f1c40f" # Yellow
        status_text = "Medium Risk"
        st.warning(f"Status: {status_text}")
    else:
        status_color = "red"
        status_text = "High Risk"
        st.error(f"Status: {status_text}")
        
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        title = {'text': "Risk Level %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': status_color},
            'steps': [
                {'range': [0, 33], 'color': "rgba(46, 204, 113, 0.2)"},
                {'range': [33, 66], 'color': "rgba(241, 196, 15, 0.2)"},
                {'range': [66, 100], 'color': "rgba(231, 76, 60, 0.2)"}
            ],
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("#### 💡 Personalized Suggestions")
    suggestions = get_suggestions(user_inputs)
    for sug in suggestions:
        st.info(sug)

st.divider()

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
st.subheader("📈 Visual Insights")

tab1, tab2, tab3 = st.tabs(["📊 You vs Average", "📉 Risk vs Screen Time", "🥧 Habit Distribution"])

with tab1:
    # Bar chart: User vs Average
    metrics_to_compare = ['Study_Time', 'Screen_Time', 'Sleep_Duration']
    user_vals = [user_inputs[m] for m in metrics_to_compare]
    avg_vals = [average_values[m] for m in metrics_to_compare]
    
    df_compare = pd.DataFrame({
        'Metric': metrics_to_compare * 2,
        'Value': user_vals + avg_vals,
        'Type': ['You'] * 3 + ['Average'] * 3
    })
    
    fig_bar = px.bar(df_compare, x='Metric', y='Value', color='Type', barmode='group',
                     color_discrete_map={'You': '#3498db', 'Average': '#95a5a6'},
                     title="Your Habits vs Average Student")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Scatter plot: Screen time vs risk (generating synthetic curve for visualization context)
    screen_range = np.linspace(0, 14, 50)
    mock_inputs = user_inputs.copy()
    risks = []
    for s in screen_range:
        mock_inputs['Screen_Time'] = s
        risks.append(predict_risk(mock_inputs) * 100)
        
    fig_line = px.line(x=screen_range, y=risks, labels={'x': 'Screen Time (hours)', 'y': 'Predicted Risk (%)'},
                       title="How Screen Time Affects Your Risk Profile")
    fig_line.add_scatter(x=[user_inputs['Screen_Time']], y=[risk_percentage], mode='markers', 
                         marker=dict(size=12, color='red'), name='You')
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    # Pie chart: Time distribution
    labels = ['Study Time', 'Screen Time', 'Sleep', 'Other']
    other_time = max(0, 24 - (study_time + screen_time + sleep_duration))
    values = [study_time, screen_time, sleep_duration, other_time]
    
    fig_pie = px.pie(values=values, names=labels, title="Your 24-Hour Distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Task Planning & Alerts
# ---------------------------------------------------------
st.subheader("📌 Task Planning & Alerts")
st.write("Plan your upcoming tasks. We'll warn you if you are at high risk of procrastinating them!")

with st.container():
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        task_name = st.text_input("Task Name", placeholder="e.g. Math Assignment")
    with col_t2:
        deadline = st.date_input("Deadline")
    with col_t3:
        email = st.text_input("Email for Alerts", placeholder="student@example.com")
        
    if st.button("Save Task", type="primary"):
        if task_name and email:
            # Check deadline proximity
            days_to_deadline = (deadline - datetime.date.today()).days
            
            # Alert Logic
            if risk_percentage > 66 and days_to_deadline <= 3:
                st.toast("High Risk & Approaching Deadline!", icon="🚨")
                st.error("⚠️ **CRITICAL WARNING:** Your procrastination risk is HIGH and this deadline is less than 3 days away. Start immediately!")
                
                # Attempt to send actual email
                try:
                    sender_email = st.secrets["EMAIL_ADDRESS"]
                    sender_password = st.secrets["EMAIL_PASSWORD"]
                    
                    if sender_email == "your_email@gmail.com" or sender_password == "your_app_password_here":
                        st.warning("⚠️ **Email Setup Required**: You need to update `.streamlit/secrets.toml` with your real email and App Password to receive emails.")
                    else:
                        msg = EmailMessage()
                        msg.set_content(f"Warning! Your task '{task_name}' is due on {deadline}.\n\nYour current procrastination risk is calculated at {risk_percentage:.2f}%. Please start working on this immediately to avoid missing your deadline.")
                        msg['Subject'] = f"Urgent: Start working on {task_name}"
                        msg['From'] = sender_email
                        msg['To'] = email
                        
                        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                            smtp.login(sender_email, sender_password)
                            smtp.send_message(msg)
                            
                        st.success(f"📧 **Alert Sent!** An email has been sent successfully to **{email}**.")
                        
                except Exception as e:
                    st.error(f"Failed to send email. Error: {e}")
                    
            else:
                st.success("Task saved successfully!")
                
            # Save to CSV
            task_data = pd.DataFrame([{
                'Task': task_name,
                'Deadline': deadline,
                'Email': email,
                'Risk_Percentage': round(risk_percentage, 2)
            }])
            
            csv_file = 'tasks.csv'
            if not os.path.exists(csv_file):
                task_data.to_csv(csv_file, index=False)
            else:
                task_data.to_csv(csv_file, mode='a', header=False, index=False)
                
            # Read and display recent tasks
            st.write("#### Saved Tasks")
            df_tasks = pd.read_csv(csv_file)
            st.dataframe(df_tasks, use_container_width=True)
            
        else:
            st.warning("Please provide both Task Name and Email.")
