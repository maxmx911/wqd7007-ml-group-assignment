import streamlit as st
import pandas as pd
from pickle import load
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import plotly.express as px
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="Career Change Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .model-name {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-yes {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .prediction-no {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .probability-text {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data configuration
@st.cache_data
def load_config():
    return {
        'field_of_study_selection': ['Medicine', 'Education', 'Computer Science', 'Business',
                                    'Mechanical Engineering', 'Biology', 'Law', 'Economics', 'Psychology'],
        'current_occupation_selection': ['Business Analyst', 'Economist', 'Biologist', 'Doctor', 'Lawyer',
                                        'Software Developer', 'Artist', 'Psychologist', 'Teacher', 'Mechanical Engineer'],
        'field_of_study_one_hot_columns': ['field_of_study_Biology', 'field_of_study_Business',
                                          'field_of_study_Computer Science', 'field_of_study_Economics',
                                          'field_of_study_Education', 'field_of_study_Law',
                                          'field_of_study_Mechanical Engineering', 'field_of_study_Medicine',
                                          'field_of_study_Psychology'],
        'current_occupation_one_hot_columns': ['current_occupation_Biologist', 'current_occupation_Business Analyst',
                                              'current_occupation_Doctor', 'current_occupation_Economist',
                                              'current_occupation_Lawyer', 'current_occupation_Mechanical Engineer',
                                              'current_occupation_Psychologist', 'current_occupation_Software Developer',
                                              'current_occupation_Teacher'],
        'gender_selection': ['Male', 'Female'],
        'education_level_selection': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
        'industry_growth_rate_selection': ['Low', 'Medium', 'High'],
        'family_influence_selection': ['None', 'Low', 'Medium', 'High']
    }

# Load models and data
@st.cache_resource
def load_models_and_data():
    try:
        std_scaler = load(open('std_scaler.pkl', 'rb'))
        model_dict = {
            'top1': load(open('top1_xgb_model_3.pkl', 'rb')),
            'top2': load(open('top2_rf_model_1.pkl', 'rb')),
            # 'top3': load(open('top3_svm_model_3.pkl', 'rb')),
        }
        
        custom_na_filter = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', 
                           '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        df_data = pd.read_csv('career_change_prediction_dataset_dirty_missing.csv', 
                             keep_default_na=False, na_values=custom_na_filter)
        
        return std_scaler, model_dict, df_data
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None, None

# Model configuration
MODEL_CONFIG = {
    'top1': {
        'name': 'XGBoost Model',
        'description': ' Top 1 model',
        'input_cols': ['job_satisfaction', 'salary', 'field_of_study'],
        'processed_cols': ['job_satisfaction', 'salary', 'field_of_study_Medicine', 'field_of_study_Computer Science']
    },
    'top2': {
        'name': 'Random Forest Model',
        'description': ' Top 2 model',
        'input_cols': [],  # Empty list indicates all fields are editable
        'processed_cols': []
    },
    # 'top3': {
    #     'name': 'SVM Model',
    #     'description': ' Top 3 model',
    #     'input_cols': ['job_satisfaction', 'salary', 'field_of_study'],
    #     'processed_cols': ['job_satisfaction', 'salary', 'field_of_study_Medicine', 'field_of_study_Computer Science']
    # }
}

def encode_categorical_columns(df, config):
    binary_category_cols = [
        'gender', 'mentorship_available', 'certifications', 
        'freelancing_experience', 'geographic_mobility', 'likely_to_change_occupation'
    ]
    
    multi_category_cols = ['field_of_study', 'current_occupation']
    
    ordinal_category_order_mapping = {
        'education_level': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
        'industry_growth_rate': ['Low', 'Medium', 'High'],
        'family_influence': ['None', 'Low', 'Medium', 'High']
    }
    
    # Ordinal Encoding
    for col, order in ordinal_category_order_mapping.items():
        if col in df.columns:
            enc = OrdinalEncoder(categories=[order])
            df[[col]] = enc.fit_transform(df[[col]])
    
    # Label Encoding for binary columns
    for col in binary_category_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # One-Hot Encoding for multi-category columns
    for col in multi_category_cols:
        if col not in df.columns:
            continue
            
        if col == 'current_occupation':
            current_occupation_one_hot_dict = dict.fromkeys(config['current_occupation_one_hot_columns'], 0)
            dummy = pd.get_dummies(df[col], prefix=col)
            if len(dummy.columns) > 0 and dummy.columns[0] in current_occupation_one_hot_dict.keys():
                current_occupation_one_hot_dict[dummy.columns[0]] = 1
                df = pd.concat([df, pd.DataFrame([current_occupation_one_hot_dict])], axis=1)
                df.drop(columns=[col], inplace=True)
                
        elif col == 'field_of_study':
            field_of_study_one_hot_dict = dict.fromkeys(config['field_of_study_one_hot_columns'], 0)
            dummy = pd.get_dummies(df[col], prefix=col)
            if len(dummy.columns) > 0 and dummy.columns[0] in field_of_study_one_hot_dict.keys():
                field_of_study_one_hot_dict[dummy.columns[0]] = 1
                df = pd.concat([df, pd.DataFrame([field_of_study_one_hot_dict])], axis=1)
                df.drop(columns=[col], inplace=True)
    
    return df

def preprocessing(df, processed_columns_to_keep, std_scaler, config):
    # Rename columns
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    
    # Encode categorical columns
    df = encode_categorical_columns(df, config)
    
    # Standardize numerical columns
    numerical_columns = ['age', 'years_of_experience', 'job_satisfaction', 'work_life_balance',
                        'job_opportunities', 'salary', 'job_security', 'skills_gap',
                        'professional_networks', 'career_change_events', 'technology_adoption']
    ordinal_category_cols = ['education_level', 'industry_growth_rate', 'family_influence']
    
    # Filter existing columns
    numerical_columns = [col for col in numerical_columns if col in df.columns]
    ordinal_category_cols = [col for col in ordinal_category_cols if col in df.columns]
    
    if numerical_columns + ordinal_category_cols:
        df[numerical_columns + ordinal_category_cols] = std_scaler.transform(df[numerical_columns + ordinal_category_cols])
    
    if processed_columns_to_keep:
        df = df[processed_columns_to_keep]
    
    return df

# Initialize session state
def init_session_state():
    required_fields = {
        'age': 30,
        'gender': 'Male',
        'years_of_experience': 5,
        'education_level': "Bachelor's",
        'industry_growth_rate': "Medium",
        'job_satisfaction': 5,
        'work_life_balance': 5,
        'job_opportunities': 50,
        'salary': 60000,
        'job_security': 5,
        'skills_gap': 3,
        'family_influence': "None",
        'mentorship_available': "No",
        'certifications': "No",
        'freelancing_experience': "No",  # New field initialization
        'geographic_mobility': "No",
        'professional_networks': 5,
        'career_change_events': 0,
        'technology_adoption': 7,
        'field_of_study': "Computer Science",
        'current_occupation': "Software Developer",
        'career_change_events': 0
    }

    for field, default in required_fields.items():
        if field not in st.session_state:
            st.session_state[field] = default

def display_editable_profile_summary(input_columns_to_keep, enable_all_fields):
    """
    Displays an optimized user profile summary, showing only editable fields
    """
    # Define mappings for all fields and display information
    field_mapping = {
        'age': {'display_name': 'Age', 'value': st.session_state['age'], 'unit': ''},
        'gender': {'display_name': 'Gender', 'value': st.session_state['gender'], 'unit': ''},
        'years_of_experience': {'display_name': 'Experience', 'value': st.session_state['years_of_experience'], 'unit': ' years'},
        'education_level': {'display_name': 'Education', 'value': st.session_state['education_level'], 'unit': ''},
        'field_of_study': {'display_name': 'Field of Study', 'value': st.session_state['field_of_study'], 'unit': ''},
        'current_occupation': {'display_name': 'Current Role', 'value': st.session_state['current_occupation'], 'unit': ''},
        'industry_growth_rate': {'display_name': 'Industry Growth', 'value': st.session_state['industry_growth_rate'], 'unit': ''},
        'job_satisfaction': {'display_name': 'Job Satisfaction', 'value': st.session_state['job_satisfaction'], 'unit': '/10'},
        'work_life_balance': {'display_name': 'Work-Life Balance', 'value': st.session_state['work_life_balance'], 'unit': '/10'},
        'job_opportunities': {'display_name': 'Job Opportunities', 'value': st.session_state['job_opportunities'], 'unit': '/100'},
        'salary': {'display_name': 'Annual Salary', 'value': f"${st.session_state['salary']:,}", 'unit': ''},
        'job_security': {'display_name': 'Job Security', 'value': st.session_state['job_security'], 'unit': '/10'},
        'skills_gap': {'display_name': 'Skills Gap', 'value': st.session_state['skills_gap'], 'unit': '/10'},
        'family_influence': {'display_name': 'Family Influence', 'value': st.session_state['family_influence'], 'unit': ''},
        'mentorship_available': {'display_name': 'Mentorship Available', 'value': st.session_state['mentorship_available'], 'unit': ''},
        'certifications': {'display_name': 'Certifications', 'value': st.session_state['certifications'], 'unit': ''},
        'freelancing_experience': {'display_name': 'Freelancing Experience', 'value': st.session_state['freelancing_experience'], 'unit': ''},
        'geographic_mobility': {'display_name': 'Geographic Mobility', 'value': st.session_state['geographic_mobility'], 'unit': ''},
        'professional_networks': {'display_name': 'Professional Networks', 'value': st.session_state['professional_networks'], 'unit': '/10'},
        'career_change_events': {'display_name': 'Career Change Events', 'value': st.session_state['career_change_events'], 'unit': ''},
        'technology_adoption': {'display_name': 'Technology Adoption', 'value': st.session_state['technology_adoption'], 'unit': '/10'}
    }
    
    # Determine which fields to display
    if enable_all_fields:
        # If all fields are editable, show key fields
        fields_to_show = ['age', 'years_of_experience', 'job_satisfaction', 'field_of_study', 
                         'current_occupation', 'education_level', 'salary', 'work_life_balance', 'skills_gap']
    else:
        # Show only editable fields
        fields_to_show = input_columns_to_keep
    
    if not fields_to_show:
        return  # Return immediately if there are no fields to display
    
    with st.expander("📋 Your Profile Summary", expanded=False):
        # Dynamically calculate the number of columns
        num_fields = len(fields_to_show)
        cols_per_row = min(3, num_fields)  # Maximum of 3 columns
        num_rows = (num_fields + cols_per_row - 1) // cols_per_row  # Round up
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                field_idx = row * cols_per_row + col_idx
                
                if field_idx < num_fields:
                    field_key = fields_to_show[field_idx]
                    
                    if field_key in field_mapping:
                        field_info = field_mapping[field_key]
                        
                        with cols[col_idx]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                field_info['display_name'], 
                                f"{field_info['value']}{field_info['unit']}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

def create_prediction_interface(model_key, model_dict, std_scaler, config):
    model_config = MODEL_CONFIG[model_key]
    input_columns_to_keep = model_config['input_cols']
    enable_all_fields = len(input_columns_to_keep) == 0
    
    # Main title
    st.markdown('<h1 class="main-header">🔮 Career Change Prediction</h1>', unsafe_allow_html=True)
    # st.markdown(f'<div class="model-name">📊 {model_config["name"]} - {model_config["description"]}</div>',
    st.markdown(f'<div class="model-name">📊 {model_config["name"]}</div>', 
                unsafe_allow_html=True)
    
    with st.form("user_input_form"):
        # Demographic information
        st.markdown('<div class="section-header">👤 Demographic Information</div>', unsafe_allow_html=True)
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            st.number_input("Age", 16, 120, key="age", 
                          disabled=("age" not in input_columns_to_keep and not enable_all_fields),
                          help="Your current age")
        with demo_col2:
            st.selectbox("Gender", config['gender_selection'], key="gender",
                        disabled=("gender" not in input_columns_to_keep and not enable_all_fields))
        
        # Educational and professional background
        st.markdown('<div class="section-header">🎓 Educational & Professional Background</div>', unsafe_allow_html=True)
        edu_col1, edu_col2 = st.columns(2)
        
        with edu_col1:
            st.selectbox("Field of Study", config['field_of_study_selection'], key="field_of_study",
                        disabled=("field_of_study" not in input_columns_to_keep and not enable_all_fields))
            st.selectbox("Education Level", config['education_level_selection'], key="education_level",
                        disabled=("education_level" not in input_columns_to_keep and not enable_all_fields))
            st.number_input("Years of Working Experience", 0, 50, key="years_of_experience",
                           disabled=("years_of_experience" not in input_columns_to_keep and not enable_all_fields))
            
        with edu_col2:
            st.selectbox("Freelancing Experience?", ["Yes", "No"], key="freelancing_experience",
                        disabled=("freelancing_experience" not in input_columns_to_keep and not enable_all_fields))
            st.selectbox("Relevant Certifications?", ["Yes", "No"], key="certifications",
                        disabled=("certifications" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Number of Career Change Events", 0, 10, key="career_change_events",
                     disabled=("career_change_events" not in input_columns_to_keep and not enable_all_fields))
        
        # Job and industry background
        st.markdown('<div class="section-header">💼 Job & Industry Context</div>', unsafe_allow_html=True)
        job_col1, job_col2 = st.columns(2)
        
        with job_col1:
            st.selectbox("Current Occupation", config['current_occupation_selection'], key="current_occupation",
                        disabled=("current_occupation" not in input_columns_to_keep and not enable_all_fields))
            st.number_input("Annual Salary ($)", 0, 1000000, key="salary",
                           disabled=("salary" not in input_columns_to_keep and not enable_all_fields))
            
        with job_col2:
            st.selectbox("Industry Growth Rate", config['industry_growth_rate_selection'], key="industry_growth_rate",
                        disabled=("industry_growth_rate" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Job Opportunities (1-100)", 1, 100, key="job_opportunities",
                     disabled=("job_opportunities" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Job Security (1-10)", 1, 10, key="job_security",
                     disabled=("job_security" not in input_columns_to_keep and not enable_all_fields))
        
        # Workplace and personal factors
        st.markdown('<div class="section-header">⚖️ Workplace and Personal Factors</div>', unsafe_allow_html=True)
        work_col1, work_col2 = st.columns(2)
        
        with work_col1:
            st.slider("Job Satisfaction (1-10)", 1, 10, key="job_satisfaction",
                     disabled=("job_satisfaction" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Skills Gap (1-10)", 1, 10, key="skills_gap",
                     disabled=("skills_gap" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Technology Adoption (1-10)", 1, 10, key="technology_adoption",
                     disabled=("technology_adoption" not in input_columns_to_keep and not enable_all_fields))
            st.slider("Professional Networks (1-10)", 1, 10, key="professional_networks",
                     disabled=("professional_networks" not in input_columns_to_keep and not enable_all_fields))
            
        with work_col2:
            st.slider("Work-Life Balance (1-10)", 1, 10, key="work_life_balance",
                     disabled=("work_life_balance" not in input_columns_to_keep and not enable_all_fields))
            st.selectbox("Mentorship Available?", ["Yes", "No"], key="mentorship_available",
                        disabled=("mentorship_available" not in input_columns_to_keep and not enable_all_fields))
            st.selectbox("Family Influence", config['family_influence_selection'], key="family_influence",
                        disabled=("family_influence" not in input_columns_to_keep and not enable_all_fields))
            st.selectbox("Willing to Relocate?", ["Yes", "No"], key="geographic_mobility",
                        disabled=("geographic_mobility" not in input_columns_to_keep and not enable_all_fields))
        
        # Prediction button
        submit_button = st.form_submit_button("🔮 Predict Career Change Probability", 
                                             use_container_width=True)
        
        if submit_button:
            with st.spinner('🤖 Analyzing your profile...'):
                # Create input DataFrame
                input_df = pd.DataFrame({
                    'Field of Study': [st.session_state['field_of_study']],
                    'Current Occupation': [st.session_state['current_occupation']],
                    'Age': [st.session_state['age']],
                    'Gender': [st.session_state['gender']],
                    'Years of Experience': [st.session_state['years_of_experience']],
                    'Education Level': [st.session_state['education_level']],
                    'Industry Growth Rate': [st.session_state['industry_growth_rate']],
                    'Job Satisfaction': [st.session_state['job_satisfaction']],
                    'Work-Life Balance': [st.session_state['work_life_balance']],
                    'Job Opportunities': [st.session_state['job_opportunities']],
                    'Salary': [st.session_state['salary']],
                    'Job Security': [st.session_state['job_security']],
                    'Skills Gap': [st.session_state['skills_gap']],
                    'Family Influence': [st.session_state['family_influence']],
                    'Mentorship Available': [st.session_state['mentorship_available']],
                    'Certifications': [st.session_state['certifications']],
                    'Freelancing Experience': [st.session_state['freelancing_experience']],
                    'Geographic Mobility': [st.session_state['geographic_mobility']],
                    'Professional Networks': [st.session_state['professional_networks']],
                    'Career Change Events': [st.session_state['career_change_events']],
                    'Technology Adoption': [st.session_state['technology_adoption']],
                })
                
                # Pre-process data
                df = preprocessing(input_df, model_config['processed_cols'], std_scaler, config)
                
                # Make prediction
                prediction = model_dict[model_key].predict(df)
                
                try:
                    # Attempt to get prediction probabilities
                    prediction_proba = model_dict[model_key].predict_proba(df)[0]
                    probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                except:
                    probability = None
                
                # Display results
                result_class = "prediction-yes" if prediction[0] == 1 else "prediction-no"
                result_text = "Yes, likely to change career" if prediction[0] == 1 else "No, likely to stay in current career"
                result_emoji = "✅" if prediction[0] == 1 else "❌"
                
                st.markdown(f'''
                <div class="prediction-result {result_class}">
                    <div class="probability-text">{result_emoji} {result_text}</div>
                    {f"<div>Confidence: {probability:.2%}</div>" if probability is not None else ""}
                </div>
                ''', unsafe_allow_html=True)
                
                # Optimized user input summary - only display editable fields
                display_editable_profile_summary(input_columns_to_keep, enable_all_fields)

def create_charts_dashboard(model_dict):
    st.markdown('<h1 class="main-header">Career Change Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Load the data files
    custom_na_filter = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', 
                        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    df_data = pd.read_csv('career_change_prediction_dataset_dirty_missing.csv', 
                           keep_default_na=False, na_values=custom_na_filter)

    # Load training data for both models
    try:
        d1_data = pd.read_csv('d1_x_train.csv')  # For top2
        # d3_data = pd.read_csv('d3_x_train.csv')  # For top1
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return

    # Function to get feature importance
    def get_feature_importance(model, data):
        importance = model.feature_importances_
        feature_names = data.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        return importance_df.sort_values(by='Importance', ascending=False)

    # Get feature importance for both models
    # importance_df_top1 = get_feature_importance(model_dict['top1'], d3_data)
    importance_df_top2 = get_feature_importance(model_dict['top2'], d1_data)

    # Replace raw feature names with more user-friendly versions
    def format_feature_names(importance_df):
        formatted_features = []
        for feature in importance_df['Feature']:
            if 'years_of_experience' in feature:
                formatted_features.append('Experience')
            elif 'field_of_study_' in feature:
                formatted_features.append("Field of Study: " + feature.split('_')[-1].replace('_', ' ').title())
            elif 'current_occupation_' in feature:
                formatted_features.append("Current Occupation:" + feature.split('_')[-1].replace('_', ' ').title())
            else:
                formatted_features.append(feature.replace('_', ' ').title())
        importance_df['Formatted Feature'] = formatted_features
        return importance_df

    # importance_df_top1 = format_feature_names(importance_df_top1)
    importance_df_top2 = format_feature_names(importance_df_top2)

    # Create bar plots for feature importance without filtering
    # fig_top1 = go.Figure(data=[
    #   go.Bar(x=importance_df_top1['Importance'], y=importance_df_top1['Formatted Feature'], orientation='h',
    #          marker_color='royalblue', name='Top 1 Model')
    #])

    fig_top2 = go.Figure(data=[
        go.Bar(x=importance_df_top2['Importance'], y=importance_df_top2['Formatted Feature'], orientation='h',
               marker_color='orange', name='Top 2 Model')
    ])

    # Update layouts
    #fig_top1.update_layout(
    #    title='Feature Importance for Free Version',
    #    xaxis_title='Importance',
    #    yaxis_title='Features',
    #    yaxis=dict(autorange='reversed')
    #)

    fig_top2.update_layout(
        title='Feature Importance for Subscribed Version',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=800,
        margin=dict(l=150),  # Increase left margin
        yaxis=dict(autorange='reversed'),
        yaxis_tickfont=dict(size=12),
    ).update_yaxes(tickangle=0)  # Change to 0 or adjust for horizontal labels



    # Create interactive charts
    # tab1, tab2, tab3 = st.tabs(["📈 Factor Analysis", "🔍 Distribution Analysis", "📊 Correlation Matrix"])
    tab1, tab2 = st.tabs(["📈 Factor Analysis", "🔍 Distribution Analysis"])

    with tab1:
        st.subheader("Key Factors vs Career Change Likelihood")
        
        # Select factors to analyze
        factors = [
            'Job Satisfaction', 'Salary', 'Field of Study', 'Current Occupation', 
            'Age', 'Gender', 'Job Opportunities', 'Education Level', 'Work-Life Balance', 'Job Security', 
            'Family Influence', 'Skills Gap'
        ]
        
        selected_factors = st.multiselect("Select factors to analyze:", factors, default=factors[:4])
        
        if selected_factors:
            cols = st.columns(2)
            for idx, factor in enumerate(selected_factors):
                with cols[idx % 2]:
                    if factor in df_data.columns:
                        fig = px.box(df_data, x='Likely to Change Occupation', y=factor,
                                     title=f'{factor} vs Career Change Likelihood',
                                     color='Likely to Change Occupation',
                                     color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'})
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        st.subheader("Feature Importance")

        st.plotly_chart(fig_top2, use_container_width=True)
        
        # Display both feature importance plots side by side
        # col1, col2 = st.columns(2)
        # with col1:
        #    st.plotly_chart(fig_top1, use_container_width=True)
        # with col2:
        #    st.plotly_chart(fig_top2, use_container_width=True)


    with tab2:
        st.subheader("Distribution Analysis")
        features_to_display = [
            'Job Satisfaction', 'Salary', 'Field of Study', 'Current Occupation', 
            'Age', 'Gender', 'Job Opportunities', 'Experience',
            'Education Level', 'Work-Life Balance', 'Job Security', 'Skills Gap'
        ]
        
        cols = st.columns(2)

        for idx, feature in enumerate(features_to_display):
            if feature in df_data.columns:
                with cols[idx % 2]:  # Alternate between the two columns
                    fig = px.histogram(
                        df_data,
                        x=feature,
                        color='Likely to Change Occupation',
                        title=f'{feature} Distribution by Career Change Likelihood',
                        color_discrete_map={'Yes': '#667eea', 'No': '#f093fb'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # with tab3:
    #     st.subheader("Feature Correlation Heatmap")
        
    #     # Convert categorical variables to numeric
    #     df_data['Likely to Change Occupation'] = df_data['Likely to Change Occupation'].map({'Yes': 1, 'No': 0})
        
    #     # Identify and convert categorical columns
    #     categorical_cols = ['gender', 'education_level', 'industry_growth_rate', 
    #                         'family_influence', 'mentorship_available', 
    #                         'certifications', 'freelancing_experience', 
    #                         'geographic_mobility']

    #     for col in categorical_cols:
    #         if col in df_data.columns:
    #             df_data[col] = df_data[col].astype('category').cat.codes
        
    #     # Generate numeric columns list
    #     numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
        
    #     # Calculate correlation matrix
    #     corr_matrix = df_data[numeric_cols].corr()

    #     # Create heatmap using seaborn
    #     plt.figure(figsize=(20, 20))
    #     sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    #     plt.title('All Columns Correlation Heatmap')

    #     # Display the heatmap in Streamlit
    #     st.pyplot(plt)

def main():
    # Load configuration and data
    config = load_config()
    std_scaler, model_dict, df_data = load_models_and_data()
    
    if std_scaler is None or model_dict is None:
        st.error("Unable to load model files, please check file paths")
        return
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    sidebar_option = st.sidebar.radio(
        "Choose an option:",
        # top 1 model is Free Version,top 2 model is Subscribed Version
        ["🤖 Free Version", "🌟 Subscribed Version", "📊 Analytics Dashboard"],
        help="Select a model for prediction or view analytics"
    )
    
    # Add model information
    if "Model" in sidebar_option:
        model_key = sidebar_option.split()[1].lower()
        if model_key in MODEL_CONFIG:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Model Information")
            st.sidebar.info(MODEL_CONFIG[model_key]['description'])
    
    # Main interface
    if "Free Version" in sidebar_option:
        create_prediction_interface("top1", model_dict, std_scaler, config)
    elif "Subscribed Version" in sidebar_option:
        create_prediction_interface("top2", model_dict, std_scaler, config)
    elif "Analytics Dashboard" in sidebar_option:
        create_charts_dashboard(model_dict)

if __name__ == "__main__":
    main()