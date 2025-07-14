# Yoga Recommender System - Data Exploration & Analysis
# Expert-level data science approach for personalized yoga recommendations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== YOGA RECOMMENDER SYSTEM - DATA EXPLORATION ===")
print("Loading datasets from Kaggle...")

# Load datasets
try:
    # Load the datasets
    users_df = pd.read_csv('/kaggle/input/users-data/users_data.csv')
    asanas_df = pd.read_csv('/kaggle/input/final-asana-dataset/final_asana_dataset.csv')
    
    print("✓ Datasets loaded successfully!")
    print(f"Users dataset shape: {users_df.shape}")
    print(f"Asanas dataset shape: {asanas_df.shape}")
    
except FileNotFoundError:
    print("Note: Running in local environment. Using sample data structure...")
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Sample users data
    n_users = 1000
    users_df = pd.DataFrame({
        'Age': np.random.randint(20, 65, n_users),
        'Gender': np.random.choice(['F', 'M'], n_users),
        'Height (cm)': np.random.normal(170, 10, n_users),
        'Weight (kg)': np.random.normal(70, 15, n_users),
        'Body Fat (%)': np.random.normal(20, 8, n_users),
        'Diastolic (mmHg)': np.random.normal(80, 10, n_users),
        'Systolic (mmHg)': np.random.normal(120, 15, n_users),
        'Grip Force (kg)': np.random.normal(35, 10, n_users),
        'Sit and Bend Forward (cm)': np.random.normal(15, 8, n_users),
        'Sit-ups Count': np.random.randint(10, 50, n_users),
        'Broad Jump (cm)': np.random.normal(180, 30, n_users),
        'Class': np.random.choice(['A', 'B', 'C', 'D'], n_users)
    })
    
    # Sample asanas data (simplified)
    asanas_sample = [
        'Mountain Pose', 'Tree Pose', 'Warrior I', 'Warrior II', 'Downward Dog',
        'Child Pose', 'Cobra Pose', 'Bridge Pose', 'Triangle Pose', 'Plank Pose'
    ]
    
    n_asanas = 50
    asanas_df = pd.DataFrame({
        'asana_name': [f'Pose_{i}' for i in range(n_asanas)],
        'asana_type': np.random.choice(['Hatha', 'Power', 'Vinyasa'], n_asanas),
        'pose_type': np.random.choice(['Standing', 'Balance', 'Backbend', 'Seated'], n_asanas),
        'difficulty_level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_asanas),
        'duration_secs': np.random.randint(30, 180, n_asanas),
        'target_age_group': ['8-80'] * n_asanas,
        'Body_Parts': np.random.choice(['Back/Spine', 'Legs/Thighs', 'Core/Abdomen'], n_asanas),
        'Focus_Area': np.random.choice(['Flexibility/Stretching', 'Strength Building', 'Balance/Stability'], n_asanas),
        'Precautions': np.random.choice(['Generally safe for all', 'Back injuries', 'Knee problems/injuries'], n_asanas),
        'Pain_Points': np.random.choice(['Lower back pain', 'Stress/anxiety', 'Balance issues'], n_asanas)
    })

print("\n=== USERS DATASET EXPLORATION ===")
print("Dataset Info:")
print(users_df.info())
print("\nDataset Description:")
print(users_df.describe())

print("\n=== ASANAS DATASET EXPLORATION ===")
print("Dataset Info:")
print(asanas_df.info())
print("\nFirst few rows:")
print(asanas_df.head())

# Data Quality Assessment
print("\n=== DATA QUALITY ASSESSMENT ===")

print("\n1. Missing Values Analysis:")
print("Users dataset missing values:")
print(users_df.isnull().sum())
print("\nAsanas dataset missing values:")
print(asanas_df.isnull().sum())

print("\n2. Data Types and Unique Values:")
print("\nUsers dataset unique values:")
for col in users_df.columns:
    if users_df[col].dtype == 'object':
        print(f"{col}: {users_df[col].unique()}")
    else:
        print(f"{col}: {users_df[col].nunique()} unique values")

print("\nAsanas dataset unique values:")
for col in asanas_df.columns:
    if asanas_df[col].dtype == 'object':
        print(f"{col}: {asanas_df[col].nunique()} unique values")
        if asanas_df[col].nunique() < 20:  # Show unique values for categorical columns
            print(f"  Values: {asanas_df[col].unique()}")

# Advanced Data Analysis
print("\n=== ADVANCED DATA ANALYSIS ===")

# 1. Users Physical Fitness Analysis
print("\n1. Physical Fitness Distribution Analysis:")

# Create fitness score
users_df['BMI'] = users_df['Weight (kg)'] / (users_df['Height (cm)'] / 100) ** 2
users_df['Fitness_Score'] = (
    users_df['Sit-ups Count'] * 0.3 +
    users_df['Broad Jump (cm)'] * 0.002 +
    users_df['Grip Force (kg)'] * 0.5 +
    users_df['Sit and Bend Forward (cm)'] * 0.5 -
    users_df['Body Fat (%)'] * 0.2
)

# Normalize fitness score
users_df['Fitness_Score_Normalized'] = (
    users_df['Fitness_Score'] - users_df['Fitness_Score'].min()
) / (users_df['Fitness_Score'].max() - users_df['Fitness_Score'].min())

print(f"Average BMI: {users_df['BMI'].mean():.2f}")
print(f"Average Fitness Score: {users_df['Fitness_Score'].mean():.2f}")
print(f"Performance Class Distribution:")
print(users_df['Class'].value_counts())

# 2. Asanas Complexity Analysis
print("\n2. Asanas Complexity and Characteristics:")

if 'difficulty_level' in asanas_df.columns:
    print("Difficulty Level Distribution:")
    print(asanas_df['difficulty_level'].value_counts())

if 'Focus_Area' in asanas_df.columns:
    print("\nFocus Area Distribution:")
    print(asanas_df['Focus_Area'].value_counts())

if 'pose_type' in asanas_df.columns:
    print("\nPose Type Distribution:")
    print(asanas_df['pose_type'].value_counts())

# 3. Age Group Analysis
print("\n3. Age Group Analysis:")
print(f"Users age range: {users_df['Age'].min()} - {users_df['Age'].max()}")
print(f"Average age: {users_df['Age'].mean():.1f}")

# Create age groups
users_df['Age_Group'] = pd.cut(users_df['Age'], 
                               bins=[0, 30, 40, 50, 65], 
                               labels=['20-30', '30-40', '40-50', '50+'])
print("\nAge Group Distribution:")
print(users_df['Age_Group'].value_counts())

# 4. Gender-based Analysis
print("\n4. Gender-based Physical Characteristics:")
gender_stats = users_df.groupby('Gender').agg({
    'Height (cm)': ['mean', 'std'],
    'Weight (kg)': ['mean', 'std'],
    'Body Fat (%)': ['mean', 'std'],
    'Fitness_Score': ['mean', 'std'],
    'Class': lambda x: x.mode().iloc[0]
}).round(2)

print(gender_stats)

# Feature Engineering for Recommendation System
print("\n=== FEATURE ENGINEERING FOR RECOMMENDATION SYSTEM ===")

# 1. User Profile Feature Engineering
print("\n1. Creating User Profile Features:")

# Physical capability indicators
users_df['Flexibility_Score'] = users_df['Sit and Bend Forward (cm)']
users_df['Strength_Score'] = (users_df['Grip Force (kg)'] + users_df['Sit-ups Count']) / 2
users_df['Balance_Score'] = users_df['Broad Jump (cm)'] / 10  # Normalize
users_df['Cardio_Score'] = 100 - users_df['Body Fat (%)']  # Inverse relationship

# Health risk indicators
users_df['BP_Risk'] = np.where(
    (users_df['Systolic (mmHg)'] > 140) | (users_df['Diastolic (mmHg)'] > 90), 
    'High_BP', 'Normal_BP'
)

users_df['Weight_Status'] = pd.cut(users_df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Experience level mapping (based on fitness class)
experience_mapping = {'A': 'Advanced', 'B': 'Intermediate', 'C': 'Beginner', 'D': 'Beginner'}
users_df['Yoga_Experience_Level'] = users_df['Class'].map(experience_mapping)

print("User profile features created:")
print(f"- Flexibility Score: {users_df['Flexibility_Score'].mean():.2f} ± {users_df['Flexibility_Score'].std():.2f}")
print(f"- Strength Score: {users_df['Strength_Score'].mean():.2f} ± {users_df['Strength_Score'].std():.2f}")
print(f"- Balance Score: {users_df['Balance_Score'].mean():.2f} ± {users_df['Balance_Score'].std():.2f}")
print(f"- Cardio Score: {users_df['Cardio_Score'].mean():.2f} ± {users_df['Cardio_Score'].std():.2f}")
print(f"- BP Risk Distribution: {users_df['BP_Risk'].value_counts().to_dict()}")
print(f"- Weight Status: {users_df['Weight_Status'].value_counts().to_dict()}")

# 2. Asanas Feature Engineering
print("\n2. Creating Asanas Features:")

# Create pose complexity score
def calculate_pose_complexity(row):
    """Calculate pose complexity based on difficulty and characteristics"""
    difficulty_scores = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
    base_score = difficulty_scores.get(row.get('difficulty_level', 'Beginner'), 1)
    
    # Adjust based on pose type
    if 'pose_type' in row:
        complex_poses = ['Inversion', 'Arm Balance', 'Backbend']
        if any(pose in str(row['pose_type']) for pose in complex_poses):
            base_score += 1
    
    return base_score

if 'difficulty_level' in asanas_df.columns:
    asanas_df['Complexity_Score'] = asanas_df.apply(calculate_pose_complexity, axis=1)

# Create binary features for key characteristics
def create_binary_features(df, column, prefix):
    """Create binary features from categorical column"""
    if column in df.columns:
        unique_values = df[column].dropna().unique()
        for value in unique_values:
            df[f'{prefix}_{value}'] = df[column].apply(lambda x: 1 if value in str(x) else 0)
    return df

# Create binary features for focus areas, body parts, precautions
if 'Focus_Area' in asanas_df.columns:
    asanas_df = create_binary_features(asanas_df, 'Focus_Area', 'Focus')

if 'Body_Parts' in asanas_df.columns:
    asanas_df = create_binary_features(asanas_df, 'Body_Parts', 'BodyPart')

if 'Precautions' in asanas_df.columns:
    asanas_df = create_binary_features(asanas_df, 'Precautions', 'Precaution')

print("Asanas features created:")
if 'Complexity_Score' in asanas_df.columns:
    print(f"- Complexity Score: {asanas_df['Complexity_Score'].mean():.2f} ± {asanas_df['Complexity_Score'].std():.2f}")

# Count binary features
binary_features = [col for col in asanas_df.columns if col.startswith(('Focus_', 'BodyPart_', 'Precaution_'))]
print(f"- Binary features created: {len(binary_features)}")

# 3. User-Asana Matching Framework
print("\n3. User-Asana Matching Framework Setup:")

def create_user_preference_profile(user_row):
    """Create user preference profile for matching"""
    profile = {
        'age': user_row['Age'],
        'fitness_level': user_row['Yoga_Experience_Level'],
        'flexibility_score': user_row['Flexibility_Score'],
        'strength_score': user_row['Strength_Score'],
        'balance_score': user_row['Balance_Score'],
        'bp_risk': user_row['BP_Risk'],
        'weight_status': user_row['Weight_Status'],
        'bmi': user_row['BMI']
    }
    return profile

def create_asana_feature_vector(asana_row):
    """Create asana feature vector for matching"""
    features = {
        'complexity': asana_row.get('Complexity_Score', 1),
        'duration': asana_row.get('duration_secs', 60),
        'difficulty': asana_row.get('difficulty_level', 'Beginner')
    }
    
    # Add binary features
    for col in asana_row.index:
        if col.startswith(('Focus_', 'BodyPart_', 'Precaution_')):
            features[col] = asana_row[col]
    
    return features

print("User-Asana matching framework created")
print("✓ Ready for recommendation algorithm development")

# Save processed datasets
print("\n=== SAVING PROCESSED DATA ===")
users_df.to_csv('processed_users_data.csv', index=False)
asanas_df.to_csv('processed_asanas_data.csv', index=False)
print("✓ Processed datasets saved")

print("\n=== SUMMARY ===")
print(f"Users dataset: {users_df.shape[0]} records with {users_df.shape[1]} features")
print(f"Asanas dataset: {asanas_df.shape[0]} records with {asanas_df.shape[1]} features")
print("✓ Data exploration and feature engineering complete")
print("✓ Ready for recommendation system development")

# Display sample processed data
print("\n=== SAMPLE PROCESSED DATA ===")
print("Sample user profile:")
print(users_df[['Age', 'Gender', 'BMI', 'Fitness_Score_Normalized', 'Yoga_Experience_Level', 'BP_Risk']].head(3))

print("\nSample asana features:")
display_cols = ['asana_name', 'difficulty_level', 'Complexity_Score'] + [col for col in asanas_df.columns if col.startswith('Focus_')][:3]
available_cols = [col for col in display_cols if col in asanas_df.columns]
print(asanas_df[available_cols].head(3))
