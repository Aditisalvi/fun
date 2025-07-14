# Expert Yoga Recommender System - Data Preprocessing & Feature Engineering
# State-of-the-art personalized recommendations with safety-first approach

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

class YogaRecommenderPreprocessor:
    """
    Expert-level preprocessing and feature engineering for yoga recommendation system
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.mlb_focus = MultiLabelBinarizer()
        self.mlb_body_parts = MultiLabelBinarizer()
        self.mlb_precautions = MultiLabelBinarizer()
        self.mlb_pain_points = MultiLabelBinarizer()
        
        # Define mappings based on your specifications
        self.fitness_to_yoga_mapping = {
            'A': 'Advanced',     # Best fitness -> Advanced yoga
            'B': 'Intermediate', # Good fitness -> Intermediate yoga  
            'C': 'Beginner',     # Average fitness -> Beginner yoga
            'D': 'Beginner'      # Poor fitness -> Beginner yoga
        }
        
        # Safety priority mapping - health conditions to avoid
        self.safety_conditions = {
            'high_bp': ['High blood pressure'],
            'low_bp': ['Low blood pressure'], 
            'back_injury': ['Back injuries'],
            'knee_injury': ['Knee problems/injuries'],
            'neck_injury': ['Neck injuries'],
            'shoulder_injury': ['Shoulder injuries'],
            'ankle_injury': ['Ankle injuries'],
            'wrist_injury': ['Wrist injuries'],
            'hip_injury': ['Hip injuries'],
            'heart_condition': ['Heart conditions'],
            'pregnancy': ['Pregnancy'],
            'glaucoma': ['Glaucoma/eye conditions'],
            'carpal_tunnel': ['Carpal tunnel'],
            'hamstring_injury': ['Hamstring injuries'],
            'asthma': ['Asthma'],
            'migraine': ['Migraine'],
            'insomnia': ['Insomnia'],
            'diarrhea': ['Diarrhea'],
            'menstruation': ['Menstruation'],
            'balance_disorder': ['Balance disorders']
        }
        
        # Focus area priority mapping
        self.focus_areas = [
            'Flexibility/Stretching', 'Strength Building', 'Balance/Stability',
            'Stress Relief/Calming', 'Posture Improvement', 'Meditation/Focus',
            'Digestion Support', 'Cardiovascular Fitness', 'Endurance Building',
            'Energy Building', 'Circulation Enhancement', 'Coordination',
            'Relaxation/Restorative', 'Emotional Release', 'Detoxification',
            'Breathing Improvement'
        ]
        
        print("=== YOGA RECOMMENDER SYSTEM INITIALIZED ===")
        print("✓ Safety-first approach enabled")
        print("✓ Multi-criteria recommendation framework ready")
        
    def load_and_validate_data(self):
        """Load and validate datasets with comprehensive error handling"""
        try:
            # Load datasets
            print("\n1. Loading datasets...")
            self.users_df = pd.read_csv('/kaggle/input/users-data/bodyPerformance.csv')
            self.asanas_df = pd.read_csv('/kaggle/input/final-asana-dataset/new_final_dataset (1).csv')
            
            print(f"✓ Users dataset loaded: {self.users_df.shape}")
            print(f"✓ Asanas dataset loaded: {self.asanas_df.shape}")
            
            # Validate essential columns
            required_user_cols = ['age', 'gender', 'height_cm', 'weight_kg', 'class']
            required_asana_cols = ['asana_name', 'difficulty_level', 'Focus_Area', 'Precautions']
            
            missing_user_cols = [col for col in required_user_cols if col not in self.users_df.columns]
            missing_asana_cols = [col for col in required_asana_cols if col not in self.asanas_df.columns]
            
            if missing_user_cols:
                print(f"⚠️  Missing user columns: {missing_user_cols}")
            if missing_asana_cols:
                print(f"⚠️  Missing asana columns: {missing_asana_cols}")
                
            # Display dataset info
            print("\n2. Dataset Overview:")
            print("Users dataset columns:", list(self.users_df.columns))
            print("Asanas dataset columns:", list(self.asanas_df.columns))
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def preprocess_users_data(self):
        """Advanced preprocessing of users dataset for yoga recommendations"""
        print("\n3. Preprocessing Users Dataset...")
        
        # Clean column names
        self.users_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in self.users_df.columns]
        
        # Handle missing values
        numeric_columns = self.users_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.users_df[col].fillna(self.users_df[col].median(), inplace=True)
        
        categorical_columns = self.users_df.select_dtypes(include=[object]).columns
        for col in categorical_columns:
            self.users_df[col].fillna(self.users_df[col].mode().iloc[0], inplace=True)
            
        # Create comprehensive user profile features
        print("   Creating user profile features...")
        
        # 1. Physical Health Metrics
        self.users_df['bmi'] = self.users_df['weight_kg'] / (self.users_df['height_cm'] / 100) ** 2
        self.users_df['bmi_category'] = pd.cut(self.users_df['bmi'], 
                                               bins=[0, 18.5, 25, 30, 100], 
                                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # 2. Cardiovascular Health Assessment
        self.users_df['bp_systolic'] = self.users_df.get('systolic_mmhg', 120)
        self.users_df['bp_diastolic'] = self.users_df.get('diastolic_mmhg', 80)
        self.users_df['bp_risk'] = np.where(
            (self.users_df['bp_systolic'] > 140) | (self.users_df['bp_diastolic'] > 90), 
            'high_bp', 
            np.where(
                (self.users_df['bp_systolic'] < 90) | (self.users_df['bp_diastolic'] < 60),
                'low_bp', 'normal_bp'
            )
        )
        
        # 3. Physical Capability Scores (0-100 scale)
        if 'sit_and_bend_forward_cm' in self.users_df.columns:
            self.users_df['flexibility_score'] = np.clip(
                (self.users_df['sit_and_bend_forward_cm'] + 10) * 2.5, 0, 100
            )
        else:
            self.users_df['flexibility_score'] = np.random.normal(50, 15, len(self.users_df))
            
        if 'gripforce' in self.users_df.columns:
            self.users_df['strength_score'] = np.clip(
                self.users_df['gripforce'] * 2, 0, 100
            )
        else:
            self.users_df['strength_score'] = np.random.normal(50, 15, len(self.users_df))
            
        if 'broad_jump_cm' in self.users_df.columns:
            self.users_df['balance_score'] = np.clip(
                self.users_df['broad_jump_cm'] / 3, 0, 100
            )
        else:
            self.users_df['balance_score'] = np.random.normal(50, 15, len(self.users_df))
            
        # 4. Fitness Level Mapping
        self.users_df['yoga_difficulty_level'] = self.users_df['class'].map(self.fitness_to_yoga_mapping)
        
        # 5. Age-based adjustments
        self.users_df['age_group'] = pd.cut(self.users_df['age'], 
                                            bins=[0, 30, 45, 60, 100], 
                                            labels=['Young', 'Adult', 'Middle_Age', 'Senior'])
        
        # 6. Overall fitness composite score
        self.users_df['fitness_composite'] = (
            self.users_df['flexibility_score'] * 0.3 +
            self.users_df['strength_score'] * 0.3 +
            self.users_df['balance_score'] * 0.2 +
            (100 - self.users_df['body_fat_%']) * 0.2
        )
        
        # 7. Risk assessment for yoga practice
        self.users_df['practice_risk_level'] = 'low'
        
        # High risk conditions
        high_risk_conditions = (
            (self.users_df['bp_risk'] == 'high_bp') |
            (self.users_df['age'] > 65) |
            (self.users_df['bmi'] > 35) |
            (self.users_df['fitness_composite'] < 25)
        )
        self.users_df.loc[high_risk_conditions, 'practice_risk_level'] = 'high'
        
        # Medium risk conditions
        medium_risk_conditions = (
            (self.users_df['bp_risk'] == 'low_bp') |
            (self.users_df['age'] > 55) |
            (self.users_df['bmi'] > 30) |
            (self.users_df['fitness_composite'] < 40)
        )
        self.users_df.loc[medium_risk_conditions & (self.users_df['practice_risk_level'] == 'low'), 'practice_risk_level'] = 'medium'
        
        print(f"   ✓ User profiles created with {len(self.users_df.columns)} features")
        print(f"   ✓ Yoga difficulty mapping: {self.users_df['yoga_difficulty_level'].value_counts().to_dict()}")
        print(f"   ✓ Practice risk assessment: {self.users_df['practice_risk_level'].value_counts().to_dict()}")
        
        return self.users_df
    
    def preprocess_asanas_data(self):
        """Advanced preprocessing of asanas dataset for recommendation matching"""
        print("\n4. Preprocessing Asanas Dataset...")
        
        # Clean column names
        self.asanas_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in self.asanas_df.columns]
        
        # Handle missing values
        self.asanas_df['asana_name'].fillna('Unknown_Pose', inplace=True)
        self.asanas_df['difficulty_level'].fillna('Beginner', inplace=True)
        self.asanas_df['focus_area'].fillna('Flexibility/Stretching', inplace=True)
        self.asanas_df['precautions'].fillna('Generally safe for all', inplace=True)
        
        # 1. Parse multi-value fields
        print("   Parsing multi-value categorical fields...")
        
        def parse_multi_value_field(field_value, delimiter=','):
            """Parse comma-separated values and clean them"""
            if pd.isna(field_value):
                return []
            items = str(field_value).split(delimiter)
            return [item.strip() for item in items if item.strip()]
        
        # Parse focus areas
        self.asanas_df['focus_area_list'] = self.asanas_df['focus_area'].apply(parse_multi_value_field)
        
        # Parse body parts
        if 'body_parts' in self.asanas_df.columns:
            self.asanas_df['body_parts_list'] = self.asanas_df['body_parts'].apply(parse_multi_value_field)
        else:
            self.asanas_df['body_parts_list'] = [['Full Body']] * len(self.asanas_df)
        
        # Parse precautions
        self.asanas_df['precautions_list'] = self.asanas_df['precautions'].apply(parse_multi_value_field)
        
        # Parse pain points
        if 'pain_points' in self.asanas_df.columns:
            self.asanas_df['pain_points_list'] = self.asanas_df['pain_points'].apply(parse_multi_value_field)
        else:
            self.asanas_df['pain_points_list'] = [['General wellness']] * len(self.asanas_df)
        
        # 2. Create complexity scoring system
        print("   Creating pose complexity scoring...")
        
        difficulty_scores = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.asanas_df['difficulty_score'] = self.asanas_df['difficulty_level'].map(difficulty_scores)
        
        def calculate_complexity_score(row):
            base_score = row['difficulty_score']
            
            # Increase complexity for certain pose types
            if 'pose_type' in row and pd.notna(row['pose_type']):
                complex_poses = ['Inversion', 'Arm Balance', 'Backbend', 'Twist']
                if any(pose_type in str(row['pose_type']) for pose_type in complex_poses):
                    base_score += 0.5
            
            # Increase complexity for longer durations
            if 'duration_secs' in row and pd.notna(row['duration_secs']):
                if row['duration_secs'] > 120:  # More than 2 minutes
                    base_score += 0.3
            
            return min(base_score, 3.5)  # Cap at 3.5
        
        self.asanas_df['complexity_score'] = self.asanas_df.apply(calculate_complexity_score, axis=1)
        
        # 3. Create binary feature matrices for matching
        print("   Creating binary feature matrices...")
        
        # Focus area binary features
        all_focus_areas = []
        for focus_list in self.asanas_df['focus_area_list']:
            all_focus_areas.extend(focus_list)
        unique_focus_areas = list(set(all_focus_areas))
        
        focus_matrix = self.mlb_focus.fit_transform(self.asanas_df['focus_area_list'])
        focus_columns = [f'focus_{area.lower().replace("/", "_").replace(" ", "_")}' for area in self.mlb_focus.classes_]
        focus_df = pd.DataFrame(focus_matrix, columns=focus_columns, index=self.asanas_df.index)
        
        # Ensure focus_df columns are numeric
        focus_df = focus_df.astype(int)  # Convert to integer type to ensure numeric values
        
        # Body parts binary features
        all_body_parts = []
        for body_list in self.asanas_df['body_parts_list']:
            all_body_parts.extend(body_list)
        unique_body_parts = list(set(all_body_parts))
        
        body_matrix = self.mlb_body_parts.fit_transform(self.asanas_df['body_parts_list'])
        body_columns = [f'body_{part.lower().replace("/", "_").replace(" ", "_")}' for part in self.mlb_body_parts.classes_]
        body_df = pd.DataFrame(body_matrix, columns=body_columns, index=self.asanas_df.index)
        
        # Precautions binary features (for safety filtering)
        precautions_matrix = self.mlb_precautions.fit_transform(self.asanas_df['precautions_list'])
        precautions_columns = [f'precaution_{prec.lower().replace("/", "_").replace(" ", "_")}' for prec in self.mlb_precautions.classes_]
        precautions_df = pd.DataFrame(precautions_matrix, columns=precautions_columns, index=self.asanas_df.index)
    
        # Pain points binary features
        pain_matrix = self.mlb_pain_points.fit_transform(self.asanas_df['pain_points_list'])
        pain_columns = [f'pain_{point.lower().replace("/", "_").replace(" ", "_")}' for point in self.mlb_pain_points.classes_]
        pain_df = pd.DataFrame(pain_matrix, columns=pain_columns, index=self.asanas_df.index)
        
        # 4. Combine all features
        print("   Combining all asana features...")
        
        self.asanas_processed = pd.concat([
            self.asanas_df[['asana_name', 'difficulty_level', 'difficulty_score', 'complexity_score', 
                           'focus_area', 'precautions', 'duration_secs']],
            focus_df,
            body_df,
            precautions_df,
            pain_df
        ], axis=1)
        
        # 5. Create age appropriateness scoring
        if 'target_age_group' in self.asanas_df.columns:
            def parse_age_range(age_str):
                """Parse age range string to get min and max ages"""
                if pd.isna(age_str):
                    return 8, 80  # Default range
                numbers = re.findall(r'\d+', str(age_str))
                if len(numbers) >= 2:
                    return int(numbers[0]), int(numbers[1])
                elif len(numbers) == 1:
                    return int(numbers[0]), 80
                else:
                    return 8, 80
            
            age_ranges = self.asanas_df['target_age_group'].apply(parse_age_range)
            self.asanas_processed['min_age'] = [age_range[0] for age_range in age_ranges]
            self.asanas_processed['max_age'] = [age_range[1] for age_range in age_ranges]
        else:
            self.asanas_processed['min_age'] = 8
            self.asanas_processed['max_age'] = 80
        
        # 6. Create recommendation scoring features
        print("   Creating recommendation scoring features...")
        
        # Safety score (higher = safer)
        self.asanas_processed['safety_score'] = 1.0  # Default safe
        
        # Reduce safety score for poses with specific precautions
        risky_precautions = ['high_blood_pressure', 'heart_conditions', 'back_injuries', 
                            'knee_problems/injuries', 'neck_injuries', 'pregnancy']
        
        for precaution in risky_precautions:
            precaution_col = f'precaution_{precaution.lower().replace("/", "_").replace(" ", "_")}'
            if precaution_col in self.asanas_processed.columns:
                self.asanas_processed.loc[self.asanas_processed[precaution_col] == 1, 'safety_score'] -= 0.2
        
        # Accessibility score (higher = more accessible)
        self.asanas_processed['accessibility_score'] = 1.0 - (self.asanas_processed['complexity_score'] - 1) / 2.5
        
        # Effectiveness score (based on focus areas covered)
        focus_columns = [col for col in self.asanas_processed.columns if col.startswith('focus_')]
        # Verify and convert focus columns to numeric
        for col in focus_columns:
            self.asanas_processed[col] = pd.to_numeric(self.asanas_processed[col], errors='coerce').fillna(0).astype(int)
        
        self.asanas_processed['effectiveness_score'] = self.asanas_processed[focus_columns].sum(axis=1) / 5
        
        print(f"   ✓ Asanas processed with {len(self.asanas_processed.columns)} features")
        print(f"   ✓ Difficulty distribution: {self.asanas_processed['difficulty_level'].value_counts().to_dict()}")
        print(f"   ✓ Complexity score range: {self.asanas_processed['complexity_score'].min():.2f} - {self.asanas_processed['complexity_score'].max():.2f}")
        print(f"   ✓ Safety score range: {self.asanas_processed['safety_score'].min():.2f} - {self.asanas_processed['safety_score'].max():.2f}")
        
        return self.asanas_processed   

    def create_user_asana_features(self):
        """Create comprehensive features for user-asana matching"""
        print("\n5. Creating User-Asana Matching Features...")
        
        # 1. Create user feature vectors for similarity matching
        print("   Creating user feature vectors...")
        
        user_features = ['age', 'bmi', 'flexibility_score', 'strength_score', 'balance_score', 
                        'fitness_composite', 'difficulty_score']
        
        # Add difficulty score based on yoga level
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.users_df['difficulty_score'] = self.users_df['yoga_difficulty_level'].map(difficulty_mapping)
        
        # Normalize user features
        self.user_features_normalized = self.scaler.fit_transform(self.users_df[user_features])
        self.user_features_df = pd.DataFrame(self.user_features_normalized, columns=user_features, index=self.users_df.index)
        
        # 2. Create asana feature vectors for similarity matching
        print("   Creating asana feature vectors...")
        
        asana_numeric_features = ['difficulty_score', 'complexity_score', 'safety_score', 
                                 'accessibility_score', 'effectiveness_score']
        
        # Add age compatibility features
        asana_features_extended = asana_numeric_features + ['min_age', 'max_age']
        
        # Get focus area features
        focus_features = [col for col in self.asanas_processed.columns if col.startswith('focus_')]
        
        # Get body part features
        body_features = [col for col in self.asanas_processed.columns if col.startswith('body_')]
        
        # Combine all asana features
        all_asana_features = asana_features_extended + focus_features + body_features
        available_features = [col for col in all_asana_features if col in self.asanas_processed.columns]
        
        self.asana_features_df = self.asanas_processed[available_features].fillna(0)
        
        # 3. Create similarity matching framework
        print("   Creating similarity matching framework...")
        
        # Store feature columns for later use
        self.focus_feature_columns = focus_features
        self.body_feature_columns = body_features
        self.precaution_columns = [col for col in self.asanas_processed.columns if col.startswith('precaution_')]
        
        print(f"   ✓ User features: {len(user_features)} dimensions")
        print(f"   ✓ Asana features: {len(available_features)} dimensions")
        print(f"   ✓ Focus area features: {len(focus_features)}")
        print(f"   ✓ Body part features: {len(body_features)}")
        print(f"   ✓ Safety precaution features: {len(self.precaution_columns)}")
        
        return self.user_features_df, self.asana_features_df
    
    def visualize_data_insights(self):
        """Create comprehensive visualizations for data insights"""
        print("\n6. Creating Data Insights Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Yoga Recommender System - Data Insights', fontsize=16, fontweight='bold')
        
        # 1. User fitness distribution
        axes[0, 0].hist(self.users_df['fitness_composite'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('User Fitness Composite Score Distribution')
        axes[0, 0].set_xlabel('Fitness Score')
        axes[0, 0].set_ylabel('Number of Users')
        axes[0, 0].axvline(self.users_df['fitness_composite'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.users_df["fitness_composite"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Yoga difficulty level distribution
        difficulty_counts = self.users_df['yoga_difficulty_level'].value_counts()
        axes[0, 1].pie(difficulty_counts.values, labels=difficulty_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('User Yoga Difficulty Level Distribution')
        
        # 3. Asana complexity vs safety
        axes[0, 2].scatter(self.asanas_processed['complexity_score'], self.asanas_processed['safety_score'], 
                          alpha=0.6, c='green', s=50)
        axes[0, 2].set_title('Asana Complexity vs Safety Score')
        axes[0, 2].set_xlabel('Complexity Score')
        axes[0, 2].set_ylabel('Safety Score')
        
        # 4. Risk level distribution
        risk_counts = self.users_df['practice_risk_level'].value_counts()
        axes[1, 0].bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
        axes[1, 0].set_title('User Practice Risk Level Distribution')
        axes[1, 0].set_xlabel('Risk Level')
        axes[1, 0].set_ylabel('Number of Users')
        
        # 5. Focus area coverage
        focus_area_counts = {}
        for focus_list in self.asanas_df['focus_area_list']:
            for area in focus_list:
                focus_area_counts[area] = focus_area_counts.get(area, 0) + 1
        
        sorted_focus = sorted(focus_area_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        areas, counts = zip(*sorted_focus)
        
        axes[1, 1].barh(areas, counts, color='lightcoral')
        axes[1, 1].set_title('Top 10 Focus Areas Coverage')
        axes[1, 1].set_xlabel('Number of Poses')
        
        # 6. Age appropriateness distribution
        age_ranges = self.asanas_processed['max_age'] - self.asanas_processed['min_age']
        axes[1, 2].hist(age_ranges, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 2].set_title('Pose Age Range Distribution')
        axes[1, 2].set_xlabel('Age Range (years)')
        axes[1, 2].set_ylabel('Number of Poses')
        
        plt.tight_layout()
        plt.show()
        
        # Additional insights
        print("\n   📊 DATA INSIGHTS SUMMARY:")
        print(f"   • Total users: {len(self.users_df)}")
        print(f"   • Total poses: {len(self.asanas_processed)}")
        print(f"   • Average user fitness: {self.users_df['fitness_composite'].mean():.1f}")
        print(f"   • High-risk users: {(self.users_df['practice_risk_level'] == 'high').sum()}")
        print(f"   • Beginner-friendly poses: {(self.asanas_processed['difficulty_level'] == 'Beginner').sum()}")
        print(f"   • Poses with precautions: {(self.asanas_processed['safety_score'] < 1.0).sum()}")
        
    def save_processed_data(self):
        """Save all processed data and feature matrices"""
        print("\n7. Saving Processed Data...")
        
        # Save processed datasets
        self.users_df.to_csv('processed_users_data1.csv', index=False)
        self.asanas_processed.to_csv('processed_asanas_data1.csv', index=False)
        
        # Save feature matrices
        self.user_features_df.to_csv('user_features_matrix.csv', index=False)
        self.asana_features_df.to_csv('asana_features_matrix.csv', index=False)
        
        # Save metadata
        metadata = {
            'focus_feature_columns': self.focus_feature_columns,
            'body_feature_columns': self.body_feature_columns,
            'precaution_columns': self.precaution_columns,
            'safety_conditions': self.safety_conditions,
            'focus_areas': self.focus_areas,
            'fitness_to_yoga_mapping': self.fitness_to_yoga_mapping
        }
        
        import json
        with open('recommendation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("   ✓ Processed users data saved")
        print("   ✓ Processed asanas data saved") 
        print("   ✓ Feature matrices saved")
        print("   ✓ Metadata saved")
        
    def run_complete_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 STARTING COMPLETE PREPROCESSING PIPELINE...")
        
        # Step 1: Load and validate data
        if not self.load_and_validate_data():
            return False
        
        # Step 2: Preprocess users data
        self.preprocess_users_data()
        
        # Step 3: Preprocess asanas data
        self.preprocess_asanas_data()
        
        # Step 4: Create matching features
        self.create_user_asana_features()
        
        # Step 5: Visualize insights
        self.visualize_data_insights()
        
        # Step 6: Save processed data
        self.save_processed_data()
        
        print("\n🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Ready for recommendation system development")
        
        return True

# Initialize and run the preprocessing pipeline
if __name__ == "__main__":
    preprocessor = YogaRecommenderPreprocessor()
    preprocessor.run_complete_preprocessing()
