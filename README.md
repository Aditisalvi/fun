# Dataset info

## User's dataset - 
### Dataset Overview
Data Shape: (13393, 12) – The dataset contains 13,393 records and 12 attributes.
Age Range: 20 to 64 years.
Gender: F (Female), M (Male).
Target Variable (Class): A (best), B, C, D (worst). This represents the performance grade, with A being the best and D being the lowest.
### Columns and Description
- Age: Age of the individual (in years).
- Gender: Gender of the individual (F for female, M for male).
- Height (cm): Height of the individual in centimeters (To convert to feet, divide by 30.48).
- Weight (kg): Weight of the individual in kilograms.
- Body Fat (%): Percentage of body fat in the individual.
- Diastolic (mmHg): Diastolic blood pressure (minimum pressure).
- Systolic (mmHg): Systolic blood pressure (maximum pressure).
- Grip Force (kg): Grip force in kilograms.
- Sit and Bend Forward (cm): The distance the individual can bend forward, measured in centimeters.
- Sit-ups Count: Number of sit-ups the individual can perform.
- Broad Jump (cm): The distance the individual can jump, measured in centimeters.
- Class: Performance grade classification (A, B, C, D).

## Asana dataset - 
### Dataset Overview
Data Shape: (141, 13) – The dataset contains 141 records and 13 attributes.
Age Range: 5 to 80 years.
Target Variable: None explicitly defined in the dataset. The dataset focuses on yoga poses (asanas) with their characteristics, benefits, and precautions, suitable for various age groups and difficulty levels.

### Columns and Description
- asana_name: The name of the yoga pose (e.g., Mountain Pose, Tree Pose).
- asana_type: The style or category of yoga the pose belongs to (e.g., Hatha, Power).
- pose_type: The physical category of the pose (e.g., Standing, Balance, Backbend, Seated, Supine, Prone, Twist, Flow, Side Bend, Arm Balance, Inversion, Squat).
- description: A brief description of how to perform the yoga pose.
- benefits: The physical, mental, or emotional benefits of performing the pose (e.g., improves posture, strengthens legs, calms mind).
- target_age_group: The recommended age range for performing the pose (e.g., 8-80, 15-65).
- difficulty_level: The difficulty level of the pose (Beginner, Intermediate, Advanced).
- duration_secs: The recommended duration to hold the pose, in seconds.
- repetition: The recommended number of repetitions, varying by age group (e.g., "8-45: 5 reps, 45+: 3 reps").
- Body_Parts: The body parts engaged or targeted by the pose (e.g., Ankles/Feet, Core/Abdomen, Hips).
- Focus_Area: The primary goals or effects of the pose (e.g., Strength Building, Stress Relief/Calming, Flexibility/Stretching).
- Precautions: Health conditions or injuries to consider before attempting the pose (e.g., High blood pressure, Knee problems/injuries).
- Pain_Points: Common physical or mental issues the pose may help address (e.g., Lower back pain, Stress/anxiety, Balance issues).

```
Body Parts Coverage:
Unique body parts found: 18
  Ankles/Feet: 5 poses
  Arms/Wrists: 26 poses
  Back/Spine: 53 poses
  Calves: 2 poses
  Chest/Heart: 19 poses
  Core/Abdomen: 22 poses
  Glutes/Buttocks: 6 poses
  Hamstrings: 13 poses
  Hip Flexors: 6 poses
  Hips: 26 poses
  IT Band: 1 poses
  Intercostal Muscles: 1 poses
  Knees: 1 poses
  Legs/Thighs: 36 poses
  Neck: 2 poses
  Nervous System: 9 poses
  Quadriceps: 2 poses
  Shoulders: 13 poses
```

```
Focus Areas Coverage:
Unique focus areas found: 16
  Balance/Stability: 25 poses
  Breathing Improvement: 1 poses
  Cardiovascular Fitness: 4 poses
  Circulation Enhancement: 3 poses
  Coordination: 4 poses
  Detoxification: 2 poses
  Digestion Support: 12 poses
  Emotional Release: 5 poses
  Endurance Building: 4 poses
  Energy Building: 3 poses
  Flexibility/Stretching: 38 poses
  Meditation/Focus: 9 poses
  Posture Improvement: 10 poses
  Relaxation/Restorative: 4 poses
  Strength Building: 62 poses
  Stress Relief/Calming: 22 poses
```

```
Precautions Coverage:
Unique precautions found: 21
  Ankle injuries: 5 poses
  Asthma: 1 poses
  Back injuries: 26 poses
  Balance disorders: 4 poses
  Carpal tunnel: 5 poses
  Diarrhea: 1 poses
  Generally safe for all: 38 poses
  Glaucoma/eye conditions: 2 poses
  Hamstring injuries: 2 poses
  Heart conditions: 4 poses
  High blood pressure: 9 poses
  Hip injuries: 3 poses
  Insomnia: 1 poses
  Knee problems/injuries: 31 poses
  Low blood pressure: 8 poses
  Menstruation: 1 poses
  Migraine: 1 poses
  Neck injuries: 11 poses
  Pregnancy: 10 poses
  Shoulder injuries: 12 poses
  Wrist injuries: 5 poses
```

```
Pain Points Coverage:
Unique pain points found: 39
  Ankle stiffness: 3 poses
  Arm weakness: 14 poses
  Balance issues: 20 poses
  Calf tightness: 1 poses
  Cardiovascular fitness: 4 poses
  Chest tightness: 19 poses
  Core weakness: 19 poses
  Depression: 1 poses
  Digestive issues: 16 poses
  Emotional stress: 2 poses
  Fear/confidence issues: 3 poses
  Hamstring tightness: 14 poses
  High blood pressure: 1 poses
  Hip flexor tightness: 9 poses
  Hip tightness: 24 poses
  IT band tightness: 4 poses
  Insomnia: 1 poses
  Knee stiffness: 2 poses
  Leg weakness: 21 poses
  Low energy: 6 poses
  Lower back pain: 23 poses
  Neck stiffness: 3 poses
  Overstimulation: 1 poses
  Poor circulation: 8 poses
  Poor concentration: 1 poses
  Poor coordination: 3 poses
  Poor posture: 18 poses
  Quadricep tightness: 5 poses
  Respiratory issues: 2 poses
  Sciatica: 2 poses
  Shallow breathing: 1 poses
  Shoulder tension/stiffness: 15 poses
  Side body tightness: 6 poses
  Spinal stiffness: 24 poses
  Stress/anxiety: 28 poses
  Tension headaches: 1 poses
  Thigh tightness: 3 poses
  Upper back pain/stiffness: 13 poses
  Wrist stiffness/pain: 2 poses
```
