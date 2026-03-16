
import pandas as pd
import numpy as np
import random

def generate_student_data(num_students=500):
    # --- Configuration ---
    branches = ['Computer Science', 'Electronics', 'Mechanical', 'Civil', 'Information Technology']
    years = [1, 2, 3, 4]
    socio_economic_levels = ['Low', 'Middle', 'High']
    parent_education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    
    data = []

    for i in range(num_students):
        student_id = f"STU_{1000+i}"
        name = f"Student_{i+1}"
        branch = random.choice(branches)
        current_year = random.choice(years)
        
        # Socio-Economic Factors
        family_income_level = random.choice(socio_economic_levels)
        parent_education = random.choice(parent_education_levels)
        access_to_resources = random.choice(['Yes', 'No']) # Internet, Laptop etc.
        community_involvement = random.choice(['Low', 'Moderate', 'High'])
        
        # Personal/Psychological Factors
        study_hours_per_week = np.random.randint(5, 40)
        attendance_rate = np.random.randint(60, 100) # Percentage
        stress_level = random.choice(['Low', 'Moderate', 'High'])
        health_status = random.choice(['Good', 'Average', 'Poor'])
        
        # Previous Academic Data (Semester-wise)
        # We start with a base capability and add noise for each semester
        base_capability = np.random.normal(7.0, 1.5) # Mean GPA 7.0
        
        # Adjust capability based on factors (Synthetic correlations)
        if family_income_level == 'High': base_capability += 0.5
        if parent_education in ['Master', 'PhD']: base_capability += 0.5
        if study_hours_per_week > 20: base_capability += 1.0
        if stress_level == 'High': base_capability -= 0.8
        
        # Clip to realistic range
        base_capability = max(0.0, min(10.0, base_capability))
        
        # Generate Semester GPAs
        semesters_completed = (current_year - 1) * 2
        if semesters_completed == 0: semesters_completed = 1 # Just started
        
        sem_grades = {}
        avg_gpa = 0
        
        for sem in range(1, 9): # Up to 8 semesters
            if sem <= semesters_completed:
                # Add some variance per semester
                sem_gpa = np.random.normal(base_capability, 0.5)
                sem_gpa = max(4.0, min(10.0, sem_gpa)) # Standard 10 point scale typically
                sem_grades[f'Sem_{sem}_GPA'] = round(sem_gpa, 2)
            else:
                sem_grades[f'Sem_{sem}_GPA'] = None # Future semesters

        # Calculate current cumulative metrics
        valid_gpas = [v for k,v in sem_grades.items() if v is not None]
        cgpa = round(sum(valid_gpas) / len(valid_gpas), 2) if valid_gpas else 0
        
        # Define Performance Category
        if cgpa >= 8.5:
            category = 'Excellent'
        elif cgpa >= 7.0:
            category = 'Good'
        elif cgpa >= 5.0:
            category = 'Average'
        else:
            category = 'Low/At-Risk'

        # Row Data
        row = {
            'Student_ID': student_id,
            'Name': name,
            'Branch': branch,
            'Current_Year': current_year,
            'Family_Income': family_income_level,
            'Parent_Education': parent_education,
            'Resources_Access': access_to_resources,
            'Community_Involvement': community_involvement,
            'Study_Hours_Week': study_hours_per_week,
            'Attendance_Rate': attendance_rate,
            'Stress_Level': stress_level,
            'Health_Status': health_status,
            'CGPA': cgpa,
            'Performance_Category': category
        }
        # Merge semester grades
        row.update(sem_grades)
        
        data.append(row)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating comprehensive synthetic dataset...")
    df = generate_student_data(1000)
    
    # Save to data folder
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "data", "new_student_data.csv")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} records and saved to {output_path}")
    print("Columns:", df.columns.tolist())
    print(df.head())
