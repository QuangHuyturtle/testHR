#!/usr/bin/env python3
"""
Create sample data for HR DSS
Tạo dữ liệu mẫu cho hệ thống
"""

import pandas as pd
import numpy as np
import os

def create_sample_data(num_samples=100):
    """Tạo dữ liệu mẫu"""
    
    print(f"Creating {num_samples} sample records...")
    
    # Seed for reproducible results
    np.random.seed(42)
    
    # Sample data pools
    skills_pool = [
        'python', 'java', 'javascript', 'sql', 'machine learning', 'data analysis',
        'project management', 'communication', 'teamwork', 'leadership', 'excel',
        'react', 'nodejs', 'docker', 'kubernetes', 'aws', 'azure', 'git',
        'agile', 'scrum', 'statistics', 'tableau', 'powerbi', 'hadoop',
        'spark', 'tensorflow', 'pytorch', 'django', 'spring', 'angular'
    ]
    
    education_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']
    education_weights = [0.1, 0.15, 0.5, 0.2, 0.05]  # Realistic distribution
    
    positions = ['analyst', 'developer', 'manager', 'scientist', 'consultant', 
                'engineer', 'specialist', 'coordinator', 'lead', 'director']
    
    # Generate data
    data = []
    
    for i in range(num_samples):
        # Basic info
        candidate_id = f'CAND_{i+1:04d}'
        years_exp = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20], 
                                   p=[0.05, 0.1, 0.1, 0.12, 0.12, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
        
        education = np.random.choice(education_levels, p=education_weights)
        position = np.random.choice(positions)
        
        # Skills (2-8 skills per candidate)
        num_skills = np.random.randint(2, 9)
        candidate_skills = np.random.choice(skills_pool, num_skills, replace=False)
        skills_str = ', '.join(candidate_skills)
        
        # Experience description
        exp_templates = [
            f"Experienced {position} with {years_exp} years in {candidate_skills[0]} and {candidate_skills[1] if len(candidate_skills) > 1 else 'various technologies'}",
            f"{years_exp}-year {position} specializing in {candidate_skills[0]} with strong background in {candidate_skills[1] if len(candidate_skills) > 1 else 'technology'}",
            f"Professional {position} with expertise in {', '.join(candidate_skills[:3])} and {years_exp} years of experience",
            f"Skilled {position} with {years_exp} years experience in {candidate_skills[0]} and proven track record in {candidate_skills[1] if len(candidate_skills) > 1 else 'software development'}"
        ]
        
        experience_desc = np.random.choice(exp_templates)
        
        # Determine suitability based on realistic criteria
        suitability_score = 0
        
        # Experience points
        if years_exp >= 5: suitability_score += 3
        elif years_exp >= 2: suitability_score += 2
        elif years_exp >= 1: suitability_score += 1
        
        # Education points
        edu_points = {'phd': 4, 'master': 3, 'bachelor': 2, 'associate': 1, 'high_school': 0}
        suitability_score += edu_points[education]
        
        # Skills points
        valuable_skills = ['python', 'machine learning', 'java', 'leadership', 'project management', 'sql']
        skill_bonus = len([s for s in candidate_skills if s in valuable_skills])
        suitability_score += skill_bonus
        
        # Number of skills bonus
        if num_skills >= 6: suitability_score += 2
        elif num_skills >= 4: suitability_score += 1
        
        # Add some randomness
        suitability_score += np.random.randint(-2, 3)
        
        # Final decision (threshold = 6)
        suitable = 1 if suitability_score >= 6 else 0
        
        # Add some noise to make it more realistic
        if np.random.random() < 0.1:  # 10% random flip
            suitable = 1 - suitable
        
        data.append({
            'candidate_id': candidate_id,
            'years_experience': years_exp,
            'education_level': education,
            'skills': skills_str,
            'experience_description': experience_desc,
            'position_applied': position,
            'suitable': suitable
        })
    
    return pd.DataFrame(data)

def main():
    """Main function"""
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    train_df = create_sample_data(200)
    train_file = 'data/training_data.csv'
    train_df.to_csv(train_file, index=False)
    print(f"✓ Training data saved: {train_file}")
    print(f"  - Total samples: {len(train_df)}")
    print(f"  - Suitable: {train_df['suitable'].sum()}")
    print(f"  - Not suitable: {len(train_df) - train_df['suitable'].sum()}")
    
    # Generate test data
    print("\nGenerating test data...")
    test_df = create_sample_data(50)
    test_file = 'data/test_candidates.csv'
    test_df.to_csv(test_file, index=False)
    print(f"✓ Test data saved: {test_file}")
    
    # Create small demo file
    print("\nGenerating demo data...")
    demo_df = create_sample_data(10)
    demo_file = 'data/demo_candidates.csv'
    demo_df.to_csv(demo_file, index=False)
    print(f"✓ Demo data saved: {demo_file}")
    
    # Display sample
    print("\n📋 Sample records:")
    print(demo_df[['candidate_id', 'years_experience', 'education_level', 'suitable']].head())
    
    print("\n🎉 Sample data creation completed!")
    print("\nFiles created:")
    print(f"  📄 {train_file} - Training data (200 records)")
    print(f"  📄 {test_file} - Test data (50 records)")  
    print(f"  📄 {demo_file} - Demo data (10 records)")

if __name__ == "__main__":
    main()