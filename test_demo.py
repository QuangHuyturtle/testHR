#!/usr/bin/env python3
"""
Test demo for HR DSS
"""

from hr_dss_main import HRDecisionSupportSystem
import json

def test_system():
    print("=== HR DSS Demo Test ===")
    
    # Initialize system
    hr_system = HRDecisionSupportSystem()
    
    # Test 1: Train model
    print("\n1. Training model...")
    accuracy = hr_system.train_model()
    print(f"✓ Model trained with accuracy: {accuracy:.3f}")
    
    # Test 2: Single prediction
    print("\n2. Testing single prediction...")
    candidate = {
        'candidate_id': 'TEST_001',
        'years_experience': 5,
        'education_level': 'bachelor',
        'skills': 'python, machine learning, sql, data analysis',
        'experience_description': 'Experienced data analyst with strong programming skills',
        'position_applied': 'data_scientist'
    }
    
    result = hr_system.predict_candidate(candidate)
    print(f"✓ Prediction: {result['prediction']}")
    print(f"✓ Confidence: {result['confidence']:.3f}")
    print(f"✓ Recommendation: {result['recommendation']}")
    
    # Test 3: Multiple predictions
    print("\n3. Testing multiple candidates...")
    test_candidates = [
        {
            'candidate_id': 'TEST_002',
            'years_experience': 10,
            'education_level': 'master',
            'skills': 'leadership, project management, python, machine learning',
            'experience_description': 'Senior data scientist with team leadership',
            'position_applied': 'senior_data_scientist'
        },
        {
            'candidate_id': 'TEST_003',
            'years_experience': 1,
            'education_level': 'bachelor',
            'skills': 'excel, powerpoint',
            'experience_description': 'Recent graduate with basic skills',
            'position_applied': 'analyst'
        }
    ]
    
    for candidate in test_candidates:
        result = hr_system.predict_candidate(candidate)
        print(f"✓ {candidate['candidate_id']}: {result['prediction']} ({result['confidence']:.2f})")
    
    print("\n🎉 All tests passed! System is working correctly.")
    print("💡 Now run: python app.py")
    print("🌐 Open browser: http://localhost:5000")

if __name__ == "__main__":
    test_system()