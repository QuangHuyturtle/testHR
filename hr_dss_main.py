#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Decision Support System - Main Application
Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự

Author: Student
Date: 2025
Course: Hệ hỗ trợ ra quyết định, Hệ điều hành và lập trình Linux
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import logging
from datetime import datetime
import argparse
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not exists
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Setup logging
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hr_dss.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HRDecisionSupportSystem:
    """
    Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự
    """
    
    def __init__(self, model_path="models/", data_path="data/"):
        """
        Khởi tạo hệ thống HR DSS
        
        Args:
            model_path (str): Đường dẫn lưu mô hình
            data_path (str): Đường dẫn dữ liệu
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        
        # Tạo thư mục nếu chưa tồn tại
        self.model_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
        # Load model if exists
        self.load_model()
        
        logger.info("HR Decision Support System initialized")
    
    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản (CV, mô tả kỹ năng)
        
        Args:
            text (str): Văn bản cần xử lý
            
        Returns:
            str: Văn bản đã xử lý
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Chuyển về chữ thường
        text = str(text).lower()
        
        # Loại bỏ ký tự đặc biệt, giữ lại chữ cái và số
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize và loại bỏ stop words
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            return text
    
    def extract_features(self, df):
        """
        Trích xuất đặc trưng từ dữ liệu ứng viên
        
        Args:
            df (pd.DataFrame): Dữ liệu ứng viên
            
        Returns:
            pd.DataFrame: Dữ liệu đã trích xuất đặc trưng
        """
        logger.info("Extracting features from candidate data...")
        
        # Tạo bản sao để không ảnh hưởng dữ liệu gốc
        processed_df = df.copy()
        
        # Tiền xử lý văn bản
        processed_df['skills_processed'] = processed_df['skills'].apply(self.preprocess_text)
        processed_df['experience_description_processed'] = processed_df['experience_description'].apply(self.preprocess_text)
        
        # Kết hợp các trường văn bản
        processed_df['combined_text'] = (
            processed_df['skills_processed'] + ' ' + 
            processed_df['experience_description_processed']
        )
        
        # Tính toán các đặc trưng số
        processed_df['education_score'] = processed_df['education_level'].map({
            'high_school': 1,
            'associate': 2, 
            'bachelor': 3,
            'master': 4,
            'phd': 5
        }).fillna(1)
        
        # Chuẩn hóa kinh nghiệm (năm)
        processed_df['years_experience'] = pd.to_numeric(processed_df['years_experience'], errors='coerce').fillna(0)
        
        # Tính điểm kỹ năng (số lượng kỹ năng)
        processed_df['num_skills'] = processed_df['skills'].str.count(',') + 1
        processed_df['num_skills'] = processed_df['num_skills'].fillna(0)
        
        return processed_df
    
    def create_sample_data(self, num_samples=1000):
        """
        Tạo dữ liệu mẫu cho training
        
        Args:
            num_samples (int): Số lượng mẫu
            
        Returns:
            pd.DataFrame: Dữ liệu mẫu
        """
        logger.info(f"Creating sample data with {num_samples} samples...")
        
        np.random.seed(42)
        
        # Danh sách kỹ năng phổ biến
        skills_pool = [
            'python', 'java', 'javascript', 'sql', 'machine learning', 'data analysis',
            'project management', 'communication', 'teamwork', 'leadership',
            'react', 'nodejs', 'docker', 'kubernetes', 'aws', 'azure',
            'agile', 'scrum', 'git', 'linux', 'statistics', 'excel'
        ]
        
        education_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']
        positions = ['developer', 'analyst', 'manager', 'designer', 'consultant']
        
        data = []
        
        for i in range(num_samples):
            # Random features
            years_exp = np.random.randint(0, 20)
            education = np.random.choice(education_levels)
            position = np.random.choice(positions)
            
            # Random skills (2-8 skills per candidate)
            num_skills = np.random.randint(2, 9)
            candidate_skills = np.random.choice(skills_pool, num_skills, replace=False)
            skills_str = ', '.join(candidate_skills)
            
            # Experience description
            exp_desc = f"Worked as {position} for {years_exp} years with expertise in {', '.join(candidate_skills[:3])}"
            
            # Decision logic for labeling
            score = 0
            if years_exp >= 3: score += 2
            if education in ['bachelor', 'master', 'phd']: score += 2
            if num_skills >= 5: score += 1
            if 'python' in candidate_skills or 'java' in candidate_skills: score += 1
            if 'leadership' in candidate_skills or 'project management' in candidate_skills: score += 1
            
            # Label: suitable if score >= 4
            suitable = 1 if score >= 4 else 0
            
            data.append({
                'candidate_id': f'CAND_{i+1:04d}',
                'years_experience': years_exp,
                'education_level': education,
                'skills': skills_str,
                'experience_description': exp_desc,
                'position_applied': position,
                'suitable': suitable
            })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        sample_file = self.data_path / 'sample_candidates.csv'
        df.to_csv(sample_file, index=False)
        logger.info(f"Sample data saved to {sample_file}")
        
        return df
    
    def train_model(self, df=None):
        """
        Training mô hình phân loại
        
        Args:
            df (pd.DataFrame): Dữ liệu training (None để dùng sample data)
        """
        logger.info("Starting model training...")
        
        if df is None:
            df = self.create_sample_data()
        
        # Extract features
        processed_df = self.extract_features(df)
        
        # Prepare text features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        text_features = self.vectorizer.fit_transform(processed_df['combined_text'])
        
        # Prepare numerical features
        numerical_cols = ['education_score', 'years_experience', 'num_skills']
        self.feature_columns = numerical_cols
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        numerical_features = self.scaler.fit_transform(processed_df[numerical_cols])
        
        # Combine features
        X_text = text_features.toarray()
        X_numerical = numerical_features
        X = np.hstack([X_text, X_numerical])
        
        # Target variable
        y = processed_df['suitable']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully!")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def predict_candidate(self, candidate_data):
        """
        Dự đoán độ phù hợp của ứng viên
        
        Args:
            candidate_data (dict): Thông tin ứng viên
            
        Returns:
            dict: Kết quả dự đoán
        """
        if self.model is None:
            raise ValueError("Model chưa được training. Hãy chạy train_model() trước.")
        
        # Convert to DataFrame
        df = pd.DataFrame([candidate_data])
        
        # Extract features
        processed_df = self.extract_features(df)
        
        # Prepare features
        text_features = self.vectorizer.transform(processed_df['combined_text'])
        numerical_features = self.scaler.transform(processed_df[self.feature_columns])
        
        # Combine features
        X_text = text_features.toarray()
        X_numerical = numerical_features
        X = np.hstack([X_text, X_numerical])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = {
            'candidate_id': candidate_data.get('candidate_id', 'Unknown'),
            'prediction': 'Suitable' if prediction == 1 else 'Not Suitable',
            'confidence': max(probability),
            'probability_suitable': probability[1] if len(probability) > 1 else probability[0],
            'recommendation': self.get_recommendation(prediction, max(probability))
        }
        
        logger.info(f"Prediction for {result['candidate_id']}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return result
    
    def get_recommendation(self, prediction, confidence):
        """
        Đưa ra khuyến nghị dựa trên dự đoán
        
        Args:
            prediction (int): Kết quả dự đoán (0 hoặc 1)
            confidence (float): Độ tin cậy
            
        Returns:
            str: Khuyến nghị
        """
        if prediction == 1:
            if confidence > 0.8:
                return "Highly recommended for interview"
            elif confidence > 0.6:
                return "Recommended for interview"
            else:
                return "Consider for interview with caution"
        else:
            if confidence > 0.8:
                return "Not recommended"
            else:
                return "Need further evaluation"
    
    def batch_predict(self, csv_file):
        """
        Dự đoán hàng loạt từ file CSV
        
        Args:
            csv_file (str): Đường dẫn file CSV
            
        Returns:
            pd.DataFrame: Kết quả dự đoán
        """
        logger.info(f"Processing batch prediction from {csv_file}")
        
        df = pd.read_csv(csv_file)
        results = []
        
        for idx, row in df.iterrows():
            candidate_data = row.to_dict()
            try:
                result = self.predict_candidate(candidate_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting candidate {candidate_data.get('candidate_id', idx)}: {e}")
                results.append({
                    'candidate_id': candidate_data.get('candidate_id', f'CAND_{idx}'),
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'probability_suitable': 0.0,
                    'recommendation': 'Processing error'
                })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.data_path / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Batch prediction results saved to {output_file}")
        
        return results_df
    
    def save_model(self):
        """
        Lưu mô hình và các thành phần
        """
        try:
            # Save model components
            with open(self.model_path / 'rf_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.model_path / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(self.model_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(self.model_path / 'feature_columns.json', 'w') as f:
                json.dump(self.feature_columns, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """
        Load mô hình đã lưu
        """
        try:
            if (self.model_path / 'rf_model.pkl').exists():
                with open(self.model_path / 'rf_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.model_path / 'vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.model_path / 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open(self.model_path / 'feature_columns.json', 'r') as f:
                    self.feature_columns = json.load(f)
                
                logger.info("Model loaded successfully")
                return True
                
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
    
    def generate_report(self, results_df=None):
        """
        Tạo báo cáo tổng hợp
        
        Args:
            results_df (pd.DataFrame): Kết quả dự đoán
            
        Returns:
            dict: Báo cáo tổng hợp
        """
        if results_df is None:
            logger.warning("No results data provided for report")
            return {}
        
        total_candidates = len(results_df)
        suitable_candidates = len(results_df[results_df['prediction'] == 'Suitable'])
        avg_confidence = results_df['confidence'].mean()
        
        report = {
            'total_candidates': total_candidates,
            'suitable_candidates': suitable_candidates,
            'suitable_percentage': (suitable_candidates / total_candidates * 100) if total_candidates > 0 else 0,
            'average_confidence': avg_confidence,
            'high_confidence_suitable': len(results_df[
                (results_df['prediction'] == 'Suitable') & 
                (results_df['confidence'] > 0.8)
            ]),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.data_path / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated and saved to {report_file}")
        return report


def main():
    """
    Hàm main để chạy ứng dụng
    """
    parser = argparse.ArgumentParser(description='HR Decision Support System')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'demo'], 
                       default='demo', help='Chế độ chạy')
    parser.add_argument('--input', type=str, help='File input cho batch prediction')
    parser.add_argument('--model-path', type=str, default='models/', 
                       help='Đường dẫn thư mục model')
    parser.add_argument('--data-path', type=str, default='data/', 
                       help='Đường dẫn thư mục data')
    
    args = parser.parse_args()
    
    # Initialize system
    hr_system = HRDecisionSupportSystem(args.model_path, args.data_path)
    
    if args.mode == 'train':
        logger.info("Training mode selected")
        accuracy = hr_system.train_model()
        print(f"Model training completed with accuracy: {accuracy:.3f}")
    
    elif args.mode == 'predict':
        logger.info("Single prediction mode")
        # Example candidate
        candidate = {
            'candidate_id': 'DEMO_001',
            'years_experience': 5,
            'education_level': 'bachelor',
            'skills': 'python, machine learning, sql, data analysis, teamwork',
            'experience_description': 'Experienced data analyst with strong programming skills',
            'position_applied': 'analyst'
        }
        
        result = hr_system.predict_candidate(candidate)
        print(json.dumps(result, indent=2))
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input file required for batch mode")
            return
        
        logger.info(f"Batch prediction mode for file: {args.input}")
        results = hr_system.batch_predict(args.input)
        report = hr_system.generate_report(results)
        print("Batch prediction completed")
        print(json.dumps(report, indent=2))
    
    else:  # demo mode
        logger.info("Demo mode - Training and testing")
        
        # Train model
        accuracy = hr_system.train_model()
        print(f"Model trained with accuracy: {accuracy:.3f}")
        
        # Demo predictions
        demo_candidates = [
            {
                'candidate_id': 'DEMO_001',
                'years_experience': 8,
                'education_level': 'master',
                'skills': 'python, machine learning, leadership, project management, sql',
                'experience_description': 'Senior data scientist with team leadership experience',
                'position_applied': 'manager'
            },
            {
                'candidate_id': 'DEMO_002', 
                'years_experience': 1,
                'education_level': 'bachelor',
                'skills': 'excel, communication',
                'experience_description': 'Recent graduate with basic skills',
                'position_applied': 'analyst'
            }
        ]
        
        print("\n=== Demo Predictions ===")
        for candidate in demo_candidates:
            result = hr_system.predict_candidate(candidate)
            print(f"\nCandidate: {result['candidate_id']}")
            print(f"Decision: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()