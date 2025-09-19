#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Decision Support System - Main Application (Vietnamese Version) - Updated
Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự - Phiên bản cải thiện

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
    print("Đang tải dữ liệu NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    print("✓ Tải dữ liệu NLTK hoàn tất")

# Setup logging
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hr_dss.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HRDecisionSupportSystem:
    """
    Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự - Phiên bản cải thiện
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
        
        # Load position requirements
        self.position_requirements = self.get_position_requirements()
        
        # Load model if exists
        self.load_model()
        
        logger.info("🚀 Hệ thống Hỗ trợ Ra quyết định Tuyển dụng đã được khởi tạo (Phiên bản cải thiện)")

    def get_position_requirements(self):
        """
        Định nghĩa yêu cầu kinh nghiệm tối thiểu cho từng loại vị trí
        """
        return {
            # Vị trí cấp cao - yêu cầu kinh nghiệm cao
            'senior_developer': {'min_years': 5, 'min_education': 'bachelor'},
            'senior_analyst': {'min_years': 5, 'min_education': 'bachelor'}, 
            'data_scientist': {'min_years': 3, 'min_education': 'bachelor'},
            'scientist': {'min_years': 3, 'min_education': 'bachelor'},
            'lead': {'min_years': 4, 'min_education': 'bachelor'},
            'manager': {'min_years': 3, 'min_education': 'bachelor'},
            'director': {'min_years': 7, 'min_education': 'master'},
            
            # Vị trí trung cấp
            'developer': {'min_years': 1, 'min_education': 'bachelor'},
            'analyst': {'min_years': 1, 'min_education': 'bachelor'},
            'consultant': {'min_years': 2, 'min_education': 'bachelor'},
            'specialist': {'min_years': 2, 'min_education': 'bachelor'},
            'engineer': {'min_years': 1, 'min_education': 'bachelor'},
            
            # Vị trí cơ bản - có thể chấp nhận ít kinh nghiệm hơn
            'junior_developer': {'min_years': 0, 'min_education': 'bachelor'},
            'junior_analyst': {'min_years': 0, 'min_education': 'bachelor'},
            'intern': {'min_years': 0, 'min_education': 'high_school'},
            'fresher': {'min_years': 0, 'min_education': 'bachelor'},
            'coordinator': {'min_years': 0, 'min_education': 'associate'},
            'designer': {'min_years': 0, 'min_education': 'associate'},
        }

    def get_education_score(self, education_level):
        """Chuyển đổi trình độ học vấn thành điểm số"""
        education_mapping = {
            'high_school': 1,
            'associate': 2, 
            'bachelor': 3,
            'master': 4,
            'phd': 5
        }
        return education_mapping.get(education_level.lower(), 1)

    def improved_suitability_logic(self, candidate_data):
        """
        Logic đánh giá cải thiện với yêu cầu kinh nghiệm theo vị trí
        """
        years_exp = int(candidate_data.get('years_experience', 0))
        education = candidate_data.get('education_level', 'high_school').lower()
        skills = candidate_data.get('skills', '').lower()
        position = candidate_data.get('position_applied', '').lower()
        
        # Tìm yêu cầu phù hợp nhất cho vị trí
        position_req = None
        for req_position, req_data in self.position_requirements.items():
            if req_position in position or position in req_position:
                position_req = req_data
                break
        
        # Nếu không tìm thấy yêu cầu cụ thể, sử dụng yêu cầu mặc định
        if position_req is None:
            # Phân loại dựa trên từ khóa trong tên vị trí
            if any(word in position for word in ['senior', 'lead', 'manager', 'director']):
                position_req = {'min_years': 4, 'min_education': 'bachelor'}
            elif any(word in position for word in ['scientist', 'specialist']):
                position_req = {'min_years': 3, 'min_education': 'bachelor'}
            elif any(word in position for word in ['junior', 'intern', 'fresher']):
                position_req = {'min_years': 0, 'min_education': 'bachelor'}
            else:
                position_req = {'min_years': 1, 'min_education': 'bachelor'}  # Mặc định
        
        # Tính điểm đánh giá
        score = 0
        max_score = 10
        feedback = []
        
        # 1. Kiểm tra kinh nghiệm (40% trọng số)
        min_years = position_req['min_years']
        if years_exp >= min_years + 2:
            score += 4  # Vượt yêu cầu
            feedback.append(f"✓ Kinh nghiệm vượt yêu cầu ({years_exp} năm >= {min_years} năm)")
        elif years_exp >= min_years:
            score += 3  # Đạt yêu cầu
            feedback.append(f"✓ Kinh nghiệm đạt yêu cầu ({years_exp} năm >= {min_years} năm)")
        elif years_exp >= min_years - 1 and min_years > 0:
            score += 2  # Gần đạt yêu cầu
            feedback.append(f"⚠ Kinh nghiệm gần đạt yêu cầu ({years_exp} năm, yêu cầu {min_years} năm)")
        else:
            score += 0  # Không đạt yêu cầu
            feedback.append(f"✗ Kinh nghiệm chưa đạt yêu cầu ({years_exp} năm < {min_years} năm)")
        
        # 2. Kiểm tra học vấn (25% trọng số)
        min_education = position_req['min_education']
        candidate_edu_score = self.get_education_score(education)
        min_edu_score = self.get_education_score(min_education)
        
        if candidate_edu_score >= min_edu_score + 1:
            score += 2.5  # Vượt yêu cầu
            feedback.append(f"✓ Học vấn vượt yêu cầu ({education} >= {min_education})")
        elif candidate_edu_score >= min_edu_score:
            score += 2  # Đạt yêu cầu
            feedback.append(f"✓ Học vấn đạt yêu cầu ({education} >= {min_education})")
        else:
            score += 1  # Chưa đạt yêu cầu
            feedback.append(f"✗ Học vấn chưa đạt yêu cầu ({education} < {min_education})")
        
        # 3. Kiểm tra kỹ năng (35% trọng số)
        valuable_skills = ['python', 'java', 'machine learning', 'leadership', 'project management', 
                          'sql', 'data analysis', 'communication', 'teamwork']
        
        skill_list = [s.strip() for s in skills.split(',')]
        skill_count = len(skill_list)
        valuable_skill_count = sum(1 for skill in valuable_skills if skill in skills)
        
        if valuable_skill_count >= 4 and skill_count >= 6:
            score += 3.5  # Kỹ năng xuất sắc
            feedback.append(f"✓ Kỹ năng xuất sắc ({valuable_skill_count} kỹ năng giá trị, {skill_count} tổng)")
        elif valuable_skill_count >= 2 and skill_count >= 4:
            score += 2.5  # Kỹ năng tốt
            feedback.append(f"✓ Kỹ năng tốt ({valuable_skill_count} kỹ năng giá trị, {skill_count} tổng)")
        elif valuable_skill_count >= 1:
            score += 1.5  # Kỹ năng cơ bản
            feedback.append(f"⚠ Kỹ năng cơ bản ({valuable_skill_count} kỹ năng giá trị, {skill_count} tổng)")
        else:
            score += 0.5  # Kỹ năng hạn chế
            feedback.append(f"✗ Kỹ năng hạn chế ({valuable_skill_count} kỹ năng giá trị, {skill_count} tổng)")
        
        # Tính toán kết quả cuối cùng
        percentage = (score / max_score) * 100
        
        # Quyết định cuối cùng dựa trên điều kiện nghiêm ngặt
        suitable = False
        confidence_level = "Thấp"
        
        if score >= 8:  # 80%
            suitable = True
            confidence_level = "Cao"
            recommendation = "Rất khuyến khích mời phỏng vấn"
        elif score >= 6.5:  # 65%
            suitable = True  
            confidence_level = "Trung bình"
            recommendation = "Khuyến khích mời phỏng vấn"
        elif score >= 5:  # 50%
            suitable = False
            confidence_level = "Thấp"
            recommendation = "Cần đánh giá thêm hoặc xem xét vị trí thấp hơn"
        else:
            suitable = False
            confidence_level = "Rất thấp"
            recommendation = "Không phù hợp với vị trí này"
        
        return {
            'suitable': suitable,
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'confidence_level': confidence_level,
            'recommendation': recommendation,
            'feedback': feedback,
            'position_requirements': {
                'min_years': min_years,
                'min_education': min_education,
                'position': position
            }
        }
    
    def preprocess_text(self, text):
        """
        Tiền xử lý văn bản (CV, mô tả kỹ năng)
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
    
    def create_sample_data(self, num_samples=1000):
        """
        Tạo dữ liệu mẫu cho training với logic cải thiện
        """
        logger.info(f"🎲 Tạo dữ liệu mẫu với {num_samples} mẫu (logic cải thiện)...")
        
        np.random.seed(42)
        
        # Danh sách kỹ năng phổ biến
        skills_pool = [
            'python', 'java', 'javascript', 'sql', 'machine learning', 'data analysis',
            'project management', 'communication', 'teamwork', 'leadership',
            'react', 'nodejs', 'docker', 'kubernetes', 'aws', 'azure',
            'agile', 'scrum', 'git', 'linux', 'statistics', 'excel'
        ]
        
        education_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']
        positions = ['developer', 'analyst', 'manager', 'designer', 'consultant', 
                    'data_scientist', 'senior_developer', 'junior_developer', 'intern', 'fresher']
        
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
            
            # Sử dụng logic cải thiện để tạo nhãn
            candidate_data = {
                'years_experience': years_exp,
                'education_level': education,
                'skills': skills_str,
                'position_applied': position
            }
            
            assessment = self.improved_suitability_logic(candidate_data)
            suitable = 1 if assessment['suitable'] else 0
            
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
        sample_file = self.data_path / 'sample_candidates_improved.csv'
        df.to_csv(sample_file, index=False)
        logger.info(f"✓ Dữ liệu mẫu cải thiện đã lưu tại {sample_file}")
        
        return df

    def extract_features(self, df):
        """
        Trích xuất đặc trưng từ dữ liệu ứng viên
        """
        logger.info("📊 Đang trích xuất đặc trưng từ dữ liệu ứng viên...")
        
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
        processed_df['education_score'] = processed_df['education_level'].apply(self.get_education_score)
        
        # Chuẩn hóa kinh nghiệm (năm)
        processed_df['years_experience'] = pd.to_numeric(processed_df['years_experience'], errors='coerce').fillna(0)
        
        # Tính điểm kỹ năng (số lượng kỹ năng)
        processed_df['num_skills'] = processed_df['skills'].str.count(',') + 1
        processed_df['num_skills'] = processed_df['num_skills'].fillna(0)
        
        logger.info("✓ Trích xuất đặc trưng hoàn tất")
        return processed_df
    
    def train_model(self, df=None):
        """
        Training mô hình phân loại với logic cải thiện
        """
        logger.info("🧠 Bắt đầu huấn luyện mô hình với logic cải thiện...")
        
        if df is None:
            logger.info("📦 Tạo dữ liệu mẫu với logic cải thiện...")
            df = self.create_sample_data()
        
        # Extract features
        logger.info("🔧 Trích xuất đặc trưng từ dữ liệu...")
        processed_df = self.extract_features(df)
        
        # Prepare text features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        logger.info("📝 Xử lý đặc trưng văn bản...")
        text_features = self.vectorizer.fit_transform(processed_df['combined_text'])
        
        # Prepare numerical features
        numerical_cols = ['education_score', 'years_experience', 'num_skills']
        self.feature_columns = numerical_cols
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        logger.info("📏 Chuẩn hóa đặc trưng số...")
        numerical_features = self.scaler.fit_transform(processed_df[numerical_cols])
        
        # Combine features
        X_text = text_features.toarray()
        X_numerical = numerical_features
        X = np.hstack([X_text, X_numerical])
        
        # Target variable
        y = processed_df['suitable']
        
        logger.info(f"📊 Tổng số mẫu: {len(X)}")
        logger.info(f"🎯 Số đặc trưng: {X.shape[1]}")
        logger.info(f"⚖️ Phân phối nhãn - Phù hợp: {y.sum()}, Chưa phù hợp: {len(y) - y.sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"🔨 Dữ liệu huấn luyện: {len(X_train)} mẫu")
        logger.info(f"🔍 Dữ liệu kiểm thử: {len(X_test)} mẫu")
        
        # Train model
        logger.info("🌲 Bắt đầu huấn luyện mô hình Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("📈 Đánh giá hiệu suất mô hình...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"🎉 Huấn luyện mô hình hoàn tất!")
        logger.info(f"🏆 Độ chính xác: {accuracy:.3f}")
        logger.info(f"📋 Báo cáo phân loại:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        logger.info("💾 Lưu mô hình...")
        self.save_model()
        
        return accuracy
    
    def predict_candidate(self, candidate_data):
        """
        Dự đoán độ phù hợp của ứng viên sử dụng cả ML model và logic cải thiện
        """
        if self.model is None:
            # Nếu chưa có model, chỉ sử dụng logic cải thiện
            logger.warning("⚠️ Chưa có mô hình ML, sử dụng logic đánh giá cải thiện")
            assessment = self.improved_suitability_logic(candidate_data)
            
            result = {
                'candidate_id': candidate_data.get('candidate_id', 'Không xác định'),
                'prediction': 'Suitable' if assessment['suitable'] else 'Not Suitable',
                'prediction_vietnamese': 'Phù hợp' if assessment['suitable'] else 'Chưa phù hợp',
                'confidence': assessment['percentage'] / 100,
                'probability_suitable': assessment['percentage'] / 100,
                'recommendation': assessment['recommendation'],
                'recommendation_vietnamese': assessment['recommendation'],
                'education_display': self.get_education_vietnamese(candidate_data.get('education_level', '')),
                'summary': self.generate_candidate_summary_vietnamese(candidate_data, assessment['suitable'], assessment['percentage'] / 100),
                'detailed_feedback': assessment['feedback'],
                'position_requirements': assessment['position_requirements'],
                'assessment_method': 'Improved Logic Only'
            }
            
            logger.info(f"🎯 Đánh giá logic cho {result['candidate_id']}: {result['prediction_vietnamese']} ({result['confidence']:.3f})")
            return result
        
        # Sử dụng cả ML model và logic cải thiện
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
        
        # ML Prediction
        ml_prediction = self.model.predict(X)[0]
        ml_probability = self.model.predict_proba(X)[0]
        
        # Logic Assessment
        logic_assessment = self.improved_suitability_logic(candidate_data)
        
        # Combine both approaches (weighted average)
        ml_confidence = max(ml_probability)
        logic_confidence = logic_assessment['percentage'] / 100
        
        # Trọng số: 60% ML, 40% Logic
        final_confidence = 0.6 * ml_confidence + 0.4 * logic_confidence
        
        # Quyết định cuối cùng: phải đạt cả ML và Logic hoặc có confidence cao
        final_suitable = (ml_prediction == 1 and logic_assessment['suitable']) or final_confidence > 0.8
        
        result = {
            'candidate_id': candidate_data.get('candidate_id', 'Không xác định'),
            'prediction': 'Suitable' if final_suitable else 'Not Suitable',
            'prediction_vietnamese': 'Phù hợp' if final_suitable else 'Chưa phù hợp',
            'confidence': final_confidence,
            'probability_suitable': final_confidence,
            'recommendation': self.get_recommendation_vietnamese(1 if final_suitable else 0, final_confidence),
            'recommendation_vietnamese': self.get_recommendation_vietnamese(1 if final_suitable else 0, final_confidence),
            'education_display': self.get_education_vietnamese(candidate_data.get('education_level', '')),
            'summary': self.generate_candidate_summary_vietnamese(candidate_data, final_suitable, final_confidence),
            'detailed_feedback': logic_assessment['feedback'],
            'position_requirements': logic_assessment['position_requirements'],
            'ml_prediction': 'Suitable' if ml_prediction == 1 else 'Not Suitable',
            'ml_confidence': ml_confidence,
            'logic_prediction': 'Suitable' if logic_assessment['suitable'] else 'Not Suitable',
            'logic_confidence': logic_confidence,
            'assessment_method': 'Combined ML + Improved Logic'
        }
        
        logger.info(f"🎯 Dự đoán kết hợp cho {result['candidate_id']}: {result['prediction_vietnamese']} (tin cậy cuối: {result['confidence']:.3f})")
        
        return result
    
    def get_recommendation_vietnamese(self, prediction, confidence):
        """
        Đưa ra khuyến nghị bằng tiếng Việt dựa trên dự đoán
        """
        if prediction == 1:
            if confidence > 0.8:
                return "Rất khuyến khích mời phỏng vấn"
            elif confidence > 0.6:
                return "Khuyến khích mời phỏng vấn"
            else:
                return "Cân nhắc mời phỏng vấn với thái độ thận trọng"
        else:
            if confidence > 0.8:
                return "Không phù hợp với vị trí này"
            elif confidence > 0.6:
                return "Cần đánh giá thêm hoặc xem xét vị trí thấp hơn"
            else:
                return "Không khuyến khích"

    def get_education_vietnamese(self, education_level):
        """Chuyển đổi trình độ học vấn sang tiếng Việt"""
        education_mapping = {
            'high_school': 'Tốt nghiệp THPT',
            'associate': 'Cao đẳng', 
            'bachelor': 'Cử nhân',
            'master': 'Thạc sĩ',
            'phd': 'Tiến sĩ'
        }
        return education_mapping.get(education_level, 'Không xác định')

    def generate_candidate_summary_vietnamese(self, candidate_data, prediction, confidence):
        """Tạo tóm tắt ứng viên bằng tiếng Việt"""
        years_exp = candidate_data.get('years_experience', 0)
        education = self.get_education_vietnamese(candidate_data.get('education_level', ''))
        skills_count = len(candidate_data.get('skills', '').split(','))
        position = candidate_data.get('position_applied', 'chưa xác định')
        
        summary = f"Ứng viên ứng tuyển vị trí {position}, có {years_exp} năm kinh nghiệm, trình độ {education}, "
        summary += f"sở hữu {skills_count} kỹ năng chính. "
        
        if prediction:
            if confidence > 0.8:
                summary += "Đây là ứng viên tiềm năng cao, rất phù hợp với vị trí ứng tuyển."
            else:
                summary += "Ứng viên có tiềm năng tốt, phù hợp với vị trí ứng tuyển."
        else:
            if confidence > 0.6:
                summary += "Ứng viên cần được đánh giá kỹ hơn hoặc xem xét vị trí phù hợp hơn."
            else:
                summary += "Ứng viên chưa đáp ứng đủ yêu cầu cho vị trí này."
        
        return summary
    
    def batch_predict(self, csv_file):
        """
        Dự đoán hàng loạt từ file CSV
        """
        logger.info(f"📁 Xử lý dự đoán hàng loạt từ file {csv_file}")
        
        df = pd.read_csv(csv_file)
        results = []
        
        total_candidates = len(df)
        logger.info(f"👥 Tổng số ứng viên cần xử lý: {total_candidates}")
        
        for idx, row in df.iterrows():
            candidate_data = row.to_dict()
            try:
                result = self.predict_candidate(candidate_data)
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"⏳ Đã xử lý {idx + 1}/{total_candidates} ứng viên")
                    
            except Exception as e:
                logger.error(f"❌ Lỗi dự đoán ứng viên {candidate_data.get('candidate_id', idx)}: {e}")
                results.append({
                    'candidate_id': candidate_data.get('candidate_id', f'CAND_{idx}'),
                    'prediction': 'Error',
                    'prediction_vietnamese': 'Lỗi',
                    'confidence': 0.0,
                    'probability_suitable': 0.0,
                    'recommendation': 'Processing error',
                    'recommendation_vietnamese': 'Lỗi xử lý'
                })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.data_path / f'ket_qua_du_doan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Kết quả dự đoán hàng loạt đã lưu tại {output_file}")
        
        return results_df
    
    def save_model(self):
        """
        Lưu mô hình và các thành phần
        """
        try:
            # Save model components
            with open(self.model_path / 'rf_model_improved.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.model_path / 'vectorizer_improved.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(self.model_path / 'scaler_improved.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(self.model_path / 'feature_columns_improved.json', 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Save position requirements
            with open(self.model_path / 'position_requirements.json', 'w', encoding='utf-8') as f:
                json.dump(self.position_requirements, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Mô hình cải thiện đã lưu thành công")
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu mô hình: {e}")
    
    def load_model(self):
        """
        Load mô hình đã lưu với thông báo tiếng Việt
        """
        try:
            # Try to load improved model first
            if (self.model_path / 'rf_model_improved.pkl').exists():
                logger.info("📂 Đang tải mô hình cải thiện từ file...")
                
                with open(self.model_path / 'rf_model_improved.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.model_path / 'vectorizer_improved.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.model_path / 'scaler_improved.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open(self.model_path / 'feature_columns_improved.json', 'r') as f:
                    self.feature_columns = json.load(f)
                
                logger.info("✅ Tải mô hình cải thiện thành công!")
                return True
            
            # Fallback to old model
            elif (self.model_path / 'rf_model.pkl').exists():
                logger.info("📂 Đang tải mô hình cũ từ file...")
                
                with open(self.model_path / 'rf_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.model_path / 'vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.model_path / 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open(self.model_path / 'feature_columns.json', 'r') as f:
                    self.feature_columns = json.load(f)
                
                logger.info("✅ Tải mô hình cũ thành công!")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Không thể tải mô hình: {e}")
            return False
    
    def generate_report(self, results_df=None):
        """
        Tạo báo cáo tổng hợp bằng tiếng Việt
        """
        if results_df is None:
            logger.warning("⚠️ Không có dữ liệu kết quả để tạo báo cáo")
            return {}
        
        total_candidates = len(results_df)
        suitable_candidates = len(results_df[results_df['prediction'] == 'Suitable'])
        avg_confidence = results_df['confidence'].mean()
        
        report = {
            'total_candidates': total_candidates,
            'suitable_candidates': suitable_candidates,
            'unsuitable_candidates': total_candidates - suitable_candidates,
            'suitable_percentage': (suitable_candidates / total_candidates * 100) if total_candidates > 0 else 0,
            'average_confidence': avg_confidence,
            'high_confidence_suitable': len(results_df[
                (results_df['prediction'] == 'Suitable') & 
                (results_df['confidence'] > 0.8)
            ]),
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'summary': f"Đã xử lý {total_candidates} ứng viên với logic cải thiện, trong đó {suitable_candidates} ứng viên phù hợp ({(suitable_candidates / total_candidates * 100):.1f}%)" if total_candidates > 0 else "Không có dữ liệu để xử lý",
            'system_version': 'HR DSS v2.0 - Improved Logic'
        }
        
        # Save report
        report_file = self.data_path / f'bao_cao_cai_tien_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Báo cáo cải thiện đã được tạo và lưu tại {report_file}")
        return report


def main():
    """
    Hàm main để chạy ứng dụng
    """
    parser = argparse.ArgumentParser(description='Hệ thống Hỗ trợ Ra quyết định Tuyển dụng - Phiên bản Cải thiện')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'demo'], 
                       default='demo', help='Chế độ chạy hệ thống')
    parser.add_argument('--input', type=str, help='File đầu vào cho dự đoán hàng loạt')
    parser.add_argument('--model-path', type=str, default='models/', 
                       help='Đường dẫn thư mục mô hình')
    parser.add_argument('--data-path', type=str, default='data/', 
                       help='Đường dẫn thư mục dữ liệu')
    
    args = parser.parse_args()
    
    # Initialize system
    hr_system = HRDecisionSupportSystem(args.model_path, args.data_path)
    
    if args.mode == 'train':
        logger.info("🧠 Chế độ huấn luyện được chọn (với logic cải thiện)")
        accuracy = hr_system.train_model()
        print(f"🎉 Huấn luyện mô hình hoàn tất với độ chính xác: {accuracy:.3f}")
    
    elif args.mode == 'predict':
        logger.info("🎯 Chế độ dự đoán đơn (với logic cải thiện)")
        # Test case: Data Scientist với 0 năm kinh nghiệm
        candidate = {
            'candidate_id': 'GiaHuy_Test',
            'years_experience': 0,
            'education_level': 'bachelor',
            'skills': 'python, machine learning',
            'experience_description': 'Mới tốt nghiệp với kiến thức cơ bản về lập trình',
            'position_applied': 'data_scientist'
        }
        
        result = hr_system.predict_candidate(candidate)
        print("📋 Kết quả dự đoán cải thiện:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Lỗi: Cần file --input cho chế độ dự đoán hàng loạt")
            return
        
        logger.info(f"👥 Chế độ dự đoán hàng loạt cho file: {args.input} (với logic cải thiện)")
        results = hr_system.batch_predict(args.input)
        report = hr_system.generate_report(results)
        print("✅ Dự đoán hàng loạt hoàn tất với logic cải thiện")
        print("📊 Báo cáo tổng hợp:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    else:  # demo mode
        logger.info("🎮 Chế độ demo - Kiểm thử logic cải thiện")
        
        # Test multiple scenarios
        test_cases = [
            {
                'name': 'Data Scientist với 0 năm kinh nghiệm (như trường hợp GiaHuy)',
                'data': {
                    'candidate_id': 'GiaHuy',
                    'years_experience': 0,
                    'education_level': 'bachelor',
                    'skills': 'python, machine learning',
                    'position_applied': 'data_scientist'
                }
            },
            {
                'name': 'Junior Developer tốt nghiệp mới (phù hợp)',
                'data': {
                    'candidate_id': 'JUNIOR001',
                    'years_experience': 0,
                    'education_level': 'bachelor',
                    'skills': 'python, sql, teamwork',
                    'position_applied': 'junior_developer'
                }
            },
            {
                'name': 'Senior Developer kinh nghiệm cao',
                'data': {
                    'candidate_id': 'SENIOR001',
                    'years_experience': 6,
                    'education_level': 'master',
                    'skills': 'python, java, leadership, project management, sql',
                    'position_applied': 'senior_developer'
                }
            }
        ]
        
        print("=== KIỂM THỬ LOGIC ĐÁNH GIÁ CẢI THIỆN ===\n")
        
        for test_case in test_cases:
            print(f"--- {test_case['name']} ---")
            result = hr_system.predict_candidate(test_case['data'])
            
            print(f"Kết quả: {'✓ PHÙ HỢP' if result['prediction'] == 'Suitable' else '✗ CHƯA PHÙ HỢP'}")
            print(f"Độ tin cậy: {result['confidence']:.1%}")
            print(f"Khuyến nghị: {result['recommendation_vietnamese']}")
            
            if 'detailed_feedback' in result:
                print("Chi tiết đánh giá:")
                for feedback in result['detailed_feedback']:
                    print(f"  {feedback}")
            
            if 'position_requirements' in result:
                req = result['position_requirements']
                print(f"Yêu cầu vị trí: {req['min_years']} năm kinh nghiệm, {req['min_education']}")
            
            print()

if __name__ == "__main__":
    main()