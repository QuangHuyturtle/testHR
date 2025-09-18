#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Decision Support System - Main Application (Vietnamese Version)
Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự - Phiên bản tiếng Việt

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
        
        logger.info("🚀 Hệ thống Hỗ trợ Ra quyết định Tuyển dụng đã được khởi tạo")
    
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
        
        logger.info("✓ Trích xuất đặc trưng hoàn tất")
        return processed_df
    
    def create_sample_data(self, num_samples=1000):
        """
        Tạo dữ liệu mẫu cho training
        
        Args:
            num_samples (int): Số lượng mẫu
            
        Returns:
            pd.DataFrame: Dữ liệu mẫu
        """
        logger.info(f"🎲 Tạo dữ liệu mẫu với {num_samples} mẫu...")
        
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
        logger.info(f"✓ Dữ liệu mẫu đã lưu tại {sample_file}")
        
        return df
    
    def train_model(self, df=None):
        """
        Training mô hình phân loại với thông báo tiếng Việt
        """
        logger.info("🧠 Bắt đầu huấn luyện mô hình...")
        
        if df is None:
            logger.info("📦 Tạo dữ liệu mẫu...")
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
        Dự đoán độ phù hợp của ứng viên với thông báo tiếng Việt
        
        Args:
            candidate_data (dict): Thông tin ứng viên
            
        Returns:
            dict: Kết quả dự đoán bằng tiếng Việt
        """
        if self.model is None:
            raise ValueError("❌ Mô hình chưa được huấn luyện. Hãy chạy train_model() trước.")
        
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
            'candidate_id': candidate_data.get('candidate_id', 'Không xác định'),
            'prediction': 'Suitable' if prediction == 1 else 'Not Suitable',
            'prediction_vietnamese': 'Phù hợp' if prediction == 1 else 'Chưa phù hợp',
            'confidence': max(probability),
            'probability_suitable': probability[1] if len(probability) > 1 else probability[0],
            'recommendation': self.get_recommendation(prediction, max(probability)),
            'recommendation_vietnamese': self.get_recommendation_vietnamese(prediction, max(probability)),
            'education_display': self.get_education_vietnamese(candidate_data.get('education_level', '')),
            'summary': self.generate_candidate_summary_vietnamese(candidate_data, prediction, max(probability))
        }
        
        logger.info(f"🎯 Dự đoán cho {result['candidate_id']}: {result['prediction_vietnamese']} (độ tin cậy: {result['confidence']:.3f})")
        
        return result
    
    def get_recommendation(self, prediction, confidence):
        """
        Đưa ra khuyến nghị dựa trên dự đoán (English)
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
    
    def get_recommendation_vietnamese(self, prediction, confidence):
        """
        Đưa ra khuyến nghị bằng tiếng Việt dựa trên dự đoán
        
        Args:
            prediction (int): Kết quả dự đoán (0 hoặc 1)
            confidence (float): Độ tin cậy
            
        Returns:
            str: Khuyến nghị bằng tiếng Việt
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
                return "Không khuyến khích"
            else:
                return "Cần đánh giá thêm"

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
        
        summary = f"Ứng viên có {years_exp} năm kinh nghiệm, trình độ {education}, "
        summary += f"sở hữu {skills_count} kỹ năng chính. "
        
        if prediction == 1:
            if confidence > 0.8:
                summary += "Đây là ứng viên tiềm năng cao, rất phù hợp với vị trí ứng tuyển."
            else:
                summary += "Ứng viên có tiềm năng tốt, phù hợp với vị trí ứng tuyển."
        else:
            if confidence > 0.8:
                summary += "Ứng viên chưa đáp ứng đủ yêu cầu cho vị trí này."
            else:
                summary += "Ứng viên cần được đánh giá kỹ hơn để đưa ra quyết định cuối cùng."
        
        return summary
    
    def batch_predict(self, csv_file):
        """
        Dự đoán hàng loạt từ file CSV
        
        Args:
            csv_file (str): Đường dẫn file CSV
            
        Returns:
            pd.DataFrame: Kết quả dự đoán
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
            with open(self.model_path / 'rf_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.model_path / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(self.model_path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(self.model_path / 'feature_columns.json', 'w') as f:
                json.dump(self.feature_columns, f)
            
            logger.info("✅ Mô hình đã lưu thành công")
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu mô hình: {e}")
    
    def load_model(self):
        """
        Load mô hình đã lưu với thông báo tiếng Việt
        """
        try:
            if (self.model_path / 'rf_model.pkl').exists():
                logger.info("📂 Đang tải mô hình từ file...")
                
                with open(self.model_path / 'rf_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.model_path / 'vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.model_path / 'scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open(self.model_path / 'feature_columns.json', 'r') as f:
                    self.feature_columns = json.load(f)
                
                logger.info("✅ Tải mô hình thành công!")
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
            'summary': f"Đã xử lý {total_candidates} ứng viên, trong đó {suitable_candidates} ứng viên phù hợp ({(suitable_candidates / total_candidates * 100):.1f}%)" if total_candidates > 0 else "Không có dữ liệu để xử lý"
        }
        
        # Save report
        report_file = self.data_path / f'bao_cao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Báo cáo đã được tạo và lưu tại {report_file}")
        return report


def main():
    """
    Hàm main để chạy ứng dụng
    """
    parser = argparse.ArgumentParser(description='Hệ thống Hỗ trợ Ra quyết định Tuyển dụng')
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
        logger.info("🧠 Chế độ huấn luyện được chọn")
        accuracy = hr_system.train_model()
        print(f"🎉 Huấn luyện mô hình hoàn tất với độ chính xác: {accuracy:.3f}")
    
    elif args.mode == 'predict':
        logger.info("🎯 Chế độ dự đoán đơn")
        # Example candidate
        candidate = {
            'candidate_id': 'DEMO_001',
            'years_experience': 5,
            'education_level': 'bachelor',
            'skills': 'python, machine learning, sql, data analysis, teamwork',
            'experience_description': 'Chuyên gia phân tích dữ liệu có kinh nghiệm với kỹ năng lập trình mạnh',
            'position_applied': 'analyst'
        }
        
        result = hr_system.predict_candidate(candidate)
        print("📋 Kết quả dự đoán:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Lỗi: Cần file --input cho chế độ dự đoán hàng loạt")
            return
        
        logger.info(f"👥 Chế độ dự đoán hàng loạt cho file: {args.input}")
        results = hr_system.batch_predict(args.input)
        report = hr_system.generate_report(results)
        print("✅ Dự đoán hàng loạt hoàn tất")
        print("📊 Báo cáo tổng hợp:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    else:  # demo mode
        logger.info("🎮 Chế độ demo - Huấn luyện và kiểm thử")
        
        # Train model
        accuracy = hr_system.train_model()
        print(f"🎉 Mô hình đã được huấn luyện với độ chính xác: {accuracy:.3f}")
        
        