#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Decision Support System - Enhanced Version with Required Skills Logic
Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự - Phiên bản nâng cao với logic kỹ năng bắt buộc

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
        logging.FileHandler('logs/hr_dss_enhanced.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HRDecisionSupportSystemEnhanced:
    """
    Hệ thống hỗ trợ ra quyết định tuyển dụng nhân sự - Phiên bản nâng cao với logic kỹ năng bắt buộc
    """
    
    def __init__(self, model_path="models/", data_path="data/"):
        """
        Khởi tạo hệ thống HR DSS Enhanced
        
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
        
        # Load position requirements và required skills
        self.position_requirements = self.get_position_requirements()
        self.required_skills_by_position = self.get_required_skills_by_position()
        
        # Load model if exists
        self.load_model()
        
        logger.info("🚀 Hệ thống Hỗ trợ Ra quyết định Tuyển dụng (Enhanced) đã được khởi tạo")

    def get_required_skills_by_position(self):
        """
        Định nghĩa kỹ năng bắt buộc cho từng vị trí/ngành nghề
        """
        return {
            # Data Science & Analytics
            'data_scientist': {
                'required': ['sql', 'python'],  # Bắt buộc có ít nhất 1 trong 2
                'preferred': ['machine learning', 'statistics', 'pandas', 'numpy', 'scipy'],
                'bonus': ['deep learning', 'tensorflow', 'pytorch', 'spark', 'hadoop']
            },
            'data_analyst': {
                'required': ['sql', 'excel'],  # Bắt buộc có ít nhất 1 trong 2
                'preferred': ['python', 'r', 'tableau', 'powerbi', 'statistics'],
                'bonus': ['machine learning', 'data visualization', 'pandas']
            },
            'business_analyst': {
                'required': ['excel', 'sql'],
                'preferred': ['powerbi', 'tableau', 'business intelligence', 'requirements analysis'],
                'bonus': ['python', 'r', 'process modeling', 'agile']
            },
            
            # Web Development
            'web_developer': {
                'required': ['html', 'css', 'javascript'],  # Bắt buộc có tất cả 3
                'preferred': ['react', 'vue', 'angular', 'nodejs', 'php'],
                'bonus': ['typescript', 'webpack', 'sass', 'mongodb', 'postgresql']
            },
            'frontend_developer': {
                'required': ['html', 'css', 'javascript'],
                'preferred': ['react', 'vue', 'angular', 'responsive design'],
                'bonus': ['typescript', 'sass', 'webpack', 'figma', 'ui/ux']
            },
            'backend_developer': {
                'required': ['sql', 'api'],  # Ít nhất 1 ngôn ngữ backend
                'preferred': ['nodejs', 'python', 'java', 'php', 'c#', 'go'],
                'bonus': ['microservices', 'docker', 'kubernetes', 'aws', 'azure']
            },
            'fullstack_developer': {
                'required': ['html', 'css', 'javascript', 'sql'],
                'preferred': ['react', 'nodejs', 'python', 'mongodb', 'postgresql'],
                'bonus': ['docker', 'aws', 'microservices', 'devops']
            },
            
            # Software Development
            'software_developer': {
                'required': ['programming'],  # Ít nhất 1 ngôn ngữ lập trình
                'preferred': ['python', 'java', 'javascript', 'c#', 'c++'],
                'bonus': ['git', 'agile', 'testing', 'ci/cd', 'docker']
            },
            'mobile_developer': {
                'required': ['mobile'],
                'preferred': ['react native', 'flutter', 'swift', 'kotlin', 'java'],
                'bonus': ['firebase', 'push notifications', 'app store', 'google play']
            },
            
            # DevOps & Infrastructure
            'devops_engineer': {
                'required': ['linux', 'docker'],
                'preferred': ['kubernetes', 'aws', 'azure', 'terraform', 'ansible'],
                'bonus': ['monitoring', 'jenkins', 'gitlab ci', 'prometheus', 'grafana']
            },
            'system_administrator': {
                'required': ['linux', 'windows'],
                'preferred': ['networking', 'security', 'bash', 'powershell'],
                'bonus': ['vmware', 'hyper-v', 'active directory', 'monitoring']
            },
            
            # Design
            'ui_ux_designer': {
                'required': ['design', 'ui/ux'],
                'preferred': ['figma', 'sketch', 'adobe creative suite', 'prototyping'],
                'bonus': ['user research', 'wireframing', 'html', 'css']
            },
            'graphic_designer': {
                'required': ['design', 'adobe creative suite'],
                'preferred': ['photoshop', 'illustrator', 'indesign', 'branding'],
                'bonus': ['web design', 'print design', 'typography', 'color theory']
            },
            
            # Marketing & Sales
            'digital_marketer': {
                'required': ['digital marketing'],
                'preferred': ['google ads', 'facebook ads', 'seo', 'sem', 'analytics'],
                'bonus': ['content marketing', 'email marketing', 'social media', 'conversion optimization']
            },
            'content_creator': {
                'required': ['content writing', 'communication'],
                'preferred': ['seo', 'social media', 'copywriting', 'content strategy'],
                'bonus': ['video editing', 'graphic design', 'analytics', 'wordpress']
            },
            
            # Management
            'project_manager': {
                'required': ['project management'],
                'preferred': ['agile', 'scrum', 'pmp', 'leadership', 'communication'],
                'bonus': ['jira', 'confluence', 'risk management', 'stakeholder management']
            },
            'product_manager': {
                'required': ['product management', 'communication'],
                'preferred': ['agile', 'user research', 'analytics', 'roadmap planning'],
                'bonus': ['technical skills', 'sql', 'wireframing', 'a/b testing']
            },
            
            # Quality Assurance
            'qa_engineer': {
                'required': ['testing', 'quality assurance'],
                'preferred': ['automation testing', 'selenium', 'manual testing', 'bug tracking'],
                'bonus': ['api testing', 'performance testing', 'security testing', 'ci/cd']
            },
            
            # Generic positions
            'analyst': {
                'required': ['excel', 'data analysis'],
                'preferred': ['sql', 'statistics', 'reporting'],
                'bonus': ['python', 'r', 'tableau', 'powerbi']
            },
            'developer': {
                'required': ['programming'],
                'preferred': ['python', 'java', 'javascript', 'sql'],
                'bonus': ['git', 'agile', 'testing']
            },
            'designer': {
                'required': ['design'],
                'preferred': ['adobe creative suite', 'figma', 'ui/ux'],
                'bonus': ['html', 'css', 'prototyping']
            }
        }

    def get_position_requirements(self):
        """
        Định nghĩa yêu cầu kinh nghiệm tối thiểu cho từng loại vị trí
        """
        return {
            # Senior positions - yêu cầu kinh nghiệm cao
            'senior_data_scientist': {'min_years': 5, 'min_education': 'bachelor'},
            'senior_developer': {'min_years': 5, 'min_education': 'bachelor'},
            'senior_analyst': {'min_years': 5, 'min_education': 'bachelor'}, 
            'lead': {'min_years': 4, 'min_education': 'bachelor'},
            'manager': {'min_years': 3, 'min_education': 'bachelor'},
            'director': {'min_years': 7, 'min_education': 'master'},
            'principal_engineer': {'min_years': 8, 'min_education': 'bachelor'},
            
            # Mid-level positions
            'data_scientist': {'min_years': 2, 'min_education': 'bachelor'},
            'web_developer': {'min_years': 2, 'min_education': 'bachelor'},
            'software_developer': {'min_years': 2, 'min_education': 'bachelor'},
            'fullstack_developer': {'min_years': 3, 'min_education': 'bachelor'},
            'devops_engineer': {'min_years': 3, 'min_education': 'bachelor'},
            'product_manager': {'min_years': 3, 'min_education': 'bachelor'},
            'project_manager': {'min_years': 2, 'min_education': 'bachelor'},
            
            # Entry-level positions
            'junior_developer': {'min_years': 0, 'min_education': 'bachelor'},
            'junior_analyst': {'min_years': 0, 'min_education': 'bachelor'},
            'frontend_developer': {'min_years': 1, 'min_education': 'associate'},
            'backend_developer': {'min_years': 1, 'min_education': 'bachelor'},
            'data_analyst': {'min_years': 1, 'min_education': 'bachelor'},
            'qa_engineer': {'min_years': 1, 'min_education': 'bachelor'},
            'ui_ux_designer': {'min_years': 1, 'min_education': 'associate'},
            'graphic_designer': {'min_years': 1, 'min_education': 'associate'},
            
            # Generic positions
            'intern': {'min_years': 0, 'min_education': 'high_school'},
            'fresher': {'min_years': 0, 'min_education': 'bachelor'},
            'coordinator': {'min_years': 0, 'min_education': 'associate'},
            'designer': {'min_years': 0, 'min_education': 'associate'},
            'analyst': {'min_years': 1, 'min_education': 'bachelor'},
            'developer': {'min_years': 1, 'min_education': 'bachelor'},
        }

    def check_required_skills(self, candidate_skills, position):
        """
        Kiểm tra kỹ năng bắt buộc cho vị trí cụ thể
        
        Args:
            candidate_skills (str): Kỹ năng của ứng viên (chuỗi phân cách bằng dấu phẩy)
            position (str): Vị trí ứng tuyển
            
        Returns:
            dict: Kết quả kiểm tra kỹ năng
        """
        # Chuẩn hóa kỹ năng ứng viên
        candidate_skills_list = [skill.strip().lower() for skill in candidate_skills.split(',')]
        
        # Tìm yêu cầu kỹ năng cho vị trí
        position_lower = position.lower().replace(' ', '_')
        skill_requirements = None
        
        # Tìm kiếm chính xác trước
        if position_lower in self.required_skills_by_position:
            skill_requirements = self.required_skills_by_position[position_lower]
        else:
            # Tìm kiếm mờ - tìm vị trí có chứa từ khóa
            for req_position, req_data in self.required_skills_by_position.items():
                if any(keyword in position_lower for keyword in req_position.split('_')):
                    skill_requirements = req_data
                    break
                elif any(keyword in req_position for keyword in position_lower.split('_')):
                    skill_requirements = req_data
                    break
        
        if not skill_requirements:
            # Không tìm thấy yêu cầu cụ thể, sử dụng yêu cầu chung
            return {
                'has_required_skills': True,
                'missing_required': [],
                'matching_preferred': [],
                'matching_bonus': [],
                'skill_score': 5.0,  # Điểm trung bình
                'feedback': ['⚠ Không tìm thấy yêu cầu kỹ năng cụ thể cho vị trí này'],
                'requirements_found': False
            }
        
        # Kiểm tra kỹ năng bắt buộc
        required_skills = skill_requirements.get('required', [])
        preferred_skills = skill_requirements.get('preferred', [])
        bonus_skills = skill_requirements.get('bonus', [])
        
        # Logic kiểm tra kỹ năng bắt buộc
        missing_required = []
        has_required_skills = True
        
        if required_skills:
            # Kiểm tra từng kỹ năng bắt buộc
            for req_skill in required_skills:
                found = any(req_skill.lower() in candidate_skill for candidate_skill in candidate_skills_list)
                if not found:
                    missing_required.append(req_skill)
            
            # Nếu thiếu kỹ năng bắt buộc
            if missing_required:
                has_required_skills = False
        
        # Đếm kỹ năng preferred và bonus
        matching_preferred = []
        matching_bonus = []
        
        for pref_skill in preferred_skills:
            if any(pref_skill.lower() in candidate_skill for candidate_skill in candidate_skills_list):
                matching_preferred.append(pref_skill)
        
        for bonus_skill in bonus_skills:
            if any(bonus_skill.lower() in candidate_skill for candidate_skill in candidate_skills_list):
                matching_bonus.append(bonus_skill)
        
        # Tính điểm kỹ năng (0-10)
        skill_score = 0
        if has_required_skills:
            skill_score += 6  # 6 điểm cơ bản khi có đủ kỹ năng bắt buộc
            skill_score += min(len(matching_preferred) * 0.5, 2)  # Tối đa 2 điểm từ preferred
            skill_score += min(len(matching_bonus) * 0.25, 2)  # Tối đa 2 điểm từ bonus
        else:
            # Nếu thiếu kỹ năng bắt buộc, chỉ tính điểm từ preferred và bonus
            skill_score += min(len(matching_preferred) * 0.3, 3)
            skill_score += min(len(matching_bonus) * 0.15, 2)
        
        skill_score = min(skill_score, 10)  # Giới hạn tối đa 10 điểm
        
        # Tạo feedback
        feedback = []
        if has_required_skills:
            feedback.append(f"✓ Có đủ kỹ năng bắt buộc cho vị trí {position}")
        else:
            feedback.append(f"✗ Thiếu kỹ năng bắt buộc: {', '.join(missing_required)}")
        
        if matching_preferred:
            feedback.append(f"✓ Có {len(matching_preferred)} kỹ năng ưu tiên: {', '.join(matching_preferred)}")
        
        if matching_bonus:
            feedback.append(f"⭐ Có {len(matching_bonus)} kỹ năng bonus: {', '.join(matching_bonus)}")
        
        return {
            'has_required_skills': has_required_skills,
            'missing_required': missing_required,
            'matching_preferred': matching_preferred,
            'matching_bonus': matching_bonus,
            'skill_score': skill_score,
            'feedback': feedback,
            'requirements_found': True
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

    def enhanced_suitability_logic(self, candidate_data):
        """
        Logic đánh giá nâng cao với kiểm tra kỹ năng bắt buộc theo vị trí
        """
        years_exp = int(candidate_data.get('years_experience', 0))
        education = candidate_data.get('education_level', 'high_school').lower()
        skills = candidate_data.get('skills', '').lower()
        position = candidate_data.get('position_applied', '').lower()
        
        # Tìm yêu cầu vị trí
        position_req = None
        for req_position, req_data in self.position_requirements.items():
            if req_position in position or position in req_position:
                position_req = req_data
                break
        
        if position_req is None:
            # Phân loại dựa trên từ khóa trong tên vị trí
            if any(word in position for word in ['senior', 'lead', 'manager', 'director']):
                position_req = {'min_years': 4, 'min_education': 'bachelor'}
            elif any(word in position for word in ['scientist', 'specialist']):
                position_req = {'min_years': 3, 'min_education': 'bachelor'}
            elif any(word in position for word in ['junior', 'intern', 'fresher']):
                position_req = {'min_years': 0, 'min_education': 'bachelor'}
            else:
                position_req = {'min_years': 1, 'min_education': 'bachelor'}
        
        # Kiểm tra kỹ năng bắt buộc
        skill_check = self.check_required_skills(skills, position)
        
        # Tính điểm đánh giá
        score = 0
        max_score = 10
        feedback = []
        
        # 1. Kiểm tra kỹ năng bắt buộc (50% trọng số - quan trọng nhất)
        if skill_check['has_required_skills']:
            score += 5  # Full điểm kỹ năng bắt buộc
            feedback.extend(skill_check['feedback'])
        else:
            score += 1  # Điểm rất thấp nếu thiếu kỹ năng bắt buộc
            feedback.extend(skill_check['feedback'])
        
        # 2. Kiểm tra kinh nghiệm (30% trọng số)
        min_years = position_req['min_years']
        if years_exp >= min_years + 3:
            score += 3  # Vượt yêu cầu nhiều
            feedback.append(f"✓ Kinh nghiệm vượt yêu cầu nhiều ({years_exp} năm >> {min_years} năm)")
        elif years_exp >= min_years + 1:
            score += 2.5  # Vượt yêu cầu
            feedback.append(f"✓ Kinh nghiệm vượt yêu cầu ({years_exp} năm > {min_years} năm)")
        elif years_exp >= min_years:
            score += 2  # Đạt yêu cầu
            feedback.append(f"✓ Kinh nghiệm đạt yêu cầu ({years_exp} năm >= {min_years} năm)")
        elif years_exp >= min_years - 1 and min_years > 0:
            score += 1.5  # Gần đạt yêu cầu
            feedback.append(f"⚠ Kinh nghiệm gần đạt yêu cầu ({years_exp} năm, yêu cầu {min_years} năm)")
        else:
            score += 0.5  # Thiếu kinh nghiệm
            feedback.append(f"✗ Kinh nghiệm chưa đạt yêu cầu ({years_exp} năm < {min_years} năm)")
        
        # 3. Kiểm tra học vấn (20% trọng số)
        min_education = position_req['min_education']
        candidate_edu_score = self.get_education_score(education)
        min_edu_score = self.get_education_score(min_education)
        
        if candidate_edu_score >= min_edu_score + 1:
            score += 2  # Vượt yêu cầu
            feedback.append(f"✓ Học vấn vượt yêu cầu ({education} > {min_education})")
        elif candidate_edu_score >= min_edu_score:
            score += 1.5  # Đạt yêu cầu
            feedback.append(f"✓ Học vấn đạt yêu cầu ({education} >= {min_education})")
        else:
            score += 0.5  # Chưa đạt yêu cầu
            feedback.append(f"✗ Học vấn chưa đạt yêu cầu ({education} < {min_education})")
        
        # Tính toán kết quả cuối cùng
        percentage = (score / max_score) * 100
        
        # Quyết định cuối cùng - nghiêm ngặt hơn với kỹ năng bắt buộc
        suitable = False
        confidence_level = "Thấp"
        
        if skill_check['has_required_skills']:
            if score >= 8.5:  # 85%
                suitable = True
                confidence_level = "Rất cao"
                recommendation = "Rất khuyến khích mời phỏng vấn ngay"
            elif score >= 7:  # 70%
                suitable = True  
                confidence_level = "Cao"
                recommendation = "Khuyến khích mời phỏng vấn"
            elif score >= 5.5:  # 55%
                suitable = True
                confidence_level = "Trung bình"
                recommendation = "Có thể mời phỏng vấn"
            else:
                suitable = False
                confidence_level = "Thấp"
                recommendation = "Cần cải thiện thêm"
        else:
            # Nếu thiếu kỹ năng bắt buộc, rất khó được chấp nhận
            if score >= 7:  # Cần điểm rất cao từ kinh nghiệm + học vấn
                suitable = True
                confidence_level = "Thấp"
                recommendation = "Cần đánh giá kỹ năng trong phỏng vấn"
            else:
                suitable = False
                confidence_level = "Rất thấp"
                recommendation = "Không phù hợp - thiếu kỹ năng bắt buộc"
        
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
            },
            'skill_analysis': skill_check
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
        Tạo dữ liệu mẫu cho training với logic nâng cao
        """
        logger.info(f"🎲 Tạo dữ liệu mẫu với {num_samples} mẫu (logic kỹ năng nâng cao)...")
        
        np.random.seed(42)
        
        # Danh sách kỹ năng theo ngành
        all_skills = []
        for pos_skills in self.required_skills_by_position.values():
            all_skills.extend(pos_skills.get('required', []))
            all_skills.extend(pos_skills.get('preferred', []))
            all_skills.extend(pos_skills.get('bonus', []))
        
        # Thêm các kỹ năng chung
        additional_skills = [
            'communication', 'teamwork', 'problem solving', 'time management',
            'critical thinking', 'creativity', 'adaptability', 'leadership'
        ]
        
        skills_pool = list(set(all_skills + additional_skills))
        
        education_levels = ['high_school', 'associate', 'bachelor', 'master', 'phd']
        positions = list(self.required_skills_by_position.keys())
        
        data = []
        
        for i in range(num_samples):
            # Random position
            position = np.random.choice(positions)
            
            # Kinh nghiệm phù hợp với vị trí
            pos_req = self.position_requirements.get(position, {'min_years': 1, 'min_education': 'bachelor'})
            min_years = pos_req['min_years']
            
            # Random kinh nghiệm với bias theo yêu cầu vị trí
            if np.random.random() < 0.7:  # 70% có kinh nghiệm phù hợp hoặc cao hơn
                years_exp = np.random.randint(min_years, min_years + 10)
            else:  # 30% có kinh nghiệm thấp hơn yêu cầu
                years_exp = np.random.randint(0, min_years + 1)
            
            # Random education
            education = np.random.choice(education_levels)
            
            # Tạo skills có bias theo yêu cầu vị trí
            skill_req = self.required_skills_by_position.get(position, {})
            required_skills = skill_req.get('required', [])
            preferred_skills = skill_req.get('preferred', [])
            bonus_skills = skill_req.get('bonus', [])
            
            candidate_skills = []
            
            # 60% chance có đủ kỹ năng bắt buộc
            if np.random.random() < 0.6 and required_skills:
                candidate_skills.extend(required_skills)
            
            # Random preferred skills
            if preferred_skills:
                num_preferred = np.random.randint(0, min(len(preferred_skills), 4))
                candidate_skills.extend(np.random.choice(preferred_skills, num_preferred, replace=False))
            
            # Random bonus skills
            if bonus_skills:
                num_bonus = np.random.randint(0, min(len(bonus_skills), 3))
                candidate_skills.extend(np.random.choice(bonus_skills, num_bonus, replace=False))
            
            # Thêm một số kỹ năng random
            num_random = np.random.randint(1, 4)
            random_skills = np.random.choice(additional_skills, num_random, replace=False)
            candidate_skills.extend(random_skills)
            
            # Loại bỏ duplicate và tạo string
            candidate_skills = list(set(candidate_skills))
            skills_str = ', '.join(candidate_skills)
            
            # Experience description
            exp_desc = f"Experienced {position.replace('_', ' ')} with {years_exp} years working with {', '.join(candidate_skills[:3])}"
            
            # Sử dụng logic nâng cao để tạo nhãn
            candidate_data = {
                'years_experience': years_exp,
                'education_level': education,
                'skills': skills_str,
                'position_applied': position
            }
            
            assessment = self.enhanced_suitability_logic(candidate_data)
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
        sample_file = self.data_path / 'sample_candidates_enhanced.csv'
        df.to_csv(sample_file, index=False)
        logger.info(f"✓ Dữ liệu mẫu nâng cao đã lưu tại {sample_file}")
        
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
        
        # Chuẩn hóa kinh nghiệm
        processed_df['years_experience'] = pd.to_numeric(processed_df['years_experience'], errors='coerce').fillna(0)
        
        # Tính điểm kỹ năng
        processed_df['num_skills'] = processed_df['skills'].str.count(',') + 1
        processed_df['num_skills'] = processed_df['num_skills'].fillna(0)
        
        # Tính điểm kỹ năng bắt buộc cho từng ứng viên
        skill_scores = []
        for _, row in processed_df.iterrows():
            skill_check = self.check_required_skills(row['skills'], row['position_applied'])
            skill_scores.append(skill_check['skill_score'])
        
        processed_df['skill_score'] = skill_scores
        
        logger.info("✓ Trích xuất đặc trưng hoàn tất")
        return processed_df
    
    def train_model(self, df=None):
        """
        Training mô hình phân loại với logic nâng cao
        """
        logger.info("🧠 Bắt đầu huấn luyện mô hình với logic kỹ năng nâng cao...")
        
        if df is None:
            logger.info("📦 Tạo dữ liệu mẫu với logic nâng cao...")
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
        numerical_cols = ['education_score', 'years_experience', 'num_skills', 'skill_score']
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
            max_depth=12,
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
        Dự đoán độ phù hợp của ứng viên sử dụng cả ML model và logic nâng cao
        """
        if self.model is None:
            # Nếu chưa có model, chỉ sử dụng logic nâng cao
            logger.warning("⚠️ Chưa có mô hình ML, sử dụng logic đánh giá nâng cao")
            assessment = self.enhanced_suitability_logic(candidate_data)
            
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
                'skill_analysis': assessment['skill_analysis'],
                'assessment_method': 'Enhanced Logic Only'
            }
            
            logger.info(f"🎯 Đánh giá logic cho {result['candidate_id']}: {result['prediction_vietnamese']} ({result['confidence']:.3f})")
            return result
        
        # Sử dụng cả ML model và logic nâng cao
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
        logic_assessment = self.enhanced_suitability_logic(candidate_data)
        
        # Combine both approaches với trọng số mới
        ml_confidence = max(ml_probability)
        logic_confidence = logic_assessment['percentage'] / 100
        
        # Trọng số: 40% ML, 60% Logic (ưu tiên logic hơn)
        final_confidence = 0.4 * ml_confidence + 0.6 * logic_confidence
        
        # Quyết định cuối cùng: Logic có quyền veto nếu thiếu kỹ năng bắt buộc
        if not logic_assessment['skill_analysis']['has_required_skills']:
            final_suitable = False  # Veto nếu thiếu kỹ năng bắt buộc
            final_confidence = min(final_confidence, 0.4)  # Giới hạn confidence
        else:
            final_suitable = (ml_prediction == 1 and logic_assessment['suitable']) or final_confidence > 0.75
        
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
            'skill_analysis': logic_assessment['skill_analysis'],
            'ml_prediction': 'Suitable' if ml_prediction == 1 else 'Not Suitable',
            'ml_confidence': ml_confidence,
            'logic_prediction': 'Suitable' if logic_assessment['suitable'] else 'Not Suitable',
            'logic_confidence': logic_confidence,
            'assessment_method': 'Combined ML + Enhanced Logic'
        }
        
        logger.info(f"🎯 Dự đoán kết hợp cho {result['candidate_id']}: {result['prediction_vietnamese']} (tin cậy cuối: {result['confidence']:.3f})")
        
        return result
    
    def get_recommendation_vietnamese(self, prediction, confidence):
        """
        Đưa ra khuyến nghị bằng tiếng Việt dựa trên dự đoán
        """
        if prediction == 1:
            if confidence > 0.8:
                return "Rất khuyến khích mời phỏng vấn ngay"
            elif confidence > 0.7:
                return "Khuyến khích mời phỏng vấn"
            elif confidence > 0.6:
                return "Có thể mời phỏng vấn"
            else:
                return "Cần đánh giá kỹ hơn trong phỏng vấn"
        else:
            if confidence > 0.7:
                return "Không phù hợp với vị trí này"
            elif confidence > 0.5:
                return "Cần cải thiện kỹ năng hoặc xem xét vị trí khác"
            else:
                return "Thiếu kỹ năng bắt buộc - không khuyến khích"

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
                summary += "Đây là ứng viên xuất sắc, rất phù hợp với vị trí ứng tuyển."
            elif confidence > 0.7:
                summary += "Ứng viên có tiềm năng tốt, phù hợp với vị trí ứng tuyển."
            else:
                summary += "Ứng viên có thể phù hợp nhưng cần đánh giá kỹ hơn."
        else:
            if confidence > 0.6:
                summary += "Ứng viên cần cải thiện kỹ năng hoặc xem xét vị trí phù hợp hơn."
            else:
                summary += "Ứng viên chưa đáp ứng đủ yêu cầu cho vị trí này."
        
        return summary
    
    def batch_predict(self, csv_file):
        """
        Dự đoán hàng loạt từ file CSV với logic nâng cao
        """
        logger.info(f"📁 Xử lý dự đoán hàng loạt từ file {csv_file} (logic nâng cao)")
        
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
        output_file = self.data_path / f'ket_qua_du_doan_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Kết quả dự đoán nâng cao đã lưu tại {output_file}")
        
        return results_df
    
    def save_model(self):
        """
        Lưu mô hình và các thành phần
        """
        try:
            # Save model components
            with open(self.model_path / 'rf_model_enhanced.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.model_path / 'vectorizer_enhanced.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(self.model_path / 'scaler_enhanced.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature columns
            with open(self.model_path / 'feature_columns_enhanced.json', 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Save position requirements
            with open(self.model_path / 'position_requirements_enhanced.json', 'w', encoding='utf-8') as f:
                json.dump(self.position_requirements, f, ensure_ascii=False, indent=2)
                
            # Save required skills mapping
            with open(self.model_path / 'required_skills_enhanced.json', 'w', encoding='utf-8') as f:
                json.dump(self.required_skills_by_position, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ Mô hình nâng cao đã lưu thành công")
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu mô hình: {e}")
    
    def load_model(self):
        """
        Load mô hình đã lưu với thông báo tiếng Việt
        """
        try:
            # Try to load enhanced model first
            if (self.model_path / 'rf_model_enhanced.pkl').exists():
                logger.info("📂 Đang tải mô hình nâng cao từ file...")
                
                with open(self.model_path / 'rf_model_enhanced.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.model_path / 'vectorizer_enhanced.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(self.model_path / 'scaler_enhanced.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                
                with open(self.model_path / 'feature_columns_enhanced.json', 'r') as f:
                    self.feature_columns = json.load(f)
                
                logger.info("✅ Tải mô hình nâng cao thành công!")
                return True
            
            # Fallback to older models
            elif (self.model_path / 'rf_model_improved.pkl').exists():
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
                
        except Exception as e:
            logger.warning(f"⚠️ Không thể tải mô hình: {e}")
            return False
    
    def generate_report(self, results_df=None):
        """
        Tạo báo cáo tổng hợp nâng cao bằng tiếng Việt
        """
        if results_df is None:
            logger.warning("⚠️ Không có dữ liệu kết quả để tạo báo cáo")
            return {}
        
        total_candidates = len(results_df)
        suitable_candidates = len(results_df[results_df['prediction'] == 'Suitable'])
        avg_confidence = results_df['confidence'].mean()
        
        # Thống kê theo confidence level
        high_confidence_suitable = len(results_df[
            (results_df['prediction'] == 'Suitable') & 
            (results_df['confidence'] > 0.8)
        ])
        
        medium_confidence_suitable = len(results_df[
            (results_df['prediction'] == 'Suitable') & 
            (results_df['confidence'] > 0.6) & 
            (results_df['confidence'] <= 0.8)
        ])
        
        low_confidence_suitable = len(results_df[
            (results_df['prediction'] == 'Suitable') & 
            (results_df['confidence'] <= 0.6)
        ])
        
        report = {
            'total_candidates': total_candidates,
            'suitable_candidates': suitable_candidates,
            'unsuitable_candidates': total_candidates - suitable_candidates,
            'suitable_percentage': (suitable_candidates / total_candidates * 100) if total_candidates > 0 else 0,
            'average_confidence': avg_confidence,
            'high_confidence_suitable': high_confidence_suitable,
            'medium_confidence_suitable': medium_confidence_suitable,
            'low_confidence_suitable': low_confidence_suitable,
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'summary': f"Đã xử lý {total_candidates} ứng viên với logic kỹ năng nâng cao, trong đó {suitable_candidates} ứng viên phù hợp ({(suitable_candidates / total_candidates * 100):.1f}%)" if total_candidates > 0 else "Không có dữ liệu để xử lý",
            'system_version': 'HR DSS v3.0 - Enhanced with Required Skills Logic',
            'quality_analysis': {
                'high_confidence_count': high_confidence_suitable,
                'medium_confidence_count': medium_confidence_suitable,
                'low_confidence_count': low_confidence_suitable,
                'recommendation': self.get_quality_recommendation(high_confidence_suitable, medium_confidence_suitable, total_candidates)
            }
        }
        
        # Save report
        report_file = self.data_path / f'bao_cao_nang_cao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Báo cáo nâng cao đã được tạo và lưu tại {report_file}")
        return report
    
    def get_quality_recommendation(self, high_conf, medium_conf, total):
        """
        Đưa ra khuyến nghị dựa trên chất lượng ứng viên
        """
        if total == 0:
            return "Không có dữ liệu để phân tích"
        
        high_ratio = high_conf / total
        medium_ratio = medium_conf / total
        
        if high_ratio > 0.3:
            return "Chất lượng ứng viên rất tốt, nhiều ứng viên xuất sắc"
        elif high_ratio + medium_ratio > 0.5:
            return "Chất lượng ứng viên tốt, có thể tiến hành phỏng vấn"
        elif high_ratio + medium_ratio > 0.2:
            return "Chất lượng ứng viên trung bình, cần sàng lọc kỹ hơn"
        else:
            return "Chất lượng ứng viên thấp, nên mở rộng nguồn tuyển dụng"


def main():
    """
    Hàm main để chạy ứng dụng
    """
    parser = argparse.ArgumentParser(description='Hệ thống Hỗ trợ Ra quyết định Tuyển dụng - Phiên bản Nâng cao')
    parser.add_argument('--mode', choices=['train', 'predict', 'batch', 'demo'], 
                       default='demo', help='Chế độ chạy hệ thống')
    parser.add_argument('--input', type=str, help='File đầu vào cho dự đoán hàng loạt')
    parser.add_argument('--model-path', type=str, default='models/', 
                       help='Đường dẫn thư mục mô hình')
    parser.add_argument('--data-path', type=str, default='data/', 
                       help='Đường dẫn thư mục dữ liệu')
    
    args = parser.parse_args()
    
    # Initialize system
    hr_system = HRDecisionSupportSystemEnhanced(args.model_path, args.data_path)
    
    if args.mode == 'train':
        logger.info("🧠 Chế độ huấn luyện được chọn (với logic kỹ năng nâng cao)")
        accuracy = hr_system.train_model()
        print(f"🎉 Huấn luyện mô hình hoàn tất với độ chính xác: {accuracy:.3f}")
    
    elif args.mode == 'predict':
        logger.info("🎯 Chế độ dự đoán đơn (với logic kỹ năng nâng cao)")
        # Test case: Data Scientist thiếu SQL
        candidate = {
            'candidate_id': 'TEST_DATA_SCIENTIST',
            'years_experience': 3,
            'education_level': 'bachelor',
            'skills': 'python, machine learning, statistics',  # Thiếu SQL
            'experience_description': '3 năm kinh nghiệm với machine learning và Python',
            'position_applied': 'data_scientist'
        }
        
        result = hr_system.predict_candidate(candidate)
        print("📋 Kết quả dự đoán nâng cao:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == 'batch':
        if not args.input:
            print("❌ Lỗi: Cần file --input cho chế độ dự đoán hàng loạt")
            return
        
        logger.info(f"👥 Chế độ dự đoán hàng loạt cho file: {args.input} (với logic kỹ năng nâng cao)")
        results = hr_system.batch_predict(args.input)
        report = hr_system.generate_report(results)
        print("✅ Dự đoán hàng loạt hoàn tất với logic kỹ năng nâng cao")
        print("📊 Báo cáo tổng hợp:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    else:  # demo mode
        logger.info("🎮 Chế độ demo - Kiểm thử logic kỹ năng nâng cao")
        
        # Test multiple scenarios with different skill requirements
        test_cases = [
            {
                'name': 'Data Scientist có đủ kỹ năng bắt buộc (SQL + Python)',
                'data': {
                    'candidate_id': 'DS_GOOD',
                    'years_experience': 3,
                    'education_level': 'bachelor',
                    'skills': 'python, sql, machine learning, statistics',
                    'position_applied': 'data_scientist'
                }
            },
            {
                'name': 'Data Scientist thiếu kỹ năng bắt buộc (chỉ có Python, thiếu SQL)',
                'data': {
                    'candidate_id': 'DS_MISSING_SQL',
                    'years_experience': 5,
                    'education_level': 'master',
                    'skills': 'python, machine learning, statistics, deep learning',
                    'position_applied': 'data_scientist'
                }
            },
            {
                'name': 'Web Developer có đủ kỹ năng bắt buộc (HTML + CSS + JS)',
                'data': {
                    'candidate_id': 'WEB_GOOD',
                    'years_experience': 2,
                    'education_level': 'bachelor',
                    'skills': 'html, css, javascript, react, nodejs',
                    'position_applied': 'web_developer'
                }
            },
            {
                'name': 'Web Developer thiếu kỹ năng bắt buộc (chỉ có JS, thiếu HTML+CSS)',
                'data': {
                    'candidate_id': 'WEB_MISSING',
                    'years_experience': 4,
                    'education_level': 'bachelor',
                    'skills': 'javascript, react, nodejs, mongodb',
                    'position_applied': 'web_developer'
                }
            },
            {
                'name': 'DevOps Engineer có kỹ năng phù hợp',
                'data': {
                    'candidate_id': 'DEVOPS_GOOD',
                    'years_experience': 4,
                    'education_level': 'bachelor',
                    'skills': 'linux, docker, kubernetes, aws, terraform',
                    'position_applied': 'devops_engineer'
                }
            }
        ]
        
        print("=== KIỂM THỬ LOGIC KỸ NĂNG NÂNG CAO ===\n")
        
        for test_case in test_cases:
            print(f"--- {test_case['name']} ---")
            result = hr_system.predict_candidate(test_case['data'])
            
            print(f"Kết quả: {'✓ PHÙ HỢP' if result['prediction'] == 'Suitable' else '✗ CHƯA PHÙ HỢP'}")
            print(f"Độ tin cậy: {result['confidence']:.1%}")
            print(f"Khuyến nghị: {result['recommendation_vietnamese']}")
            
            # Hiển thị phân tích kỹ năng chi tiết
            if 'skill_analysis' in result:
                skill_analysis = result['skill_analysis']
                if skill_analysis['requirements_found']:
                    print("Phân tích kỹ năng:")
                    print(f"  - Có đủ kỹ năng bắt buộc: {'✓' if skill_analysis['has_required_skills'] else '✗'}")
                    if skill_analysis['missing_required']:
                        print(f"  - Thiếu kỹ năng: {', '.join(skill_analysis['missing_required'])}")
                    if skill_analysis['matching_preferred']:
                        print(f"  - Kỹ năng ưu tiên: {', '.join(skill_analysis['matching_preferred'])}")
                    if skill_analysis['matching_bonus']:
                        print(f"  - Kỹ năng bonus: {', '.join(skill_analysis['matching_bonus'])}")
                    print(f"  - Điểm kỹ năng: {skill_analysis['skill_score']:.1f}/10")
            
            if 'detailed_feedback' in result:
                print("Chi tiết đánh giá:")
                for feedback in result['detailed_feedback']:
                    print(f"  {feedback}")
            
            print()

if __name__ == "__main__":
    main()