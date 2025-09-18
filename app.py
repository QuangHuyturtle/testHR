from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename

# Import HR DSS system
from hr_dss_main import HRDecisionSupportSystem

app = Flask(__name__)
app.secret_key = 'hr_dss_demo_key_vietnamese_2025'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize HR system
hr_system = HRDecisionSupportSystem()

@app.route('/')
def index():
    """Trang chủ - Dashboard chính"""
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Huấn luyện mô hình"""
    if request.method == 'POST':
        try:
            print("🧠 Bắt đầu quá trình huấn luyện...")
            accuracy = hr_system.train_model()
            success_msg = f'🎉 Huấn luyện mô hình thành công! Độ chính xác: {accuracy:.1%}'
            flash(success_msg, 'success')
            
            return jsonify({
                'success': True, 
                'accuracy': accuracy,
                'message': success_msg,
                'accuracy_percentage': f"{accuracy:.1%}"
            })
        except Exception as e:
            error_msg = f'❌ Huấn luyện thất bại: {str(e)}'
            flash(error_msg, 'error')
            print(f"Lỗi huấn luyện: {e}")
            return jsonify({
                'success': False, 
                'error': str(e),
                'message': error_msg
            })
    
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_single():
    """Dự đoán cho một ứng viên"""
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            candidate_data = {
                'candidate_id': request.form.get('candidate_id', '').strip(),
                'years_experience': int(request.form.get('years_experience', 0)),
                'education_level': request.form.get('education_level', ''),
                'skills': request.form.get('skills', '').strip(),
                'experience_description': request.form.get('experience_description', '').strip(),
                'position_applied': request.form.get('position_applied', '').strip()
            }
            
            # Validate dữ liệu
            if not candidate_data['candidate_id']:
                flash('❌ Vui lòng nhập Mã ứng viên', 'error')
                return render_template('predict.html', candidate=candidate_data)
            
            if candidate_data['years_experience'] < 0:
                flash('❌ Số năm kinh nghiệm không thể âm', 'error')
                return render_template('predict.html', candidate=candidate_data)
            
            if not candidate_data['skills']:
                flash('❌ Vui lòng nhập danh sách kỹ năng', 'error')
                return render_template('predict.html', candidate=candidate_data)
            
            # Thực hiện dự đoán
            print(f"🎯 Đang dự đoán cho ứng viên: {candidate_data['candidate_id']}")
            result = hr_system.predict_candidate(candidate_data)
            
            # Thông báo thành công
            success_msg = f"✅ Dự đoán thành công cho ứng viên {result['candidate_id']}: {result.get('prediction_vietnamese', result['prediction'])}"
            flash(success_msg, 'success')
            
            return render_template('predict.html', result=result, candidate=candidate_data)
            
        except ValueError as e:
            error_msg = f'⚠️ Lỗi dữ liệu: {str(e)}'
            flash(error_msg, 'error')
            print(f"Validation error: {e}")
            return render_template('predict.html', candidate=candidate_data)
        except Exception as e:
            error_msg = f'❌ Dự đoán thất bại: {str(e)}'
            flash(error_msg, 'error')
            print(f"Prediction error: {e}")
            return render_template('predict.html', error=str(e), candidate=request.form.to_dict())
    
    return render_template('predict.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Dự đoán hàng loạt từ file CSV"""
    if request.method == 'POST':
        # Kiểm tra file upload
        if 'file' not in request.files:
            flash('❌ Không tìm thấy file. Vui lòng chọn file CSV.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('❌ Chưa chọn file. Vui lòng chọn file CSV chứa dữ liệu ứng viên.', 'error')
            return redirect(request.url)
        
        # Kiểm tra định dạng file
        if file and file.filename.lower().endswith('.csv'):
            try:
                # Lưu file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                file.save(filepath)
                
                print(f"📁 File đã lưu: {filepath}")
                
                # Xử lý dự đoán hàng loạt
                print("👥 Bắt đầu xử lý dự đoán hàng loạt...")
                results = hr_system.batch_predict(filepath)
                report = hr_system.generate_report(results)
                
                # Thông báo thành công
                success_msg = f'🎉 Xử lý hàng loạt thành công! Đã xử lý {len(results)} ứng viên, trong đó {report["suitable_candidates"]} ứng viên phù hợp.'
                flash(success_msg, 'success')
                
                # Xóa file tạm
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return render_template('batch_results.html', results=results, report=report)
                
            except Exception as e:
                error_msg = f'❌ Lỗi xử lý file: {str(e)}'
                flash(error_msg, 'error')
                print(f"Batch processing error: {e}")
        else:
            flash('❌ Chỉ chấp nhận file CSV. Vui lòng chọn file có đuôi .csv', 'error')
    
    return render_template('batch.html')

@app.route('/api/status')
def api_status():
    """API trạng thái hệ thống"""
    model_loaded = hr_system.model is not None
    
    status = {
        'model_loaded': model_loaded,
        'status': 'Sẵn sàng' if model_loaded else 'Cần huấn luyện',
        'status_text': 'Mô hình đã sẵn sàng để dự đoán' if model_loaded else 'Mô hình chưa được huấn luyện',
        'timestamp': datetime.now().isoformat(),
        'formatted_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'version': '2.0.0',
        'language': 'Tiếng Việt',
        'features': {
            'single_prediction': True,
            'batch_prediction': True,
            'model_training': True,
            'vietnamese_interface': True
        }
    }
    return jsonify(status)

@app.route('/api/sample-data')
def download_sample_data():
    """Tải xuống file CSV mẫu"""
    try:
        sample_data = """candidate_id,years_experience,education_level,skills,experience_description,position_applied
UV001,5,bachelor,"python, sql, teamwork, communication","Có 5 năm kinh nghiệm phát triển phần mềm với chuyên môn về Python và SQL",developer
UV002,3,master,"data analysis, machine learning, statistics","Chuyên gia phân tích dữ liệu với bằng Thạc sĩ và 3 năm kinh nghiệm",analyst
UV003,8,bachelor,"leadership, project management, agile","Quản lý dự án có kinh nghiệm với 8 năm trong lĩnh vực công nghệ",manager
UV004,2,associate,"excel, communication, teamwork","Nhân viên mới tốt nghiệp cao đẳng với kỹ năng cơ bản",coordinator
UV005,10,master,"python, machine learning, leadership, sql","Chuyên gia cấp cao với 10 năm kinh nghiệm và khả năng lãnh đạo",senior_developer"""
        
        # Tạo response với file CSV
        from flask import Response
        
        response = Response(
            sample_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=mau_du_lieu_ung_vien.csv"}
        )
        
        flash('📥 File mẫu đã được tải xuống thành công!', 'success')
        return response
        
    except Exception as e:
        flash(f'❌ Lỗi tải file mẫu: {str(e)}', 'error')
        return redirect(url_for('batch_predict'))

@app.errorhandler(404)
def page_not_found(e):
    """Trang lỗi 404"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Trang lỗi 500"""
    return render_template('500.html'), 500

@app.context_processor
def inject_datetime():
    """Inject datetime vào tất cả templates"""
    return {
        'now': datetime.now(),
        'formatted_now': datetime.now().strftime("%d/%m/%Y %H:%M")
    }

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG HR DSS - PHIÊN BẢN TIẾNG VIỆT")
    print("=" * 60)
    
    # Kiểm tra và huấn luyện mô hình nếu cần
    if hr_system.model is None:
        print("⚠️  Mô hình chưa được huấn luyện")
        print("🧠 Đang khởi tạo và huấn luyện mô hình...")
        try:
            accuracy = hr_system.train_model()
            print(f"✅ Huấn luyện hoàn tất! Độ chính xác: {accuracy:.1%}")
        except Exception as e:
            print(f"❌ Lỗi huấn luyện: {e}")
            print("⚠️  Hệ thống sẽ chạy nhưng cần huấn luyện mô hình thủ công")
    else:
        print("✅ Mô hình đã sẵn sàng!")
    
    print("\n🌐 THÔNG TIN TRUY CẬP:")
    print("📊 Dashboard chính:     http://localhost:5000")
    print("🧠 Huấn luyện mô hình:  http://localhost:5000/train")
    print("🎯 Dự đoán đơn:        http://localhost:5000/predict")  
    print("👥 Dự đoán hàng loạt:   http://localhost:5000/batch")
    print("⚙️  API trạng thái:     http://localhost:5000/api/status")
    print("\n🎉 Hệ thống đã sẵn sàng! Nhấn Ctrl+C để dừng.")
    print("=" * 60)
    
    # Chạy ứng dụng
    app.run(
        debug=True, 
        host='127.0.0.1', 
        port=5000,
        use_reloader=True
    )