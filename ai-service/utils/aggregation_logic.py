import numpy as np
import logging

logger = logging.getLogger(__name__)


def perform_video_aggregation(frame_results):
    """
    frame_results: Danh sách các dict kết quả từ từng frame.
    Mỗi phần tử có dạng: {
        'prediction': 'AI_GENERATED'/'REAL',
        'confidence': float,
        'ai_prob': float,
        'real_prob': float,
        'shap_values': array(77,),  <-- SHAP cục bộ của từng frame
        'frame_path': str
    }
    """
    # 1. Đếm số phiếu (Majority Voting)
    ai_votes = sum(1 for r in frame_results if r['prediction'] == 'AI_GENERATED')
    real_votes = len(frame_results) - ai_votes

    # Tính toán độ nhất quán (Frame Consistency)
    consistency = (max(ai_votes, real_votes) / len(frame_results)) * 100

    # 2. Quyết định nhãn cuối cùng
    final_prediction = "AI_GENERATED" if ai_votes > real_votes else "REAL"

    # 3. Tính độ tin cậy trung bình (Soft Voting)
    if final_prediction == "AI_GENERATED":
        avg_confidence = np.mean([r['ai_prob'] for r in frame_results])
    else:
        avg_confidence = np.mean([r['real_prob'] for r in frame_results])

    # 4. LỌC PHE: Chỉ lấy bằng chứng ủng hộ kết luận
    supportive_results = [r for r in frame_results if r['prediction'] == final_prediction]

    # 5. Chọn Key Frame (Frame có độ tin cậy cao nhất trong phe thắng)
    key_frame_data = max(supportive_results, key=lambda x: x['confidence'])

    # 6. Tổng hợp SHAP Values trung bình của phe thắng
    all_shap_supportive = np.array([r['shap_values'] for r in supportive_results])
    mean_shap = np.mean(all_shap_supportive, axis=0)

    return {
        "prediction": final_prediction,
        "confidence": round(float(avg_confidence), 2),
        "consistency": round(float(consistency), 2),
        "key_frame_path": key_frame_data['frame_path'],
        "mean_shap": mean_shap,
        "votes": {"AI": ai_votes, "REAL": real_votes},
        "timeline": [round(float(r['ai_prob']), 2) for r in frame_results]
    }