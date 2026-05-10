import logging
import numpy as np

logger = logging.getLogger(__name__)


CV_FEATURE_NAMES = [
    "B_Mean", "B_Std", "B_Skew", "G_Mean", "G_Std", "G_Skew", "R_Mean", "R_Std", "R_Skew",
    "H_Mean", "H_Std", "H_Skew", "S_Mean", "S_Std", "S_Skew", "V_Mean", "V_Std", "V_Skew",
    "Edge_Laplacian", "Edge_Canny",
    "Texture_Contrast", "Texture_Correlation", "Texture_Energy", "Texture_Homogeneity",
    "Freq_FFT_Mean", "Freq_FFT_Std", "Freq_FFT_Max"
]

# Bộ từ điển song hành
DUAL_FEATURE_DICTIONARY = {
    "Color": {
        "AI": "Phân tích màu sắc cho thấy sự phân bổ pixel quá hoàn hảo hoặc dải màu bị lệch (skew) đặc trưng của các thuật toán sinh ảnh.",
        "REAL": "Sự phân bổ màu sắc và độ bão hòa nằm trong dải quang phổ tự nhiên của cảm biến máy ảnh, không có dấu hiệu can thiệp toán học."
    },
    "Edge": {
        "AI": "Các đường viền (Edges) xuất hiện sự thiếu nhất quán, quá sắc nét hoặc bị nhòe theo quy luật của các bộ lọc nhân tạo.",
        "REAL": "Độ sắc nét và các vùng chuyển tiếp cạnh (depth of field) tuân thủ đúng quy luật vật lý của thấu kính và ánh sáng tự nhiên."
    },
    "Texture": {
        "AI": "Kết cấu bề mặt quá mịn (over-smoothed) hoặc xuất hiện các họa tiết lặp lại siêu nhỏ do lỗi tái tạo của mô hình Generative.",
        "REAL": "Bề mặt vật thể giữ được độ chi tiết và hạt nhiễu (noise grain) tự nhiên của vật liệu thực tế, không bị làm mịn nhân tạo."
    },
    "Freq": {
        "AI": "Phân tích tần số FFT phát hiện các 'nhiễu định kỳ' (periodic artifacts) - bằng chứng thép của quá trình lấy mẫu từ AI.",
        "REAL": "Phổ tần số hiển thị các nhiễu trắng ngẫu nhiên và chi tiết tần số cao đặc trưng của một tấm ảnh chụp thực tế."
    }
}


def get_category(idx):
    """
    Ánh xạ index (0-26) về nhóm đặc trưng cv tương ứng
    """
    if idx < 18:
        return "Color"
    elif idx < 20:
        return "Edge"
    elif idx < 24:
        return "Texture"
    else:
        return "Freq"


def get_local_feature_impact(explainer, input_vector, prediction_label):
    """
    Giải thích cục bộ: đưa ra 5 thuộc tính cv ảnh hưởng nhiều nhất đến kết quả dự đoán
    """
    # 1. Tính SHAP values
    shap_results = explainer.shap_values(input_vector)

    class_idx = 0 if prediction_label == "AI_GENERATED" else 1

    # 2. Ép phẳng mảng về 1D
    current_shap = np.array(shap_results[:, :, class_idx]).flatten()
    # Bây giờ current_shap là mảng có 77 phần tử (1D)

    # 3. Chỉ lấy 27 thuộc tính CV đầu tiên
    cv_shap = current_shap[:27]

    # 4. Tìm Top 5 thuộc tính có giá trị đóng góp LỚN NHẤT (trị tuyệt đối)
    # Vì đã flatten nên top_indices sẽ chứa các con số nguyên (int)
    top_indices = np.argsort(np.abs(cv_shap))[-5:][::-1]

    # 5. Tính tổng đóng góp để chia tỉ lệ %
    total_abs_contribution = np.sum(np.abs(current_shap))
    if total_abs_contribution == 0: total_abs_contribution = 1e-10

    pov_key = "AI" if prediction_label == "AI_GENERATED" else "REAL"
    analysis_results = []

    for idx in top_indices:
        idx_int = int(idx)
        impact_val = cv_shap[idx_int]
        category = get_category(idx_int)

        impact_pct = (abs(impact_val) / total_abs_contribution) * 100

        analysis_results.append({
            "feature_name": CV_FEATURE_NAMES[idx_int],
            "category": category,
            "impact_score": round(float(impact_pct), 2),
            "description": DUAL_FEATURE_DICTIONARY[category][pov_key]
        })

    return analysis_results


def log_all_feature_importances(rf_model):
    """
    In ra toàn bộ 77 đặc trưng theo thứ tự giảm dần của mức độ quan trọng
    """
    # 1. Lấy mảng 77 giá trị
    importances = rf_model.feature_importances_

    # 2. Tạo danh sách 77 tên (27 CV + 50 CNN)
    cnn_feature_names = [f"CNN_Latent_{i}" for i in range(50)]
    all_feature_names = CV_FEATURE_NAMES + cnn_feature_names

    # 3. Kết hợp tên và giá trị, sau đó sắp xếp
    feature_ranking = sorted(
        zip(all_feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    # 4. In ra Log (Console)
    logger.info("=" * 50)
    logger.info(" BẢNG XẾP HẠNG 77 ĐẶC TRƯNG HYBRID ")
    logger.info("=" * 50)

    total_cv_impact = 0
    total_cnn_impact = 0

    for rank, (name, val) in enumerate(feature_ranking, 1):
        impact_pct = val * 100
        category = "CV" if name in CV_FEATURE_NAMES else "CNN"

        # Cộng dồn để tính tổng tỉ lệ
        if category == "CV":
            total_cv_impact += impact_pct
        else:
            total_cnn_impact += impact_pct

        # Chỉ in đậm hoặc đánh dấu những thuộc tính Top đầu
        marker = "⭐" if rank <= 10 else "  "
        logger.info(f"{marker} Rank {rank:02d}: {name:<20} | {category} | {impact_pct:>6.2f}%")

    logger.info("-" * 50)
    logger.info(f" TỔNG TRỌNG SỐ CV:  {total_cv_impact:>6.2f}%")
    logger.info(f" TỔNG TRỌNG SỐ CNN: {total_cnn_impact:>6.2f}%")
    logger.info("=" * 50)