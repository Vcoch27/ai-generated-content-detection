import cv2
import os
import logging
import shutil

logger = logging.getLogger(__name__)


def get_video_metadata(video_path):
    """
    Lấy các thông số cơ bản của video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def extract_frames(video_path, num_frames=20, output_dir="temp_frames"):
    """
    Cắt video thành N frame đại diện cách đều nhau.
    Trả về danh sách đường dẫn các frame đã lưu.
    """
    # 1. Dọn dẹp và tạo thư mục tạm
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    metadata = get_video_metadata(video_path)
    if not metadata or metadata["total_frames"] == 0:
        logger.error("Không thể đọc video hoặc video rỗng.")
        return []

    total_frames = metadata["total_frames"]

    # 2. Tính toán khoảng cách lấy mẫu (Step)
    # Ví dụ: video 200 frame, lấy 20 frame -> cứ 10 frame lấy 1 cái
    step = max(1, total_frames // num_frames)

    frame_paths = []
    cap = cv2.VideoCapture(video_path)

    count = 0
    extracted_count = 0

    while extracted_count < num_frames:
        # Nhảy đến vị trí frame cần lấy
        frame_id = extracted_count * step
        if frame_id >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            break

        # Lưu frame thành file ảnh để pipeline cũ có thể đọc được
        frame_name = f"frame_{extracted_count:03d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        # Lưu chất lượng cao (95%)
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        frame_paths.append(frame_path)
        extracted_count += 1

    cap.release()
    logger.info(f"Đã trích xuất {len(frame_paths)} frames từ video {video_path}")
    return frame_paths


def cleanup_frames(directory="temp_frames"):
    """
    Xóa thư mục tạm sau khi xử lý xong
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logger.info(f"Đã dọn dẹp thư mục tạm: {directory}")