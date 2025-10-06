import os
import cv2
import time
import json
import telebot
import re
import threading
from ultralytics import YOLO
from datetime import datetime

# Telegram Bot Token
BOT_TOKEN = '7516933541:AAEhWpgRzFzmWOeZRQjXhu3ZJM_yWhjvrME'
bot = telebot.TeleBot(BOT_TOKEN)

# Admin Chat ID (o'zingizning chat ID ni qo'ying, /start bu botga yuborib oling)
ADMIN_CHAT_ID = '6036366867'  # Masalan: 123456789

# Model yuklash
model2 = YOLO("/home/bahrombek/Desktop/RAILSAFE/number_lines/runs/detect/train/weights/best.pt")  # raqam joyi
model3 = YOLO("/home/bahrombek/Desktop/RAILSAFE/data_number/runs/detect/train/weights/best.pt")   # raqam belgilarini o‘qish

# Class mapping (36 belgi)
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

# Papkalar
VIOLATIONS_IMAGES_DIR = '/home/bahrombek/Desktop/RailSafeAI_v1/vehicle_data/violations/images'
VIOLATIONS_VIDEOS_DIR = '/home/bahrombek/Desktop/RailSafeAI_v1/vehicle_data/violations/videos'
NORMAL_IMAGES_DIR = '/home/bahrombek/Desktop/RailSafeAI_v1/vehicle_data/normal/images'
CROPPED_DIR = '/home/bahrombek/Desktop/RailSafeAI_v1/vehicle_data/cropped'

# Papkalarni yaratish
os.makedirs(CROPPED_DIR, exist_ok=True)

# Ishlangan fayllar va Track ID lar uchun set
processed_files = set()
processed_track_ids = set()

def draw_plate_text(image, text):
    h, w = image.shape[:2]
    panel_height = int(h * 0.15)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    font_scale = max(0.8, w / 600)
    thickness = max(2, w // 400)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - tw) // 2
    text_y = panel_height // 2 + th // 2
    cv2.putText(image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return image

def parse_txt_for_box(txt_path):
    """Txt fayldan box koordinatalarini olish"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '# Box coordinates (x1, y1, x2, y2):' in line:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        match = re.match(r'(\d+),(\d+),(\d+),(\d+)', next_line)
                        if match:
                            x1, y1, x2, y2 = map(int, match.groups())
                            return x1, y1, x2, y2
        return None
    except Exception as e:
        print(f"Xato txt o'qishda: {e}")
        return None

def extract_track_id(image_name):
    """Image nomidan track ID ni chiqarish"""
    match = re.search(r'ID(\d+)', image_name)
    if match:
        return match.group(1)
    return None

def get_timestamp_from_filename(filename, ext):
    """Fayl nomidan timestamp ni chiqarish, xatoliklarni boshqarish"""
    ts_str = filename.split('_')[-1].replace(ext, '').replace('_', '')
    try:
        return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
    except ValueError:
        print(f"Noto'g'ri timestamp: {ts_str} in {filename}")
        return datetime.min  # Eng eski vaqtni qaytarish, shunda u tanlanmaydi

def find_matching_normal_files(violation_image_name, track_id):
    """Track ID bo'yicha normal papkadagi mos rasm va txt ni topish"""
    if not track_id:
        return None, None
    
    candidates = []
    for file in os.listdir(NORMAL_IMAGES_DIR):
        if re.match(rf'.*_NORMAL_ID{track_id}_.*\.jpg$', file):
            candidates.append(file)
    
    if not candidates:
        return None, None
    
    # Eng yaqin timestamp ni topish (eng oxirgi)
    latest_file = max(candidates, key=lambda f: get_timestamp_from_filename(f, '.jpg'))
    
    img_path = os.path.join(NORMAL_IMAGES_DIR, latest_file)
    txt_path = os.path.join(NORMAL_IMAGES_DIR, latest_file.replace('.jpg', '.txt'))
    
    if os.path.exists(txt_path):
        return img_path, txt_path
    return None, None

def find_matching_video(violation_image_name, track_id):
    """Track ID bo'yicha violations videos da video topish"""
    if not track_id:
        return None
    
    candidates = []
    for file in os.listdir(VIOLATIONS_VIDEOS_DIR):
        if re.match(rf'.*_ID{track_id}_.*\.avi$', file):
            candidates.append(file)
    
    if not candidates:
        return None
    
    # Eng yaqin (eng oxirgi)
    latest_video = max(candidates, key=lambda f: get_timestamp_from_filename(f, '.avi'))
    video_path = os.path.join(VIOLATIONS_VIDEOS_DIR, latest_video)
    return video_path

def process_violation_image(violation_image_path):
    """Violations rasmini ishlash: crop, OCR, botga yuborish"""
    image_name = os.path.basename(violation_image_path)
    if image_name in processed_files:
        return
    
    track_id = extract_track_id(image_name)
    if not track_id:
        print(f"Track ID topilmadi: {image_name}")
        return
    
    # Track ID allaqachon qayta ishlatilgan bo'lsa, o'tkazib yuborish
    if track_id in processed_track_ids:
        print(f"Track ID allaqachon qayta ishlatilgan: {track_id} ({image_name})")
        processed_files.add(image_name)  # Faylni ham belgilash
        return
    
    # Normal papkadagi mos rasm va txt ni topish
    normal_img_path, txt_path = find_matching_normal_files(image_name, track_id)
    if not normal_img_path or not txt_path:
        print(f"Mos normal rasm topilmadi: {image_name} (ID: {track_id})")
        return
    
    # Box olish
    box = parse_txt_for_box(txt_path)
    if not box:
        print(f"Box topilmadi: {txt_path}")
        return
    
    x1, y1, x2, y2 = box
    
    # Normal rasm yuklash va crop qilish
    normal_img = cv2.imread(normal_img_path)
    if normal_img is None:
        print(f"Normal rasm yuklanmadi: {normal_img_path}")
        return
    
    crop_car = normal_img[y1:y2, x1:x2].copy()
    if crop_car.size == 0:
        print("Crop bo'sh")
        return
    
    # Violation rasm yuklash
    violation_img = cv2.imread(violation_image_path)
    if violation_img is None:
        print(f"Violation rasm yuklanmadi: {violation_image_path}")
        return
    
    car_id = f"car_violation_ID{track_id}"
    
    # Model2: raqam joyini aniqlash (normal crop dan)
    results2 = model2(crop_car)
    plate_text = "UNKNOWN"
    
    for res2 in results2:
        for box2 in res2.boxes:
            px1, py1, px2, py2 = map(int, box2.xyxy[0])
            crop_plate = crop_car[py1:py2, px1:px2].copy()
            
            if crop_plate.size == 0:
                continue
            
            crop_plate = cv2.resize(crop_plate, (800, int(800 * crop_plate.shape[0] / crop_plate.shape[1])))
            
            # Model3: belgilarni o‘qish
            results3 = model3(crop_plate)
            digits = []
            for res3 in results3:
                for box3 in res3.boxes:
                    cls3 = int(box3.cls[0])
                    char = CLASSES[cls3]
                    x_coord = box3.xyxy[0][0]
                    digits.append((x_coord, char))
            
            # Sort qilish
            digits = sorted(digits, key=lambda x: x[0])
            plate_text = "".join([d[1] for d in digits])
            
            if len(plate_text) < 6:
                plate_text = "UNKNOWN"
            
            # Plate ustiga yozish
            crop_plate = draw_plate_text(crop_plate, plate_text)
            
            # Saqlash
            plate_save_path = os.path.join(CROPPED_DIR, f"{car_id}_plate_{plate_text}.jpg")
            cv2.imwrite(plate_save_path, crop_plate)
    
    # Agar normal dan o'qilmasa, violation rasmdan ham sinab ko'rish (ixtiyoriy, lekin hozircha normal dan qilamiz)
    if plate_text == "UNKNOWN":
        print(f"Normal rasmdan o'qilmasdi, violation rasmdan sinab ko'ramiz...")
        # Violation crop (taxminiy box, lekin aniq emas, shuning uchun normal box ishlatamiz)
        violation_crop = violation_img[y1:y2, x1:x2].copy() if y1 < violation_img.shape[0] and x1 < violation_img.shape[1] else crop_car
        results2_viol = model2(violation_crop)
        for res2_v in results2_viol:
            for box2_v in res2_v.boxes:
                px1_v, py1_v, px2_v, py2_v = map(int, box2_v.xyxy[0])
                crop_plate_v = violation_crop[py1_v:py2_v, px1_v:px2_v].copy()
                
                if crop_plate_v.size == 0:
                    continue
                
                crop_plate_v = cv2.resize(crop_plate_v, (800, int(800 * crop_plate_v.shape[0] / crop_plate_v.shape[1])))
                
                results3_v = model3(crop_plate_v)
                digits_v = []
                for res3_v in results3_v:
                    for box3_v in res3_v.boxes:
                        cls3_v = int(box3_v.cls[0])
                        char_v = CLASSES[cls3_v]
                        x_coord_v = box3_v.xyxy[0][0]
                        digits_v.append((x_coord_v, char_v))
                
                digits_v = sorted(digits_v, key=lambda x: x[0])
                plate_text_v = "".join([d[1] for d in digits_v])
                
                if len(plate_text_v) >= 6:
                    plate_text = plate_text_v
                    crop_plate = draw_plate_text(crop_plate_v, plate_text)
                    plate_save_path = os.path.join(CROPPED_DIR, f"{car_id}_plate_{plate_text}.jpg")
                    cv2.imwrite(plate_save_path, crop_plate)
                    break
    
    # Agar hali ham UNKNOWN bo'lsa, jo'natmaymiz
    if plate_text == "UNKNOWN":
        print(f"Raqam o'qilmasdi: {image_name} (ID: {track_id})")
        processed_files.add(image_name)
        processed_track_ids.add(track_id)
        return
    
    # Crop car ustiga yozish
    crop_car = draw_plate_text(crop_car, plate_text)
    
    # Crop car saqlash
    crop_save_path = os.path.join(CROPPED_DIR, f"{car_id}_{plate_text}.jpg")
    cv2.imwrite(crop_save_path, crop_car)
    
    # Video topish
    video_path = find_matching_video(image_name, track_id)
    video_exists = video_path is not None
    
    # Botga chiroyli habar jo'natish (Markdown formatda)
    try:
        camera_info = image_name.split('_')[0] if '_' in image_name else 'unknown'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message_text = f"""🚨 *YANGI QO'IDA BUZILISHI ANIQLANDI!* 🚨

**Avtomobil raqami:** `{plate_text}`
**Kamera ID:** `{camera_info}`
**Track ID:** `{track_id}`
**Aniqlangan vaqt:** `{timestamp}`

Quyida to'liq ma'lumotlar:
"""
        
        bot.send_message(ADMIN_CHAT_ID, message_text, parse_mode='Markdown')
        
        # Normal rasm (kirish momenti) jo'natish
        normal_save_path = os.path.join(CROPPED_DIR, f"{car_id}_normal_full.jpg")
        cv2.imwrite(normal_save_path, normal_img)
        with open(normal_save_path, 'rb') as normal_photo:
            bot.send_photo(ADMIN_CHAT_ID, normal_photo, caption=f"📸 *Normal (kirish) rasm (to'liq kadr)*\nTrack ID: {track_id}\nKamera: {camera_info}", parse_mode='Markdown')
        
        # Violation rasm jo'natish
        violation_save_path = os.path.join(CROPPED_DIR, f"{car_id}_violation_full.jpg")
        cv2.imwrite(violation_save_path, violation_img)
        with open(violation_save_path, 'rb') as violation_photo:
            bot.send_photo(ADMIN_CHAT_ID, violation_photo, caption=f"⚠️ *Violation rasm (to'liq kadr)*\nTrack ID: {track_id}\nKamera: {camera_info}", parse_mode='Markdown')
        
        # Crop avtomobil jo'natish
        with open(crop_save_path, 'rb') as crop_photo:
            bot.send_photo(ADMIN_CHAT_ID, crop_photo, caption=f"🔍 *Crop qilingan avtomobil (normal dan)*\nRaqam: `{plate_text}`\nTrack ID: {track_id}", parse_mode='Markdown')
        
        # Raqam yaqin rasmi
        with open(plate_save_path, 'rb') as plate_photo:
            bot.send_photo(ADMIN_CHAT_ID, plate_photo, caption=f"🔎 *Raqam yaqindan (OCR natijasi)*\nRaqam: `{plate_text}`", parse_mode='Markdown')
        
        # Video jo'natish (agar mavjud bo'lsa)
        if video_exists:
            with open(video_path, 'rb') as video:
                bot.send_video(ADMIN_CHAT_ID, video, caption=f"🎥 *Buzilish video (to'liq)*\nTrack ID: {track_id}\nDavomiyligi: ~{os.path.getsize(video_path)/1024/1024:.1f} MB", parse_mode='Markdown')
        
        print(f"Jo'natildi: {image_name}, Raqam: {plate_text}, ID: {track_id}")
        
    except Exception as e:
        print(f"Bot xatosi: {e}")
    
    # Processed qilish: ham faylni, ham Track ID ni belgilash
    processed_files.add(image_name)
    processed_track_ids.add(track_id)

def scan_violations_dir():
    """Violations papkasini skan qilish"""
    new_files = []
    for file in os.listdir(VIOLATIONS_IMAGES_DIR):
        if file.endswith('.jpg') and file not in processed_files:
            new_files.append(file)
    
    for file in new_files:
        image_path = os.path.join(VIOLATIONS_IMAGES_DIR, file)
        process_violation_image(image_path)

# Telegram bot handlerlari (ixtiyoriy, test uchun)
@bot.message_handler(commands=['start'])
def start_message(message):
    global ADMIN_CHAT_ID
    ADMIN_CHAT_ID = str(message.chat.id)
    bot.reply_to(message, f"Bot ishga tushdi! Chat ID saqlandi: {ADMIN_CHAT_ID}")

@bot.message_handler(commands=['scan'])
def manual_scan(message):
    if str(message.chat.id) == ADMIN_CHAT_ID:
        scan_violations_dir()
        bot.reply_to(message, "Skan qilindi! Yangi fayllar tekshirildi.")

def monitoring_loop():
    """Doimiy monitoring loop"""
    print("Monitoring boshlandi. Har 10 soniyada papkani tekshirish...")
    while True:
        scan_violations_dir()
        time.sleep(5)

if __name__ == "__main__":
    print("OCR Bot ishga tushdi. Violations papkasini kuzatmoqda...")
    
    # Dastlabki skan
    scan_violations_dir()
    
    # Monitoring thread
    monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor_thread.start()
    
    # Bot polling
    bot.polling(none_stop=True)