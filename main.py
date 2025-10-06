import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict
import os
from datetime import datetime
import yaml
import threading
import time
from queue import Queue

class RailSafeMonitoring:
    def __init__(self, config_path='config.yaml'):
        """Konfiguratsiya faylidan sozlamalarni yuklash"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLO(self.config['model']['path'])
        self.target_classes = self.config['model']['target_classes']
        self.class_names = self.config['model']['class_names']
        
        self._create_directories()
        
        self.processors = []
        self.threads = []
        self.running = True
        self._initialize_cameras()
    
    def _create_directories(self):
        """Barcha kerakli papkalarni yaratish"""
        base_dirs = [
            'vehicle_data/normal/videos',
            'vehicle_data/normal/images',
            'vehicle_data/violations/videos',
            'vehicle_data/violations/images',
            'vehicle_data/reports',
            'vehicle_data/cropped'
        ]
        for dir_path in base_dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _initialize_cameras(self):
        """Barcha kameralarni ishga tushirish"""
        for cam_config in self.config['cameras']:
            if not cam_config.get('enabled', True):
                continue
            try:
                processor = CameraProcessor(
                    camera_id=cam_config['id'],
                    source=cam_config['source'],
                    polygon_file=cam_config['polygon_file'],
                    model_path=self.config['model']['path'],
                    config=self.config,
                    class_names=self.class_names,
                    target_classes=self.target_classes
                )
                self.processors.append(processor)
            except Exception as e:
                pass
    
    def run(self):
        """Asosiy monitoring tsikli"""
        if not self.processors:
            return
        
        try:
            for processor in self.processors:
                thread = threading.Thread(
                    target=self._camera_thread,
                    args=(processor,),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
            
            while self.running:
                frames_to_display = []
                
                for processor in self.processors:
                    frame = processor.get_latest_frame()
                    if frame is not None:
                        h, w = frame.shape[:2]
                        display_height = 500
                        display_width = int(w * display_height / h)
                        display_frame = cv2.resize(frame, (display_width, display_height))
                        frames_to_display.append((processor.camera_id, display_frame))
                
                if frames_to_display:
                    if len(frames_to_display) == 1:
                        combined = frames_to_display[0][1]
                    elif len(frames_to_display) == 2:
                        combined = np.hstack([frames_to_display[0][1], frames_to_display[1][1]])
                    elif len(frames_to_display) == 3:
                        top_row = np.hstack([frames_to_display[0][1], frames_to_display[1][1]])
                        pad_width = frames_to_display[0][1].shape[1]
                        bottom_row = np.hstack([frames_to_display[2][1], 
                                               np.zeros((display_height, pad_width, 3), dtype=np.uint8)])
                        combined = np.vstack([top_row, bottom_row])
                    else:
                        row1 = np.hstack([frames_to_display[0][1], frames_to_display[1][1]])
                        row2 = np.hstack([frames_to_display[2][1], frames_to_display[3][1]])
                        combined = np.vstack([row1, row2])
                    
                    cv2.imshow("RailSafe Monitoring - Barcha Kameralar", combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    self.running = False
                    break
                
                time.sleep(0.01)
        
        finally:
            self._cleanup()
    
    def _camera_thread(self, processor):
        """Har bir kamera uchun alohida thread"""
        while self.running:
            success = processor.process_frame()
            if not success:
                break
            
            time.sleep(0.005)
    
    def _cleanup(self):
        """Resurslarni tozalash"""
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        for processor in self.processors:
            processor.release()
        
        cv2.destroyAllWindows()


class CameraProcessor:
    def __init__(self, camera_id, source, polygon_file, model_path, config, class_names, target_classes):
        self.camera_id = camera_id
        self.camera_name = next((cam['name'] for cam in config['cameras'] if cam['id'] == camera_id), f"Kamera {camera_id}")
        self.source = source
        self.config = config
        self.class_names = class_names
        self.target_classes = target_classes
        
        self.model = YOLO(model_path)
        
        self.lock = threading.Lock()
        self.latest_frame = None
        
        # FPS hisoblash
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        self.fps_update_interval = 1.0
        
        # ADAPTIVE FRAME PROCESSING
        self.adaptive_mode = config['processing'].get('adaptive_mode', True)
        self.frame_skip_idle = config['processing'].get('frame_skip_idle', 2)
        self.frame_skip_active = config['processing'].get('frame_skip_active', 1)
        self.current_frame_skip = self.frame_skip_idle
        self.frame_counter = 0
        
        # Detection holati
        self.consecutive_empty_frames = 0
        self.empty_threshold = 5
        
        self.detected_count = 0
        self.entered_count = 0
        
        # Polygon holati
        self.polygon_state = "empty"
        self.max_time_in_polygon = 0
        
        # Polygonni yuklash
        with open(polygon_file, 'r') as f:
            polygon_data = json.load(f)
        self.polygon_points = np.array(polygon_data['annotations'][0]['segmentation'][0]).reshape(-1, 2)
        
        # Video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Kamera ochilmadi: {source}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Output video
        self.out = cv2.VideoWriter(
            f'vehicle_data/camera_{camera_id}_full_output.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            self.video_fps,
            (self.frame_width, self.frame_height)
        )
        
        # Tracking
        self.vehicle_tracking = defaultdict(lambda: {
            'start_time': None,
            'in_polygon': False,
            'total_time': 0,
            'class_id': None,
            'video_writer': None,
            'video_filename': None,
            'normal_image_saved': False,
            'violation_image_saved': False,
            'is_violation': False,
            'entered_polygon': False,
            'last_seen_time': 0
        })
        
        self.frame_count = 0
        self.process_count = 0
    
    def get_latest_frame(self):
        """Thread-safe ravishda oxirgi frameni olish"""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def _update_fps(self):
        """Real FPS ni hisoblash"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= self.fps_update_interval:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = current_time
    
    def _point_in_polygon(self, point, polygon):
        """Nuqta polygon ichidami tekshirish"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _update_polygon_state(self):
        """Polygon holatini yangilash"""
        vehicles_inside = 0
        max_time = 0
        
        for vehicle_id, data in self.vehicle_tracking.items():
            if data['in_polygon']:
                vehicles_inside += 1
                if data['total_time'] > max_time:
                    max_time = data['total_time']
        
        if vehicles_inside == 0:
            self.max_time_in_polygon = 0
            self.polygon_state = "empty"
        elif max_time >= self.config['thresholds']['violation']:
            self.max_time_in_polygon = max_time
            self.polygon_state = "violation"
        else:
            self.max_time_in_polygon = max_time
            self.polygon_state = "detected"
    
    def _draw_fancy_polygon(self, frame):
        """Chiroyli polygon chizish - dinamik rang bilan"""
        if self.polygon_state == "empty":
            color = (0, 255, 0)
            alpha = 0.2
            glow_color = (0, 200, 0)
            status_text = "XAVFSIZ - Bo'sh"
            status_color = (0, 255, 0)
        elif self.polygon_state == "detected":
            vehicle_count = sum(1 for data in self.vehicle_tracking.values() if data['in_polygon'])
            color = (0, 255, 255)
            alpha = 0.3
            glow_color = (0, 200, 200)
            status_text = f"KUZATILMOQDA - {vehicle_count} ta | {self.max_time_in_polygon:.1f}s"
            status_color = (0, 255, 255)
        else:
            vehicle_count = sum(1 for data in self.vehicle_tracking.values() if data['in_polygon'])
            color = (0, 0, 255)
            alpha = 0.4
            glow_color = (0, 0, 200)
            status_text = f"BUZILISH! {vehicle_count} ta | {self.max_time_in_polygon:.1f}s"
            status_color = (0, 0, 255)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon_points.astype(np.int32)], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.polylines(frame, [self.polygon_points.astype(np.int32)], True, glow_color, 8, cv2.LINE_AA)
        cv2.polylines(frame, [self.polygon_points.astype(np.int32)], True, color, 4, cv2.LINE_AA)
        
        for point in self.polygon_points:
            px, py = int(point[0]), int(point[1])
            cv2.circle(frame, (px, py), 8, glow_color, -1)
            cv2.circle(frame, (px, py), 6, color, -1)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)
        
        center_x = int(np.mean(self.polygon_points[:, 0]))
        center_y = int(np.mean(self.polygon_points[:, 1]))
        
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        badge_x1 = center_x - text_size[0] // 2 - 15
        badge_y1 = center_y - 25
        badge_x2 = center_x + text_size[0] // 2 + 15
        badge_y2 = center_y + 15
        
        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
        cv2.addWeighted(badge_overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), status_color, 2)
        
        cv2.putText(frame, status_text, 
                   (center_x - text_size[0] // 2, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        
        return frame
    
    def _get_box_color(self, time_in_polygon):
        """Vaqtga qarab rang berish"""
        thresholds = self.config['thresholds']
        if time_in_polygon < thresholds['warning']:
            return (255, 0, 0)
        elif time_in_polygon < thresholds['violation']:
            return (0, 255, 255)
        else:
            return (0, 0, 255)
    
    def _get_status_text(self, time_in_polygon):
        """Status matni"""
        thresholds = self.config['thresholds']
        if time_in_polygon < thresholds['warning']:
            return "NORMAL", (255, 0, 0)
        elif time_in_polygon < thresholds['violation']:
            return "OGOHLANTIRISH!", (0, 255, 255)
        else:
            return "QOIDA BUZILISHI!", (0, 0, 255)
    
    def _save_vehicle_image(self, frame, track_id, x1, y1, x2, y2, class_id, is_violation=False):
        """Avtomobil rasmini va box koordinatalarini saqlash"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_violation:
            folder = 'vehicle_data/violations/images'
            prefix = 'VIOLATION'
        else:
            folder = 'vehicle_data/normal/images'
            prefix = 'NORMAL'
        
        base_filename = f"cam{self.camera_id}_{prefix}_ID{track_id}_{timestamp}"
        image_filename = f"{folder}/{base_filename}.jpg"
        txt_filename = f"{folder}/{base_filename}.txt"
        
        with open(txt_filename, 'w') as f:
            f.write(f"# Kamera: {self.camera_id}\n")
            f.write(f"# Track ID: {track_id}\n")
            f.write(f"# Class: {self.class_names[class_id]}\n")
            f.write(f"# Status: {'VIOLATION' if is_violation else 'NORMAL'}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Box coordinates (x1, y1, x2, y2):\n")
            f.write(f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n")
            f.write(f"# Image size: {frame.shape[1]}x{frame.shape[0]}\n")
        
        clean_frame = frame.copy()
        
        color = (0, 0, 255) if is_violation else (255, 0, 0)
        cv2.rectangle(clean_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        info_text = f"CAM{self.camera_id} ID:{track_id} {self.class_names[class_id]}"
        cv2.putText(clean_frame, info_text, (int(x1), int(y1)-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        status = "QOIDA BUZILISHI!" if is_violation else "Normal"
        cv2.putText(clean_frame, status, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite(image_filename, clean_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config['image'].get('quality', 95)])
        return image_filename, txt_filename
    
    def process_frame(self):
        """Bir frameni ishlash"""
        success, frame = self.cap.read()
        if not success:
            return False
        
        self.frame_count += 1
        self.frame_counter += 1
        
        if self.adaptive_mode:
            process_this_frame = self.frame_counter % self.current_frame_skip == 0
        else:
            process_this_frame = self.frame_counter % self.config['processing'].get('process_every_n_frames', 1) == 0
        
        results = None
        current_time = self.frame_count / self.video_fps
        
        if process_this_frame:
            self._update_fps()
            
            self.process_count += 1
            
            results = self.model.track(frame, persist=True, classes=self.target_classes, conf=0.25, imgsz=640)
            
            detected_objects = len(results[0].boxes) if results[0].boxes is not None else 0
            self.detected_count += detected_objects
            
            if self.adaptive_mode:
                if detected_objects == 0:
                    self.consecutive_empty_frames += 1
                    if self.consecutive_empty_frames >= self.empty_threshold:
                        self.current_frame_skip = self.frame_skip_idle
                else:
                    self.consecutive_empty_frames = 0
                    self.current_frame_skip = self.frame_skip_active
            
            current_frame_ids = set()
            
            timeout_seconds = 3
            for vehicle_id in list(self.vehicle_tracking.keys()):
                if self.vehicle_tracking[vehicle_id]['entered_polygon']:
                    last_seen_time = self.vehicle_tracking[vehicle_id].get('last_seen_time', current_time)
                    if current_time - last_seen_time > timeout_seconds:
                        if vehicle_id not in current_frame_ids:
                            if self.vehicle_tracking[vehicle_id]['video_writer']:
                                self.vehicle_tracking[vehicle_id]['video_writer'].release()
                            del self.vehicle_tracking[vehicle_id]
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if hasattr(box, 'id') and box.id is not None:
                        current_frame_ids.add(int(box.id[0]))
            
            for vehicle_id in list(self.vehicle_tracking.keys()):
                if self.vehicle_tracking[vehicle_id]['in_polygon']:
                    if vehicle_id not in current_frame_ids:
                        self.vehicle_tracking[vehicle_id]['in_polygon'] = False
                        total_time = self.vehicle_tracking[vehicle_id]['total_time']
                        
                        if self.vehicle_tracking[vehicle_id]['video_writer']:
                            self.vehicle_tracking[vehicle_id]['video_writer'].release()
                            self.vehicle_tracking[vehicle_id]['video_writer'] = None
                            
                            old_video = self.vehicle_tracking[vehicle_id]['video_filename']
                            
                            if self.vehicle_tracking[vehicle_id]['is_violation']:
                                new_video = old_video.replace('/normal/videos/', '/violations/videos/')
                                os.rename(old_video, new_video)
                                self.vehicle_tracking[vehicle_id]['video_filename'] = new_video
        
        self._update_polygon_state()
        
        frame = self._draw_fancy_polygon(frame)
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    self.vehicle_tracking[track_id]['class_id'] = class_id
                    
                    self.vehicle_tracking[track_id]['last_seen_time'] = current_time
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)
                    
                    is_inside = self._point_in_polygon((center_x, center_y), self.polygon_points)
                    
                    if is_inside:
                        if not self.vehicle_tracking[track_id]['in_polygon']:
                            self.vehicle_tracking[track_id]['start_time'] = current_time
                            self.vehicle_tracking[track_id]['in_polygon'] = True
                            self.vehicle_tracking[track_id]['entered_polygon'] = True
                            self.entered_count += 1
                            
                            if self.config['image'].get('save_on_enter', True) and not self.vehicle_tracking[track_id]['normal_image_saved']:
                                self._save_vehicle_image(frame, track_id, x1, y1, x2, y2, class_id, is_violation=False)
                                self.vehicle_tracking[track_id]['normal_image_saved'] = True
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            video_filename = f"vehicle_data/normal/videos/cam{self.camera_id}_ID{track_id}_{timestamp}.avi"
                            
                            self.vehicle_tracking[track_id]['video_writer'] = cv2.VideoWriter(
                                video_filename,
                                cv2.VideoWriter_fourcc(*self.config['video']['codec']),
                                self.video_fps,
                                (self.frame_width, self.frame_height)
                            )
                            self.vehicle_tracking[track_id]['video_filename'] = video_filename
                        
                        time_in_polygon = current_time - self.vehicle_tracking[track_id]['start_time']
                        self.vehicle_tracking[track_id]['total_time'] = time_in_polygon
                        
                        if time_in_polygon >= self.config['thresholds']['violation']:
                            if not self.vehicle_tracking[track_id]['is_violation']:
                                self.vehicle_tracking[track_id]['is_violation'] = True
                            
                            if self.config['image'].get('save_on_violation', True) and not self.vehicle_tracking[track_id]['violation_image_saved']:
                                self._save_vehicle_image(frame, track_id, x1, y1, x2, y2, class_id, is_violation=True)
                                self.vehicle_tracking[track_id]['violation_image_saved'] = True
                        
                        box_color = self._get_box_color(time_in_polygon)
                        status_text, status_color = self._get_status_text(time_in_polygon)
                        
                        if self.vehicle_tracking[track_id]['video_writer']:
                            record_frame = frame.copy()
                            
                            cv2.rectangle(record_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                            
                            info_text = f"CAM{self.camera_id} ID:{track_id} {self.class_names[class_id]}"
                            cv2.putText(record_frame, info_text, (int(x1), int(y1)-60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                            
                            time_text = f"Vaqt: {time_in_polygon:.1f}s"
                            cv2.putText(record_frame, time_text, (int(x1), int(y1)-40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                            
                            cv2.putText(record_frame, status_text, (int(x1), int(y1)-20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                            
                            self.vehicle_tracking[track_id]['video_writer'].write(record_frame)
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                        cv2.putText(frame, f"ID:{track_id} {self.class_names[class_id]}", 
                                  (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        cv2.putText(frame, f"Vaqt: {time_in_polygon:.1f}s", 
                                  (int(x1), int(y1)-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        cv2.putText(frame, status_text, 
                                  (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    
                    else:
                        if self.vehicle_tracking[track_id]['in_polygon']:
                            self.vehicle_tracking[track_id]['in_polygon'] = False
                            total_time = self.vehicle_tracking[track_id]['total_time']
                            
                            if self.vehicle_tracking[track_id]['video_writer']:
                                self.vehicle_tracking[track_id]['video_writer'].release()
                                self.vehicle_tracking[track_id]['video_writer'] = None
                                
                                old_video = self.vehicle_tracking[track_id]['video_filename']
                                
                                if self.vehicle_tracking[track_id]['is_violation']:
                                    new_video = old_video.replace('/normal/videos/', '/violations/videos/')
                                    os.rename(old_video, new_video)
                                    self.vehicle_tracking[track_id]['video_filename'] = new_video
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        if self.vehicle_tracking[track_id]['entered_polygon']:
                            cv2.putText(frame, f"ID:{track_id} Tashqarida", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"{self.camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.adaptive_mode:
            mode_text = f"Mode: {'FAST' if self.current_frame_skip == 1 else 'IDLE'} (1/{self.current_frame_skip})"
            mode_color = (0, 255, 0) if self.current_frame_skip == 1 else (100, 100, 100)
            cv2.putText(frame, mode_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        status_x = frame.shape[1] - 250
        status_y = 30
        
        if self.polygon_state == "empty":
            status_text = "BO'SH"
            indicator_color = (0, 255, 0)
        elif self.polygon_state == "detected":
            status_text = "ANIQLANDI"
            indicator_color = (0, 255, 255)
        else:
            status_text = "BUZILISH!"
            indicator_color = (0, 0, 255)
        
        cv2.rectangle(frame, (status_x - 10, status_y - 25), 
                     (status_x + 240, status_y + 15), (0, 0, 0), -1)
        cv2.rectangle(frame, (status_x - 10, status_y - 25), 
                     (status_x + 240, status_y + 15), indicator_color, 2)
        
        cv2.putText(frame, f"Polygon: {status_text}", 
                   (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
        
        self._draw_legend(frame)
        
        # Output video yozish - barcha annotatsiyalardan keyin
        self.out.write(frame)
        
        with self.lock:
            self.latest_frame = frame.copy()
        
        return True
    
    def _draw_legend(self, frame):
        """Legendani chizish"""
        legend_y = frame.shape[0] - 140
        legend_x = 10
        
        cv2.rectangle(frame, (legend_x - 5, legend_y - 30), 
                     (legend_x + 480, legend_y + 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 5, legend_y - 30), 
                     (legend_x + 480, legend_y + 120), (255, 255, 255), 2)
        
        cv2.putText(frame, "RANG KODLARI:", (legend_x, legend_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Yashil Polygon: Bo'sh, xavfsiz", (legend_x, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Sariq Polygon: Avtomobil aniqlandi (0-{self.config['thresholds']['violation']}s)", (legend_x, legend_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Qizil Polygon: BUZILISH! ({self.config['thresholds']['violation']}s+)", (legend_x, legend_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Yashil Box: Tashqarida", (legend_x, legend_y + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def get_statistics(self):
        """Statistika"""
        normal_count = 0
        violation_count = 0
        
        for vehicle_id, data in self.vehicle_tracking.items():
            if data['entered_polygon']:
                if data['is_violation']:
                    violation_count += 1
                else:
                    normal_count += 1
        
        return normal_count, violation_count
    
    def release(self):
        """Resurslarni tozalash"""
        for vehicle_id, data in self.vehicle_tracking.items():
            if data['video_writer']:
                data['video_writer'].release()
        
        self.cap.release()
        self.out.release()


if __name__ == "__main__":
    try:
        system = RailSafeMonitoring('config.yaml')
        system.run()
    except FileNotFoundError:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()