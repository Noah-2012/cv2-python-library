import cv2
import mediapipe as mp
import numpy as np

# Initialisierungen
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Konstanten
MAX_DISTANCE = 500
LINE_LENGTH = 200  # Länge der Blickrichtunglinie

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Gesichtserkennung
    face_results = face_mesh.process(rgb_frame)
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Gesichtsmesh zeichnen
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Wichtige Punkte identifizieren
            nose_tip = face_landmarks.landmark[4]  # Nasenspitze
            forehead = face_landmarks.landmark[10]  # Stirnmitte
            chin = face_landmarks.landmark[152]  # Kinn
            
            # Mittelpunkt des Gesichts berechnen
            face_center_x = int((nose_tip.x + forehead.x + chin.x) / 3 * width)
            face_center_y = int((nose_tip.y + forehead.y + chin.y) / 3 * height)
            
            # Blickrichtung (nach vorne, vereinfacht)
            # Wir verwenden die Nasenrichtung als Proxy für die Blickrichtung
            direction_x = int(nose_tip.x * width + LINE_LENGTH * np.sin(nose_tip.z * 10))
            direction_y = int(nose_tip.y * height - LINE_LENGTH * np.cos(nose_tip.z * 10))
            
            # Blickrichtunglinie zeichnen
            cv2.line(frame, 
                    (int(nose_tip.x * width), int(nose_tip.y * height)),
                    (direction_x, direction_y),
                    (0, 255, 0), 2)
            
            # Gesichtszentrum markieren
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
    
    # Hand-Erkennung (wie zuvor)
    hand_results = hands.process(rgb_frame)
    hand_points = []
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Fingerpositionen
            finger_pos = {}
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                if landmark_id in [4, 8, 12, 16, 20]:  # Nur Fingerspitzen
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    finger_pos[landmark_id] = (x, y)
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
            
            # Zeigefinger und Daumen
            if 8 in finger_pos and 4 in finger_pos:
                idx_pos = finger_pos[8]
                thb_pos = finger_pos[4]
                hand_points.append((idx_pos, thb_pos))
                
                # Linie zwischen Zeigefinger und Daumen
                cv2.line(frame, idx_pos, thb_pos, (255, 255, 255), 3)
    
    # Verbindung zwischen Händen (wenn zwei Hände erkannt)
    if len(hand_points) == 2:
        (idx1, thb1), (idx2, thb2) = hand_points
        
        # Mittelpunkte berechnen
        mid1 = ((idx1[0] + thb1[0]) // 2, (idx1[1] + thb1[1]) // 2)
        mid2 = ((idx2[0] + thb2[0]) // 2, (idx2[1] + thb2[1]) // 2)
        
        # Farbige Verbindungslinie
        distance = np.sqrt((mid2[0] - mid1[0])**2 + (mid2[1] - mid1[1])**2)
        ratio = min(distance / MAX_DISTANCE, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))
        cv2.line(frame, mid1, mid2, color, 3)
    
    cv2.imshow('Gesichts- und Handtracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
