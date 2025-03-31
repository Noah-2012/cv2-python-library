import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hände-Modul initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Kamera initialisieren
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 500  # Maximale Distanz für Farbberechnung

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand-Erkennung
    results = hands.process(rgb_frame)
    hand_points = []  # Speichert nur Zeigefinger/Daumen-Paare
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Alle Fingerkuppen markieren (Landmarks 4, 8, 12, 16, 20)
            for landmark_id in [4, 8, 12, 16, 20]:
                landmark = hand_landmarks.landmark[landmark_id]
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)  # Weißer Punkt
            
            # Nur Zeigefinger (8) und Daumen (4) für die Linienverbindung speichern
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            hand_points.append((
                (int(index_finger.x * width), int(index_finger.y * height)),
                (int(thumb.x * width), int(thumb.y * height))
            ))  # Korrekte Schließung aller Klammern
    
    # Originale Linienlogik (nur für Zeigefinger/Daumen)
    if len(hand_points) == 2:
        (idx1, thb1), (idx2, thb2) = hand_points
        
        # Linien zwischen Zeigefinger/Daumen jeder Hand
        cv2.line(frame, idx1, thb1, (255, 255, 255), 3)
        cv2.line(frame, idx2, thb2, (255, 255, 255), 3)
        
        # Mittelpunkte berechnen
        mid1 = ((idx1[0] + thb1[0]) // 2, (idx1[1] + thb1[1]) // 2)
        mid2 = ((idx2[0] + thb2[0]) // 2, (idx2[1] + thb2[1]) // 2)
        
        # Farbige Mittellinie (Grün → Rot)
        distance = np.sqrt((mid2[0] - mid1[0])**2 + (mid2[1] - mid1[1])**2)
        ratio = min(distance / MAX_DISTANCE, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR
        cv2.line(frame, mid1, mid2, color, 3)
        
        # Optional: Distanz anzeigen
        cv2.putText(frame, f"{int(distance)}px", (mid1[0] + 10, mid1[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Hand Tracking mit allen Fingern', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
