import cv2
import mediapipe as mp
import numpy as np

# Initialisierung
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 500

# Objekt-Eigenschaften
objekt_pos = [300, 300]
objekt_radius = 30
objekt_farbe = (0, 255, 0)  # Grün
objekt_gegriffen = False

# Mal-Eigenschaften
mal_flaeche = np.zeros((480, 640, 3), dtype=np.uint8)  # Schwarze Leinwand
aktuelle_farbe = (255, 255, 255)  # Weiß
pinsel_groesse = 5
letzter_malpunkt = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand-Erkennung
    results = hands.process(rgb_frame)
    hand_points = []
    malmodus_aktiv = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Fingerpositionen
            finger_pos = {}
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                if landmark_id in [4, 8, 12, 16, 20]:  # Nur Fingerspitzen
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    finger_pos[landmark_id] = (x, y)
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
            
            # Zeigefinger (8) und Daumen (4)
            if 8 in finger_pos and 4 in finger_pos:
                idx_pos = finger_pos[8]
                thb_pos = finger_pos[4]
                hand_points.append((idx_pos, thb_pos))
                
                # Greif-Logik
                dist = np.sqrt((idx_pos[0]-thb_pos[0])**2 + (idx_pos[1]-thb_pos[1])**2)
                mittelpunkt = ((idx_pos[0]+thb_pos[0])//2, (idx_pos[1]+thb_pos[1])//2)
                
                dist_zum_objekt = np.sqrt((mittelpunkt[0]-objekt_pos[0])**2 + (mittelpunkt[1]-objekt_pos[1])**2)
                
                if dist < 40:  # Greif-Geste
                    if dist_zum_objekt < objekt_radius*2:
                        objekt_gegriffen = True
                else:
                    objekt_gegriffen = False
                    
                if objekt_gegriffen:
                    objekt_pos = [mittelpunkt[0], mittelpunkt[1]]
            
            # Mal-Logik (kleiner Finger ausgestreckt, andere Finger gebeugt)
            if 20 in finger_pos:  # Kleiner Finger
                kleiner_finger = finger_pos[20]
                mal_bedingung = True
                
                # Prüfe ob andere Finger gebeugt sind
                for fid in [8, 12, 16]:  # Zeige-, Mittel-, Ringfinger
                    if fid in finger_pos:
                        # Prüfe ob Finger gebeugt (y-Position höher als kleiner Finger)
                        if finger_pos[fid][1] < kleiner_finger[1] + 50:  # Toleranz
                            mal_bedingung = False
                            break
                
                if mal_bedingung:
                    malmodus_aktiv = True
                    aktueller_punkt = kleiner_finger
                    
                    if letzter_malpunkt:
                        cv2.line(mal_flaeche, letzter_malpunkt, aktueller_punkt, aktuelle_farbe, pinsel_groesse)
                    letzter_malpunkt = aktueller_punkt
                else:
                    letzter_malpunkt = None
    
    # Objekt zeichnen
    cv2.circle(frame, (objekt_pos[0], objekt_pos[1]), objekt_radius, objekt_farbe, -1)
    
    # Malfläche überlagern
    frame = cv2.add(frame, mal_flaeche)
    
    # Hand-Linien zeichnen
    if len(hand_points) == 2:
        (idx1, thb1), (idx2, thb2) = hand_points
        cv2.line(frame, idx1, thb1, (255, 255, 255), 3)
        cv2.line(frame, idx2, thb2, (255, 255, 255), 3)
        
        mid1 = ((idx1[0] + thb1[0]) // 2, (idx1[1] + thb1[1]) // 2)
        mid2 = ((idx2[0] + thb2[0]) // 2, (idx2[1] + thb2[1]) // 2)
        
        distance = np.sqrt((mid2[0] - mid1[0])**2 + (mid2[1] - mid1[1])**2)
        ratio = min(distance / MAX_DISTANCE, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))
        cv2.line(frame, mid1, mid2, color, 3)
    
    # Status anzeigen
    status_text = "Malmodus" if malmodus_aktiv else "Bewegungsmodus"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Greifen und Malen', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # Leinwand löschen
        mal_flaeche = np.zeros((height, width, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
