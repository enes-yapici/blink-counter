import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

def eye_aspect_ratio(eye):
    # Göz açısını hesapla (EAR)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# EAR eşik değeri, göz kapalıysa daha düşük olur
EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 3

# sayaçlar
counter = 0
total_blinks = 0

# dlib'in yüz algılayıcısı ve göz landmark modeli
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# göz indeksleri (sol ve sağ)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# kamera başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # göz kapalıysa sayacı artır
        if ear < EYE_AR_THRESH:
            counter += 1
        else:
            if counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            counter = 0

        # görsel ekle
        cv2.putText(frame, f"Göz kırpma sayısı: {total_blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Goz Kirpma Sayaci", frame)
    if cv2.waitKey(1) == 27:  # ESC tuşu
        break

cap.release()
cv2.destroyAllWindows()