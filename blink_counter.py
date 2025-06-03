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
# Yüz işaretleyici modelini yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Göz işaret noktaları için index'leri alma
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Video kaynağını açma
cap  = cv2.VideoCapture(0)

blink_count = 0
frame_counter = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti yap
    faces  = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Sol ve Sağ göz işaret noktalarını al
        leftEye  = shape[lStart: lEnd]
        rightEye = shape[rStart: rEnd]

        # göz kırpma oranı
        leftEAR = eye_aspect_ratio(leftEye)
        rightEar  = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEar ) / 2

        # Göz kapaklarınını çizdir
        leftEyeHull =cv2.convexHull(leftEye)
        rightEyeHull =cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1 , (0,255,0), 1)

        #Göz kırpma oranı belli bir eşikten küçük ise
        if  ear < 0.25:
            frame_counter += 1
        else:
            #Gözler birkaç çerçeve kapalı kaldıktan sonra açıldığında göz kırpma say
            if frame_counter >=3 :
                blink_count += 1
                frame_counter = 0
    
    # Göz kırpmayı çerçevede göster 
    cv2.putText(frame, f'Göz kırpma Sayısı : {blink_count}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Çerçeveyi göster
    cv2.imshow('Göz kırpma algılama', frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
print(f'Toplamda {blink_count} kere göz kırptınız.')

