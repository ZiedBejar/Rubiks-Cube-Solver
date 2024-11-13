import numpy as np
import cv2

# Fonction appelée lorsque la valeur de la trackbar change
def update_k(x):
    pass

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Création d'une fenêtre pour afficher les résultats
cv2.namedWindow('dst')

# Création d'une trackbar pour ajuster la valeur de k (échelle de 1 à 1000)
cv2.createTrackbar('k', 'dst', 1, 1000, update_k)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Application du filtre bilatéral pour lisser l'image tout en préservant les bords
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = np.float32(blur)

    # Détection des coins avec Harris
    dst = cv2.cornerHarris(gray, 2, 7, 0.06)

    # Dilatation des coins pour les rendre plus visibles
    dst = cv2.dilate(dst, None)

    # Récupération de la position du trackbar pour ajuster la valeur de k
    trackbar_val = cv2.getTrackbarPos('k', 'dst')
    
    # Conversion de la valeur de la trackbar en un k entre 0.0001 et 0.1
    k = 0.0001 + (trackbar_val / 1000) * (0.1 - 0.0001)

    # Appliquer le seuil avec la valeur de k
    threshold = k * dst.max()
    interest_points = dst > threshold

    # Assurez-vous que le seuil n'est pas trop bas pour éviter de surligner toute l'image
    if np.sum(interest_points) == 0:
        interest_points = dst > 0.0001 * dst.max()  # Valeur de seuil de secours

    # Affichage des points d'intérêt détectés avec le seuil dynamique
    frame[interest_points] = [0, 0, 255]

    # Compter le nombre de points d'intérêt détectés
    num_points = np.sum(interest_points)

    # Afficher le nombre de points sur l'image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Points d\'interet: {num_points}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Affichage de la fenêtre
    cv2.imshow('dst', frame)

    # Sortie de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
