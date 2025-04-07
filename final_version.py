
import os
import threading# Pentru procesare paralelă
import cv2
from deepface import DeepFace
import time
import mediapipe as mp
from queue import Queue# Pentru gestionarea datelor între thread-uri
import numpy as np
import matplotlib.pyplot as plt

# Metoda este folosită pentru a calcula similaritatea între două fețe
# Matricea A este produsul exterior între vectorii caracteristici ai fețelor
# Valoarea proprie dominantă oferă o măsură a similarității
def power_method(A, tol=1e-6, maxiter=100):
    """
    Optimized power method implementation.
    """
    n = A.shape[0]
    y = np.ones(n) / np.sqrt(n)
    i = 0
    e = 1
    lambda_old = 0

    while e > tol and i < maxiter:
        z = np.dot(A, y)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-10:
            return 0, np.zeros_like(y), False
        z /= z_norm
        lambda_new = np.dot(y, np.dot(A, y))
        e = abs(lambda_new - lambda_old)
        lambda_old = lambda_new
        y = z
        i += 1

    return lambda_old, y, i < maxiter

# Această funcție:
# Procesează imaginea pentru a detecta mâinile
# Desenează punctele de reper pe mâini
# Numără degetele ridicate bazându-se pe poziția vârfurilor degetelor
# Afișează numărul total de degete pe imagine
def count_fingers(image, hands, mp_hands, mp_drawing):
    """
    Detectează mâinile în imagine, numără degetele ridicate și afișează rezultatul.
    """
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Preprocesează imaginea pentru detecția mâinilor

    if results.multi_hand_landmarks:  # Verifică dacă au fost detectate mâini
        total_fingers = 0  # Contor pentru degetele ridicate
        for hand_landmarks in results.multi_hand_landmarks:  # Iterează prin toate mâinile detectate
            fingers = 0  # Contor local pentru degete
            # Desenează punctele și conexiunile dintre ele
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verifică fiecare deget pe baza pozițiilor punctelor de reper
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_IP].y:  # Degetul mare
                fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_PIP].y:  # Degetul arătător
                fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:  # Degetul mijlociu
                fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.RING_FINGER_PIP].y:  # Degetul inelar
                fingers += 1
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
                mp_hands.HandLandmark.PINKY_PIP].y:  # Degetul mic
                fingers += 1

            total_fingers += fingers  # Actualizează totalul degetelor ridicate

        # Afișează numărul total de degete pe imagine
        cv2.putText(image, f'Fingers: {total_fingers}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image  # Returnează imaginea procesată

# Funcția pentru afișarea rezultatelor metodei puterii
def plot_power_method_analysis(similarity_log):
    """
    Generează și afișează grafice pe baza rezultatelor metodei puterii.
    """
    plt.figure(figsize=(15, 10))

    # Grafic pentru similaritățile calculate
    names = list(similarity_log.keys())
    for idx, name in enumerate(names):
        plt.subplot(2, 2, idx + 1)  # Creează subplot
        plt.title(f'Similarity with {name}')  # Titlul graficului
        plt.xlabel('Attempt')  # Eticheta axei X
        plt.ylabel('Similarity')  # Eticheta axei Y
        plt.grid(True)  # Activează grila
        plt.plot(range(len(similarity_log[name])), similarity_log[name], 'b-', label=f'{name}')
        plt.legend()  # Afișează legenda

    # Grafic histogramă pentru valorile finale
    plt.subplot(2, 2, len(names) + 1)
    final_scores = [scores[-1] for scores in similarity_log.values()]
    plt.bar(names, final_scores, color='orange', alpha=0.7)  # Creează histogramă
    plt.title('Final Similarities')
    plt.xlabel('Person')
    plt.ylabel('Similarity')

    plt.tight_layout()  # Aranjează subgraficele
    plt.show()  # Afișează graficul


# Clasa pentru detecția facială pe un thread separat
class FaceDetectionThread:
    def __init__(self, reference_images):
        """
        Inițializează obiectul pentru detecția facială.
        """
        self.reference_images = reference_images  # Imaginile de referință pentru recunoaștere
        self.reference_features = {}  # Caracteristicile fețelor de referință
        self.frame_queue = Queue(maxsize=2)  # Coada pentru cadrele de procesat
        self.result_queue = Queue(maxsize=2)  # Coada pentru rezultate
        self.running = False  # Indicator pentru rularea thread-ului
        self.thread = None  # Thread-ul asociat
        self.current_match = None  # Ultimul rezultat de potrivire
        self.match_timestamp = 0  # Momentul ultimei potriviri
        self.match_duration = 3  # Durata cât se păstrează potrivirea
        self.similarity_threshold = 0.4  # Pragul de similaritate pentru recunoaștere
        self.similarity_log = {}  # Log pentru similarități
        self.batch_size = 5  # Numărul de cadre procesate simultan
        self._extract_reference_features()  # Extrage caracteristicile fețelor de referință

    def _extract_reference_features(self):
        print("Extracting reference features...")
        for name, ref_img in self.reference_images.items():
            try:
                result = DeepFace.represent(
                    ref_img,
                    model_name="VGG-Face",#MODELUL UTILIZAT
                    enforce_detection=True,
                    detector_backend="opencv"
                )
                self.reference_features[name] = np.array(result[0]['embedding'])  # Salvează vectorul de caracteristici
                self.similarity_log[name] = []  # Inițializează log-ul pentru această persoană
                print(f"Extracted features for {name}")
            except Exception as e:
                print(f"Error extracting features for {name}: {str(e)}")

    def start(self):
        """
        Pornește thread-ul pentru procesarea cadrelor.
        """
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)  # Creează un thread nou
        self.thread.daemon = True  # Thread-ul se oprește odată cu procesul principal
        self.thread.start()  # Pornește thread-ul

    def stop(self):
        """
        Oprește thread-ul pentru procesarea cadrelor.
        """
        self.running = False
        if self.thread:
            self.thread.join()  # Așteaptă terminarea thread-ului

    def _process_frames(self):
        """
        Procesează cadrele din coada de cadre.
        """
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()  # Preia un cadru din coadă
                face_match = False  # Indicator pentru potrivire facială
                matched_person = ""  # Numele persoanei potrivite
                similarity_score = 0  # Scorul de similaritate

                try:
                    # Extrage caracteristicile feței detectate
                    detected_features = DeepFace.represent(
                        frame,
                        model_name="VGG-Face",
                        enforce_detection=True,
                        detector_backend="opencv"
                    )[0]['embedding']
                    detected_features = np.array(detected_features)

                    # Pre-calcularea normelor
                    detected_norm = np.linalg.norm(detected_features)
                    max_similarity = 0

                    for name, ref_features in self.reference_features.items():
                        ref_norm = np.linalg.norm(ref_features)

                        # Creează matricea de similaritate
                        similarity_matrix = np.outer(detected_features, ref_features)

                        # Aplică metoda puterii
                        lambda_val, _, success = power_method(similarity_matrix)

                        # Calculul valorilor proprii folosind NumPy la fiecare 5 cadre
                        if len(self.similarity_log[name]) % 5 == 0:
                            eigenvalues = np.linalg.eigvals(similarity_matrix)
                            lambda_np = np.max(np.abs(eigenvalues))
                            print(f"NumPy Eig - Dominant Eigenvalue: {lambda_np:.4f}")

                        if success:
                            # Calculează similaritatea
                            similarity = abs(lambda_val) / (detected_norm * ref_norm)
                            print(f"Method Power - Dominant Eigenvalue: {lambda_val:.4f}")
                            print(f"Similarity with {name}: {similarity:.4f}")

                            self.similarity_log[name].append(similarity)  # Salvează scorul

                            if similarity > max_similarity and similarity > self.similarity_threshold:
                                max_similarity = similarity
                                face_match = True
                                matched_person = name
                                similarity_score = similarity

                    if face_match:
                        self.current_match = (matched_person, similarity_score)
                        self.match_timestamp = time.time()  # Actualizează timestamp-ul potrivirii
                    elif self.current_match and (time.time() - self.match_timestamp) > self.match_duration:
                        self.current_match = None  # Resetează potrivirea dacă timpul a expirat

                except Exception as e:
                    print(f"Error in face detection: {str(e)}")

                # Actualizează coada de rezultate
                while not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.put(self.current_match)

    def process_frame(self, frame):
        while not self.frame_queue.empty():
            self.frame_queue.get()
        self.frame_queue.put(frame.copy())

    def get_result(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None


def initialize_camera():
    """
    Configurează și inițializează camera pentru captură video.

    Returnează:
        cap: Obiectul cv2.VideoCapture care gestionează fluxul video de la cameră.
    """
    # Deschide camera utilizând OpenCV.
    # 0 indică faptul că folosim camera implicită conectată la sistem.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Setează lățimea cadrului capturat la 640 de pixeli.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Setează înălțimea cadrului capturat la 480 de pixeli.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Setează numărul de cadre pe secundă (FPS) la 30 pentru o captură fluidă.
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Setează dimensiunea bufferului pentru captură la 1 cadru,
    # reducând întârzierea (latența) în citirea cadrelor de la cameră.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Returnează obiectul cv2.VideoCapture pentru a fi utilizat în fluxul video.
    return cap


def load_reference_images_from_folder(folder_path):
    """
    Încarcă imagini de referință dintr-un folder specificat și le stochează într-un dicționar.

    Parametri:
        folder_path (str): Calea către folderul care conține imaginile de referință.

    Returnează:
        dict: Un dicționar care mapează numele fișierelor (fără extensie) la imaginile încărcate.
    """
    # Inițializează un dicționar gol pentru a stoca imaginile de referință.
    reference_images = {}

    # Parcurge toate fișierele din folderul specificat.
    for filename in os.listdir(folder_path):
        # Verifică dacă fișierul are o extensie validă (PNG, JPG sau JPEG).
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # Încarcă imaginea utilizând OpenCV.
            img = cv2.imread(os.path.join(folder_path, filename))

            # Dacă imaginea este validă (nu este None), o adaugă în dicționar.
            # Cheia este numele fișierului fără extensie.
            if img is not None:
                reference_images[filename.split('.')[0]] = img

    # Returnează dicționarul care conține imaginile de referință.
    return reference_images


def main():
    # Inițializează camera
    cap = initialize_camera()

    # Încarcă imaginile de referință din folderul "faces"
    reference_images = load_reference_images_from_folder("faces")

    # Creează și pornește un thread pentru detectarea facială
    face_detector = FaceDetectionThread(reference_images)
    face_detector.start()

    frame_count = 0  # Numără cadrele procesate
    last_match = None  # Ultimul rezultat al detectării faciale
    mode = "face"  # Mod inițial (recunoaștere facială)
    is_authenticated = False  # Status de autentificare
    auth_person = None  # Persoana autenticată

    # Inițializează MediaPipe pentru numărarea degete
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,  # Confidența minimă pentru detectare
        min_tracking_confidence=0.5,  # Confidența minimă pentru urmărire
        max_num_hands=2  # Numărul maxim de mâini detectate
    )

    while cap.isOpened():  # Continuă dacă camera este deschisă
        ret, frame = cap.read()  # Citește un cadru de la cameră
        if not ret:  # Verifică dacă cadrul a fost citit cu succes
            print("Failed to grab frame")
            continue

        frame = cv2.flip(frame, 1)  # Întoarce cadrul orizontal (efect de oglindă)
        frame_count += 1  # Crește numărul de cadre procesate

        # Procesare detectare facială la fiecare 15 cadre
        if frame_count % 15 == 0:
            face_detector.process_frame(frame)

        # Obține rezultatul detectării faciale
        result = face_detector.get_result()
        if result is not None:  # Dacă există un rezultat
            last_match = result  # Actualizează ultimul rezultat
            matched_person, similarity = last_match
            if similarity > face_detector.similarity_threshold:  # Verifică pragul de similaritate
                is_authenticated = True  # Autentificare validă
                auth_person = matched_person
        elif last_match is None:  # Dacă nu există rezultate anterioare
            is_authenticated = False  # Nu este autentificat
            auth_person = None

        # Afișează statusul de autentificare pe ecran
        if is_authenticated:
            text = f"Autentificat: {auth_person} ({similarity:.2f})"
            cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Permite schimbarea modului doar dacă este autentificat
            if mode == "fingers":
                frame = count_fingers(frame, hands, mp_hands, mp_drawing)  # Procesare numărare degete
        else:
            cv2.putText(frame, "Unknown", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mode = "face"  # Forțează mod "face" dacă nu este autentificat

        # Afișează modul curent pe ecran
        mode_text = "Mode: Face Recognition" if mode == "face" else "Mode: Finger Counter"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Afișează fereastra cu cadrul procesat
        cv2.imshow('Face Recognition and Finger Counter', frame)

        key = cv2.waitKey(1) & 0xFF  # Așteaptă apăsarea unei taste
        if key == ord('q'):  # Dacă se apasă 'q', ieșim din program
            cap.release()  # Eliberăm camera
            cv2.destroyAllWindows()  # Închidem toate ferestrele
            plt.close('all')  # Închidem toate graficele Matplotlib
            plot_power_method_analysis(face_detector.similarity_log)  # Afișăm analiza similarității
        elif key == ord('m') and is_authenticated:  # Schimbă modul doar dacă este autentificat
            mode = "fingers" if mode == "face" else "face"

    face_detector.stop()  # Oprește thread-ul de detectare facială
    cap.release()  # Eliberăm camera
    cv2.destroyAllWindows()  # Închidem toate ferestrele


if __name__ == "__main__":
    main()  # Punctul de intrare în program
