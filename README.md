👁️✋ Sistem de Recunoaștere Facială și Numărare de Degete în Timp Real
📌 Descriere generală
Acest proiect integrează două funcționalități principale într-o singură aplicație interactivă:

Recunoaștere facială în timp real, pentru autentificare biometrică precisă.

Detectare și numărare de degete, bazată pe gesturi ale mâinii, pentru interacțiune hands-free.

Sistemul este optimizat pentru performanță ridicată, folosind procesare paralelă și tehnici de analiză matematică avansată.

🛠️ Tehnologii și biblioteci utilizate
Funcționalitate	Bibliotecă folosită
Recunoaștere facială	DeepFace
Procesare video & imagini	OpenCV (cv2)
Calcul numeric	NumPy
Detectarea mâinilor/degetelor	MediaPipe
Vizualizare date	Matplotlib
Procesare paralelă	threading, queue
🚀 Funcționalități detaliate
✅ Recunoaștere Facială
Extrage trăsături faciale cu DeepFace.

Calculează similaritatea între fețe folosind Metoda Puterii (Power Method).

Integrează un sistem de autentificare cu prag configurabil (default: 0.5).

Procesare eficientă prin threading și optimizare video cu buffer.

✋ Numărare de Degete
Utilizează MediaPipe pentru detecția articulațiilor mâinilor.

Verifică dacă vârful fiecărui deget (TIP) este deasupra punctului intermediar (IP) pentru a determina dacă degetul este ridicat.

Suportă detectarea simultană a două mâini.

Numărul de degete este afișat în timp real.

📊 Caracteristici speciale
Comutare moduri: apăsarea tastei m permite trecerea între modul de recunoaștere facială și numărare de degete.

Analiză performanță: tasta q generează grafice de analiză și compară metoda implementată cu np.linalg.eig() din NumPy.

Monitorizare continuă: log al scorurilor de similitudine și al performanței sistemului.

Interfață live: feedback vizual clar cu starea autentificării și reprezentarea grafică a mâinilor.

⚙️ Performanță
Modul	Timp mediu procesare per cadru
Recunoaștere facială	200–500 ms
Numărare de degete	10–30 ms
Rata de succes estimată: 90%

Comparații automate între metodele numerice la fiecare 5 cadre pentru evaluarea acurateței.

📐 Considerații matematice
După extragerea vectorilor faciali, similaritatea este evaluată prin calculul valorii proprii dominante.

Se folosește o implementare manuală a metodei puterii, validată comparativ cu metoda eig() din NumPy.

✅ Concluzie
Acest sistem inovator funcționează ca un portar digital:

Te recunoaște pe baza trăsăturilor faciale.

Îți permite să controlezi aplicația prin gesturi simple ale mâinii.
