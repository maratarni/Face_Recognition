# 👁️✋ Real-Time Face Recognition and Finger Counting System

## 📌 General Description
This project integrates two main functionalities into a single interactive application:

- **Real-time face recognition** for precise biometric authentication.  
- **Hand gesture–based finger detection and counting** for hands-free interaction.  

The system is optimized for high performance, using parallel processing and advanced mathematical analysis techniques.

---

## 🛠️ Technologies and Libraries Used

| Functionality                   | Library Used             |
|---------------------------------|---------------------------|
| Face recognition                | DeepFace                 |
| Video & image processing        | OpenCV (`cv2`)           |
| Numerical computation           | NumPy                    |
| Hand/finger detection           | MediaPipe                |
| Data visualization              | Matplotlib               |
| Parallel processing             | `threading`, `queue`     |

---

## 🚀 Detailed Functionalities

### ✅ Face Recognition
- Extracts facial features with DeepFace.  
- Computes similarity between faces using the **Power Method**.  
- Integrates an authentication system with a configurable threshold (default: `0.5`).  
- Efficient processing with threading and buffered video optimization.  

### ✋ Finger Counting
- Uses MediaPipe for hand joint detection.  
- Checks if each fingertip (**TIP**) is above its intermediate point (**IP**) to determine if a finger is raised.  
- Supports simultaneous detection of **two hands**.  
- The finger count is displayed in real time.  

---

## 📊 Special Features

- **Mode switching**: pressing the `m` key toggles between face recognition and finger counting modes.  
- **Performance analysis**: pressing `q` generates comparative charts and analyzes Power Method vs `np.linalg.eig()`.  
- **Continuous monitoring**: automatic logging of similarity scores and system performance.  
- **Live interface**: clear visual feedback with authentication status and graphical representation of hands and fingers.  

---

## ⚙️ Performance

| Module                 | Avg. processing time per frame |
|-------------------------|--------------------------------|
| Face recognition        | 200–500 ms                     |
| Finger counting         | 10–30 ms                       |

- **Estimated success rate**: `90%`  
- Automatic comparisons between numerical methods (Power vs Eig) every **5 frames**, without significant impact on overall performance.  

---

## 📐 Mathematical Considerations

- After extracting facial vectors, similarity is evaluated through **calculation of the dominant eigenvalue**.  
- A **manual implementation of the Power Method** is used, validated against NumPy’s `np.linalg.eig()`.  

---

## ✅ Conclusion

This innovative system works as a **digital gatekeeper**:

- It **recognizes you** based on your facial features.  
- It allows you to **control the application** through simple hand gestures.  

It stands as a solid example of integrating computer vision and numerical methods into a modern, interactive, and scalable system.  

---
