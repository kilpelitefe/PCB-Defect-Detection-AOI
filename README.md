#  PCB Defect Detection - Automated Optical Inspection (AOI)

> ** Project Status: Active Development **
> *This project is currently in the development phase. I am working on optimizing image processing algorithms and integrating specialized datasets.*

--

##  Overview
This project focuses on the automated detection of manufacturing defects in Printed Circuit Boards (PCBs). Using **Computer Vision** and **Image Processing**, the system identifies common production errors such as missing components, short circuits, or soldering issues to reduce human error in quality assurance.

--

##  Key Objectives
* **Quality Automation:** Minimizing manual inspection in PCB production.
* **Accuracy:** Implementing algorithms to detect micro-level defects.
* **Scalability:** Designing a system that can adapt to different PCB layouts.

--

##  Technical Stack
* **Language:** Python
* **Libraries:** OpenCV, NumPy, Matplotlib
* **Methodology:** Image Preprocessing, Thresholding, Feature Extraction
* **Hardware Context:** Integration with industrial camera systems (planned)

---

##  Current Roadmap & Progress
- [x] **Phase 1:** Research and Dataset selection (Professional PCB dataset integrated).
- [/] **Phase 2:** Image Preprocessing & Filtering (Currently optimizing noise reduction).
- [ ] **Phase 3:** Feature Detection & Defect Classification.
- [ ] **Phase 4:** Developing a User Interface (GUI) for real-time feedback.

---

##  Project Structure
```text
├── src/                # Python scripts for detection & processing
├── dataset/            # Samples and reference images
├── results/            # Output images with highlighted defects (Before/After)
└── README.md           # Project documentation
