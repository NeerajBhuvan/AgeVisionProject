"""Single source of truth for the AgeVision Final Project Report.

`build()` returns an ordered list of small instruction tuples that both the
docx and pdf renderers walk:

    ("heading_centered", text, {size, bold, italic, all_caps})
    ("section",         number, text, {level})
    ("para",            text)
    ("bullet",          [items...])
    ("numbered",        [items...])
    ("table",           headers, rows, caption, {col_widths})
    ("figure",          screenshot_label, caption)
    ("pagebreak",)
    ("mono",            text)
    ("right_text",      text, {bold})
    ("cover_para",      text, {bold, italic, size})
    ("logo",            path)

Keeping content here means the docx and pdf renderers stay thin and any
content edit takes effect in both formats simultaneously.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOGO_PATH = os.path.join(ROOT, "age_vision_logo.png")
ANNA_UNIVERSITY_LOGO_PATH = os.path.join(ROOT, "anna_university_logo.png")


# --------------------------------------------------------------------------- #
#                              Reference list                                  #
# --------------------------------------------------------------------------- #

REFERENCES = [
    "Deep Learning-Based Age Estimation and Gender Classification, arXiv "
    "preprint, 2025.",
    "Real Time Face Aging Progression Using GANs, International Journal of "
    "Creative Research Thoughts (IJCRT), 2025.",
    "NIST Face Analysis Technology Evaluation (FATE) Report 8525, "
    "National Institute of Standards and Technology, 2024.",
    "Dzenis, V. et al., \u201CEfficientNet-Based Apparent Age Prediction\u201D, "
    "Applied Computer Systems, 2024.",
    "Li, X. et al., \u201CClinical AI Facial Age Predictors\u201D, 2023.",
    "Wu, Q. et al., \u201CExplainable Conditional Adversarial Auto-Encoder "
    "for Age Estimation\u201D, Scientific Reports, 2023.",
    "Wang, K. et al., \u201CCycleGAN versus AttentionGAN for Face Aging\u201D, "
    "2022.",
    "Al-Quraishi, T. et al., \u201CCNN Age Estimation\u201D, University of "
    "Baghdad Engineering Journal, 2022.",
    "Fu, Y. et al., \u201CApparent Age Prediction: A Survey\u201D, 2022.",
    "Shukri, M. et al., \u201CDeep Learning for Face Age Estimation: A "
    "Survey\u201D, IJACSA, 2021.",
    "Abdollahi, B. et al., \u201CGAN-Based Face Age Progression for "
    "Healthcare Applications\u201D, Healthcare (MDPI), 2021.",
    "Mokady, R. et al., \u201CNull-text Inversion for Editing Real Images "
    "using Guided Diffusion Models\u201D, CVPR 2023 (FADING-diffusion basis).",
    "Yang, X. et al., \u201CHigh Resolution Face Age Editing (HRFAE)\u201D, "
    "ICPR 2020.",
    "Zhang, Z. et al., \u201CUTKFace: Aligned and Cropped Faces with Age, "
    "Gender and Ethnicity Labels\u201D, 23,000+ image dataset.",
    "Rothe, R., Timofte, R., Van Gool, L., \u201CIMDB-WIKI: 500K+ Face "
    "Images with Age and Gender Labels\u201D dataset.",
    "Ricanek, K., Tesafaye, T., \u201CMORPH: Longitudinal Face Aging Image "
    "Database\u201D dataset.",
    "Karras, T. et al., \u201CFFHQ: Flickr-Faces-HQ dataset\u201D, 70,000 "
    "high-quality face images.",
    "MiVOLO Documentation, https://github.com/WildChlamydia/MiVOLO "
    "[Accessed: April 2026].",
    "Alaluf, Y. et al., \u201CSAM: Only a Matter of Style \u2013 Age "
    "Transformation Using a Style-Based Regression Model\u201D, "
    "https://github.com/yuval-alaluf/SAM [Accessed: April 2026].",
    "Django REST Framework Documentation, https://www.django-rest-framework.org "
    "[Accessed: April 2026].",
    "Angular Documentation, https://angular.dev [Accessed: April 2026].",
    "HuggingFace ViT Face Expression Model, "
    "https://huggingface.co/trpakov/vit-face-expression [Accessed: April 2026].",
    "Modal Cloud GPU Documentation, https://modal.com/docs "
    "[Accessed: April 2026].",
    "HuggingFace Diffusers Library, https://huggingface.co/docs/diffusers "
    "[Accessed: April 2026].",
]


# --------------------------------------------------------------------------- #
#                                  HELPERS                                    #
# --------------------------------------------------------------------------- #

def _emit():
    """Generator-like helper returning an empty list to append tuples to."""
    return []


def _add(out, *cmd):
    out.append(cmd)


# --------------------------------------------------------------------------- #
#                              SECTION BUILDERS                               #
# --------------------------------------------------------------------------- #

def cover_page(out):
    _add(out, "cover_para",
         "AI BASED AGE PREDICTION AND AGE PROGRESSION SYSTEM "
         "USING FACE RECOGNITION & DEEP LEARNING",
         {"bold": True, "size": 12, "space_before": 12, "space_after": 6})
    _add(out, "cover_para", "By",
         {"size": 12, "space_before": 12, "space_after": 6})
    _add(out, "cover_para", "NEERAJ BHUVAN M",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para",
         "Roll No. 2435MCA0011    Reg. No. 67224100038",
         {"size": 12, "space_before": 6, "space_after": 18})
    _add(out, "cover_para", "A PROJECT REPORT",
         {"bold": True, "size": 12, "space_before": 18, "space_after": 6})
    _add(out, "cover_para", "Submitted to the",
         {"size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para",
         "FACULTY OF INFORMATION AND COMMUNICATION ENGINEERING",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para",
         "in partial fulfillment for the award of the degree",
         {"size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para", "Of",
         {"size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para", "MASTER OF COMPUTER APPLICATIONS",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 18})
    _add(out, "vspace", 54)
    _add(out, "logo", ANNA_UNIVERSITY_LOGO_PATH)
    _add(out, "vspace", 48)
    _add(out, "cover_para", "CENTRE FOR DISTANCE EDUCATION",
         {"bold": True, "size": 12, "space_before": 18, "space_after": 6})
    _add(out, "cover_para", "ANNA UNIVERSITY",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para", "CHENNAI 600 025",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 6})
    _add(out, "cover_para", "April 2026",
         {"bold": True, "size": 12, "space_before": 6, "space_after": 6})
    _add(out, "pagebreak")


def bonafide(out):
    _add(out, "heading_centered", "BONAFIDE CERTIFICATE",
         {"size": 14, "bold": True, "all_caps": True})
    _add(out, "para",
         "Certified that the project report titled \u201CAI Based Age "
         "Prediction and Age Progression System Using Face Recognition & "
         "Deep Learning\u201D is the Bonafide work of Mr. NEERAJ BHUVAN M "
         "who carried out the work under my supervision. Certified further "
         "that to the best of my knowledge the work reported herein does "
         "not form part of any other project report or dissertation on the "
         "basis of which a degree or award was conferred on an earlier "
         "occasion on this or any other candidate.")
    _add(out, "vspace", 90)
    _add(out, "para", "Date : ____________________")
    _add(out, "vspace", 36)
    _add(out, "table",
         ["Signature of Student", "Signature of Guide"],
         [
             ("Mr. NEERAJ BHUVAN M", "Dr. P. Geetha,"),
             ("Roll No: 2435MCA0011", "Associate Professor,"),
             ("Register No: 67224100038",
              "Department of Information Science and Technology,"),
             ("",
              "College of Engineering, Guindy, Anna University, "
              "Chennai \u2013 25"),
         ],
         None,
         {"borderless": True})
    _add(out, "pagebreak")


def viva_voce(out):
    _add(out, "heading_centered",
         "CERTIFICATE OF VIVA \u2013 VOCE EXAMINATION",
         {"size": 14, "bold": True, "all_caps": True})
    _add(out, "para",
         "This is to certify that Mr. NEERAJ BHUVAN M (Roll No: "
         "2435MCA0011, Register No: 67224100038) has been subjected to "
         "Viva \u2013 voce Examination on ___________ at the Study Centre: "
         "Centre for Distance Education, Anna University.")
    _add(out, "vspace", 48)
    _add(out, "table",
         ["Internal Examiner", "External Examiner"],
         [("Name :", "Name :"),
          ("Designation :", "Designation :"),
          ("Address :", "Address :")],
         None, {})
    _add(out, "vspace", 48)
    _add(out, "heading_centered", "COORDINATOR STUDY CENTER",
         {"size": 14, "bold": True})
    _add(out, "para", "Name :")
    _add(out, "para", "Designation :")
    _add(out, "para", "Address :")
    _add(out, "vspace", 60)
    _add(out, "pagebreak")


def acknowledgement(out):
    _add(out, "heading_centered", "ACKNOWLEDGEMENT",
         {"size": 14, "bold": True, "all_caps": True})
    paras = [
        "I take this opportunity to express my profound gratitude and deep "
        "regards to the Centre for Distance Education, Anna University, "
        "Chennai, for providing me the opportunity to undertake this Master "
        "of Computer Applications project as part of my fourth-semester "
        "curriculum. I am thankful to the entire faculty of the Faculty of "
        "Information and Communication Engineering for their continuous "
        "encouragement and scholarly guidance throughout the duration of "
        "this work.",
        "I express my sincere thanks to my project guide, Dr. P. Geetha, "
        "Associate Professor, Department of Information Science and "
        "Technology, College of Engineering, Guindy, Anna University, for "
        "her exemplary guidance, monitoring and constant encouragement "
        "throughout the course of this project. Her insights into deep "
        "learning architectures, Generative Adversarial Networks, "
        "diffusion-based generative models, and modern web frameworks have "
        "shaped every major design decision in the AgeVision platform.",
        "I extend my sincere appreciation to my fellow MCA classmates and "
        "peers who voluntarily participated in the user-acceptance testing "
        "of the AgeVision web application and provided constructive "
        "feedback on the user interface, age-prediction accuracy, the "
        "realism of the GAN- and diffusion-based age progression outputs, "
        "and the usability of the new admin and batch-processing modules. "
        "Their timely feedback was instrumental in iterating on the "
        "dashboard, history module and real-time camera prediction "
        "interface.",
        "Finally, I would like to express my heartfelt thanks to my family "
        "for their unconditional support, patience, and encouragement "
        "throughout my academic journey. The successful completion of this "
        "project would not have been possible without their constant "
        "motivation.",
    ]
    for p in paras:
        _add(out, "para", p)
    _add(out, "right_text", "(NEERAJ BHUVAN M)", {"bold": True})
    _add(out, "pagebreak")


def table_of_contents(out):
    _add(out, "heading_centered", "TABLE OF CONTENTS",
         {"size": 14, "bold": True, "all_caps": True})
    rows = [
        ("", "ABSTRACT", "viii"),
        ("", "LIST OF TABLES", "vi"),
        ("", "LIST OF FIGURES", "vii"),
        ("1", "INTRODUCTION", "1"),
        ("1.1", "Overview", "1"),
        ("1.2", "Literature Survey", "4"),
        ("1.3", "Proposed System", "9"),
        ("1.4", "Objectives and Scope", "11"),
        ("1.5", "Organization of the Report", "14"),
        ("2", "REQUIREMENTS SPECIFICATION", "15"),
        ("2.1", "Introduction", "15"),
        ("2.2", "Overall Description", "16"),
        ("2.3", "Specific Requirements", "20"),
        ("3", "SYSTEM DESIGN AND TEST PLAN", "27"),
        ("3.1", "Decomposition Description", "27"),
        ("3.2", "Dependency Description", "30"),
        ("3.3", "Detailed Design", "32"),
        ("3.4", "Proposed Sampling Methods", "38"),
        ("3.5", "Test Plan", "40"),
        ("4", "IMPLEMENTATION AND RESULTS", "47"),
        ("4.1", "Implementation", "47"),
        ("4.2", "Results", "56"),
        ("5", "CONCLUSION AND FUTURE WORK", "60"),
        ("5.1", "Summary", "60"),
        ("5.2", "Future Work", "62"),
        ("", "REFERENCES", "64"),
        ("", "APPENDICES", "66"),
    ]
    _add(out, "table",
         ["Chapter No.", "Title", "Page No."],
         rows, None,
         {"col_widths": [1.2, 4.6, 1.0],
          "bold_rows": [0, 1, 2, 3, 9, 13, 19, 22, 25, 26]})
    _add(out, "pagebreak")


def list_of_tables(out):
    _add(out, "heading_centered", "LIST OF TABLES",
         {"size": 14, "bold": True, "all_caps": True})
    rows = [
        ("2.1", "Hardware Requirements", ""),
        ("2.2", "Software Requirements", ""),
        ("2.3", "System Feature Summary", ""),
        ("3.1", "API Endpoint Summary (21 Endpoints)", ""),
        ("3.2", "Database Entity Relationships", ""),
        ("3.2b", "Hybrid Database Schema Summary", ""),
        ("3.3", "List of Modules (12 Active + Stream Helper)", ""),
        ("3.3b", "Testing Strategy Summary", ""),
        ("3.4",
         "Core Functional Test Cases \u2013 Auth, Prediction, Batch",
         ""),
        ("3.5", "Progression and History Test Cases", ""),
        ("3.6",
         "Analytics, Settings and Admin Panel Test Cases", ""),
        ("4.1", "Modules Completed Summary (13 Modules)", ""),
        ("4.2", "Deviations and Justifications", ""),
        ("4.3", "Project Roadmap and Phase Status", ""),
    ]
    _add(out, "table",
         ["Table No.", "Title of Table", "Page No."],
         rows, None,
         {"col_widths": [1.2, 4.6, 1.0]})
    _add(out, "pagebreak")


def list_of_figures(out):
    _add(out, "heading_centered", "LIST OF FIGURES",
         {"size": 14, "bold": True, "all_caps": True})
    rows = [
        ("2.1", "Use Case Diagram", ""),
        ("2.2", "Data Flow Diagram \u2013 Level 0", ""),
        ("2.3", "Data Flow Diagram \u2013 Level 1", ""),
        ("3.1", "System Architecture Diagram", ""),
        ("3.2", "Project Structure", ""),
        ("3.3", "Entity Relationship (ER) Diagram", ""),
        ("3.4", "Database Schema", ""),
        ("4.1", "User Authentication Flow", ""),
        ("4.2", "Age Prediction Interface", ""),
        ("4.3", "MiVOLO v2 Prediction Pipeline", ""),
        ("4.4", "SAM GAN Progression Flow", ""),
        ("4.5", "FADING Diffusion Progression Flow", ""),
        ("4.6", "Side-by-Side Age Progression Output", ""),
        ("4.7", "Batch Prediction Module Output", ""),
        ("4.8", "User Dashboard Interface", ""),
        ("4.9", "Real-Time Camera Prediction Interface", ""),
        ("4.10", "Admin Panel \u2013 User Management & System Health", ""),
    ]
    _add(out, "table",
         ["Figure No.", "Title of Figure", "Page No."],
         rows, None,
         {"col_widths": [1.2, 4.6, 1.0]})
    _add(out, "pagebreak")


def abstract(out):
    _add(out, "heading_centered", "ABSTRACT",
         {"size": 14, "bold": True, "all_caps": True})
    _add(out, "para",
         "AgeVision is a full-stack, AI-powered web platform that integrates "
         "high-accuracy automated age prediction with identity-preserving "
         "face age progression into a single accessible browser interface. "
         "The system was designed and implemented as the final-semester "
         "project of the Master of Computer Applications programme of Anna "
         "University. The frontend is built with Angular 19 (TypeScript "
         "5.7, Bootstrap 5.3, Chart.js, RxJS 7.8) and communicates over "
         "REST with a Django 5.2 backend that uses Django REST Framework "
         "3.16, JWT authentication, Bcrypt password hashing and Fernet "
         "symmetric encryption for sensitive fields. Persistence follows a "
         "hybrid model: MongoDB 4.6 stores schema-flexible AI-result "
         "documents across six collections (users, predictions, "
         "progressions, user_settings, batch_predictions, password_resets) "
         "while SQLite holds Django authentication and session data.")
    _add(out, "para",
         "Age prediction is delivered by an ensemble of MiVOLO v2 (Vision "
         "Transformer with YOLOv8 body context, mean absolute error of "
         "approximately 3.65) as the primary model and InsightFace "
         "buffalo_l (RetinaFace + ArcFace, MAE of approximately 8.5) as "
         "fallback. Age progression is now driven by THREE active "
         "generative engines: SAM GAN (pSp encoder + StyleGAN2 decoder) "
         "as primary, Fast-AgingGAN (a CycleGAN of approximately 11 MB) "
         "as a CPU-friendly fallback, and the newly added FADING "
         "diffusion pipeline (Stable Diffusion + null-text inversion) "
         "for the highest-quality bidirectional aging with optional "
         "cloud GPU acceleration via Modal. Identity preservation across "
         "the source and progressed images is verified using a FaceNet "
         "similarity threshold of 0.6. Emotion classification on detected "
         "faces is performed by the trpakov/vit-face-expression Vision "
         "Transformer hosted on HuggingFace.")
    _add(out, "para",
         "Twelve core modules \u2014 authentication, single- and multi-face "
         "age prediction, batch prediction, emotion detection, three-engine "
         "GAN/diffusion age progression, prediction and progression "
         "history, analytics dashboard, user settings, the main user "
         "dashboard, real-time camera prediction and an admin panel with "
         "system health and user management \u2014 have been fully "
         "implemented and validated. Twenty-one REST endpoints expose this "
         "functionality, and dedicated training pipelines (with Colab and "
         "Kaggle notebooks) are bundled for both MiVOLO v2 and SAM GAN. "
         "AgeVision establishes a reusable reference architecture for "
         "deep-learning-driven face analytics with applications in "
         "forensics, healthcare, entertainment, security and academic "
         "research.")
    _add(out, "pagebreak")


def abstract_tamil(out):
    _add(out, "heading_centered", "சுருக்கம்",
         {"size": 14, "bold": True})
    _add(out, "para",
         "AgeVision என்பது முகத்தின் படத்திலிருந்து ஒருவரின் வயதை "
         "துல்லியமாக கணிப்பதையும், அவருடைய முகத்தை இளமை அல்லது "
         "முதுமை வயதிற்கு மாற்றி காட்டுவதையும் ஒரே இணைய தளத்தில் "
         "வழங்கும் முழுமையான AI அடிப்படையிலான மென்பொருள் அமைப்பாகும். "
         "இது Anna University-ன் Master of Computer Applications "
         "இறுதி கட்டப் பணியாக வடிவமைக்கப்பட்டு செயல்படுத்தப்பட்டது. "
         "முன்பகுதி Angular 19 (TypeScript 5.7, Bootstrap 5.3, "
         "Chart.js) மற்றும் பின்பகுதி Django 5.2 மற்றும் Django "
         "REST Framework 3.16 அடிப்படையில் கட்டமைக்கப்பட்டுள்ளது. "
         "பயனர் அங்கீகாரம் JWT, Bcrypt, Fernet encryption "
         "ஆகியவற்றால் பாதுகாக்கப்படுகிறது. தரவுகள் MongoDB 4.6-ல் "
         "ஆறு collections-ஆக சேமிக்கப்படுகின்றன; Django அங்கீகாரம் "
         "SQLite-ஐப் பயன்படுத்துகிறது.")
    _add(out, "para",
         "வயது கணிப்புக்காக இரண்டு மாதிரிகள் ensemble முறையில் "
         "பயன்படுத்தப்படுகின்றன: முதன்மையாக MiVOLO v2 (Vision "
         "Transformer உடன் YOLOv8 body context; mean absolute error "
         "சுமார் 3.65) மற்றும் துணையாக InsightFace buffalo_l "
         "(RetinaFace + ArcFace). வயது முன்னேற்றத்திற்கு மூன்று "
         "இயந்திரங்கள் வழங்கப்படுகின்றன: முதன்மையான SAM GAN "
         "(pSp encoder உடன் StyleGAN2 decoder), எளிய Fast-AgingGAN "
         "(CycleGAN, 11 MB), மற்றும் புதிதாக சேர்க்கப்பட்ட FADING "
         "diffusion pipeline (Stable Diffusion + null-text "
         "inversion) — இது Modal cloud GPU மூலம் இயக்கப்பட முடியும். "
         "மூன்று இயந்திரங்களிலும் அடையாளம் பாதுகாக்கப்படுவதை "
         "FaceNet similarity threshold 0.6 உறுதி செய்கிறது. "
         "உணர்ச்சி கண்டறிதலுக்கு HuggingFace-இல் பெறப்பட்ட "
         "trpakov/vit-face-expression Vision Transformer "
         "பயன்படுத்தப்படுகிறது.")
    _add(out, "para",
         "மொத்தம் பதிமூன்று modules செயல்படுத்தப்பட்டுள்ளன — "
         "தனி மற்றும் பல-முக வயது கணிப்பு, batch prediction, "
         "emotion detection, மூன்று-இயந்திர GAN/diffusion age "
         "progression, prediction மற்றும் progression history, "
         "analytics dashboard, user settings, main dashboard, "
         "real-time camera prediction, மற்றும் புதிய admin panel. "
         "இருபத்தொரு REST endpoints வழங்கப்பட்டுள்ளன, MiVOLO v2 "
         "மற்றும் SAM GAN-க்கான training pipelines (Colab மற்றும் "
         "Kaggle notebooks உட்பட) இணைக்கப்பட்டுள்ளன. AgeVision "
         "ஆள்விளைவு, சுகாதாரம், பொழுதுபோக்கு, பாதுகாப்பு மற்றும் "
         "கல்வி ஆராய்ச்சிக்கு பயன்படுத்தக்கூடிய நடைமுறை "
         "reference architecture-ஐ வழங்குகிறது.")
    _add(out, "pagebreak")


def _abstract_tamil_legacy_unused(out):  # pragma: no cover - retained for history
    return
    _add(out, "heading_centered", "\u0b9a\u0bc1\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bae\u0bcd",
         {"size": 14, "bold": True})
    _add(out, "para",
         "AgeVision \u0b8e\u0ba9\u0bcd\u0baa\u0ba4\u0bc1 \u0b92\u0bb0\u0bc1 "
         "\u0bae\u0bc1\u0bb4\u0bc1\u0bae\u0bc8\u0baf\u0bbe\u0ba9 "
         "full-stack AI-\u0b85\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0b9f\u0bc8 "
         "\u0b87\u0ba3\u0bc8\u0baf \u0ba4\u0bb3\u0bae\u0bbe\u0b95\u0bc1\u0bae\u0bcd, "
         "\u0b87\u0ba4\u0bc1 \u0b89\u0baf\u0bb0\u0bcd \u0ba4\u0bc1\u0bb2\u0bcd\u0bb2\u0bbf\u0baf "
         "\u0ba4\u0ba9\u0bbf\u0baf\u0bbe\u0bb0\u0bcd \u0bb5\u0baf\u0ba4\u0bc1 "
         "\u0b95\u0ba3\u0bbf\u0baa\u0bcd\u0baa\u0bc8\u0baa\u0bcd\u0baa\u0bc8\u0baf\u0bc1\u0bae\u0bcd "
         "\u0b85\u0b9f\u0bc8\u0baf\u0bbe\u0bb3\u0ba4\u0bcd\u0ba4\u0bc8\u0baa\u0bcd "
         "\u0baa\u0bbe\u0ba4\u0bc1\u0b95\u0bbe\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd "
         "\u0bae\u0bc1\u0b95 \u0bb5\u0baf\u0ba4\u0bc1 "
         "\u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0bc7\u0bb1\u0bcd\u0bb1\u0ba4\u0bcd\u0ba4\u0bc8\u0baf\u0bc1\u0bae\u0bcd "
         "\u0b92\u0bb0\u0bc7 brower \u0b87\u0b9f\u0bc8\u0bae\u0bc1\u0b95\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd "
         "\u0b87\u0ba3\u0bc8\u0b95\u0bcd\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. "
         "\u0b87\u0ba8\u0bcd\u0ba4 \u0b85\u0bae\u0bc8\u0baa\u0bcd\u0baa\u0bc1 "
         "\u0b85\u0ba3\u0bcd\u0ba3\u0bbe \u0baa\u0bb2\u0bcd\u0b95\u0bb2\u0bc8\u0b95\u0bcd\u0b95\u0bb4\u0b95\u0ba4\u0bcd\u0ba4\u0bbf\u0ba9\u0bcd "
         "Master of Computer Applications \u0baa\u0b9f\u0bcd\u0b9f\u0baa\u0bcd "
         "\u0baa\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0bbf\u0ba9\u0bcd \u0b87\u0bb1\u0bc1\u0ba4\u0bbf "
         "\u0baa\u0bbe\u0b9f\u0ba4\u0bcd\u0ba4\u0bbf\u0ba9\u0bcd "
         "\u0b87\u0bb1\u0bc1\u0ba4\u0bbf\u0baa\u0bcd \u0baa\u0ba3\u0bbf\u0baf\u0bbe\u0b95 "
         "\u0bb5\u0b9f\u0bbf\u0bb5\u0bae\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1. "
         "\u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0bc8 \u0baa\u0b95\u0bc1\u0ba4\u0bbf (frontend) "
         "Angular 19 (TypeScript 5.7, Bootstrap 5.3, Chart.js, RxJS 7.8) "
         "\u0b89\u0baa\u0baf\u0bcb\u0b95\u0bbf\u0ba4\u0bcd\u0ba4\u0bc1 "
         "\u0b95\u0b9f\u0bcd\u0b9f\u0bae\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1, "
         "Django 5.2 backend \u0b89\u0b9f\u0ba9\u0bcd Django REST Framework 3.16, "
         "JWT authentication, Bcrypt password hashing, Fernet encryption "
         "\u0b86\u0b95\u0bbf\u0baf\u0bb5\u0bc8 "
         "\u0baa\u0baf\u0ba9\u0bcd\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bc1 REST "
         "\u0bae\u0bc2\u0bb2\u0bae\u0bbe\u0b95 \u0ba4\u0b95\u0bb5\u0bb2\u0bcd "
         "\u0baa\u0bb0\u0bbf\u0bae\u0bbe\u0bb1\u0bcd\u0bb1\u0bae\u0bcd "
         "\u0b9a\u0bc6\u0baf\u0bcd\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. "
         "\u0ba4\u0bb0\u0bb5\u0bc1 \u0b9a\u0bc7\u0bae\u0bbf\u0baa\u0bcd\u0baa\u0bc1 "
         "MongoDB 4.6 (\u0b86\u0bb1\u0bc1 collections: users, predictions, "
         "progressions, user_settings, batch_predictions, password_resets) "
         "\u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd Django authentication\u0b95\u0bcd\u0b95\u0bbe\u0ba9 "
         "SQLite \u0b86\u0b95\u0bbf\u0baf \u0b95\u0bb2\u0baa\u0bcd\u0baa\u0bc1 "
         "\u0bae\u0bc1\u0bb1\u0bc8\u0baf\u0bc8\u0baa\u0bcd \u0baa\u0bbf\u0ba9\u0bcd\u0baa\u0bb1\u0bcd\u0bb1\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1.")
    _add(out, "para",
         "\u0bb5\u0baf\u0ba4\u0bc1 \u0b95\u0ba3\u0bbf\u0baa\u0bcd\u0baa\u0bc1 "
         "\u0baa\u0bbe\u0ba4\u0bc1\u0b95\u0bbe\u0baa\u0bcd\u0baa\u0bc1 \u0b87\u0bb0\u0ba3\u0bcd\u0b9f\u0bc1 "
         "\u0bae\u0bbe\u0ba4\u0bbf\u0bb0\u0bbf\u0b95\u0bb3\u0bbe\u0bb2\u0bcd "
         "\u0b89\u0bb0\u0bc1\u0bb5\u0bbe\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f \u0b92\u0bb0\u0bc1 "
         "ensemble \u0bae\u0bc2\u0bb2\u0bae\u0bcd "
         "\u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1: "
         "\u0bae\u0bc1\u0ba4\u0ba9\u0bcd\u0bae\u0bc8\u0baf\u0bbe\u0ba9 MiVOLO v2 "
         "(Vision Transformer \u0b89\u0b9f\u0ba9\u0bcd YOLOv8 body context, "
         "mean absolute error \u0b9a\u0bc1\u0bae\u0bbe\u0bb0\u0bcd 3.65) \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd "
         "\u0ba4\u0bbf\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bb2\u0bbe\u0b95 InsightFace buffalo_l "
         "(RetinaFace + ArcFace, MAE \u0b9a\u0bc1\u0bae\u0bbe\u0bb0\u0bcd 8.5). "
         "\u0bb5\u0baf\u0ba4\u0bc1 \u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0bc7\u0bb1\u0bcd\u0bb1\u0bae\u0bcd "
         "\u0bae\u0bc2\u0ba9\u0bcd\u0bb1\u0bc1 \u0b9a\u0b95\u0bcd\u0ba4\u0bbf\u0bb5\u0bbe\u0baf\u0ba8\u0bcd\u0ba4 "
         "\u0b89\u0bb0\u0bc1\u0bb5\u0bbe\u0b95\u0bcd\u0b95 "
         "engines\u0b95\u0bb3\u0bbe\u0bb2\u0bcd \u0b9a\u0bbe\u0ba4\u0bbf\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1: "
         "\u0bae\u0bc1\u0ba4\u0ba9\u0bcd\u0bae\u0bc8\u0baf\u0bbe\u0ba9 SAM GAN "
         "(pSp encoder + StyleGAN2 decoder), CPU-\u0b89\u0b95\u0ba8\u0bcd\u0ba4 "
         "\u0ba4\u0bbf\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bb2\u0bbe\u0b95 Fast-AgingGAN "
         "(CycleGAN, \u0b9a\u0bc1\u0bae\u0bbe\u0bb0\u0bcd 11 MB) \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd "
         "\u0baa\u0bc1\u0ba4\u0bbf\u0ba4\u0bbe\u0b95 \u0b9a\u0bc7\u0bb0\u0bcd\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f "
         "FADING diffusion pipeline (Stable Diffusion + null-text inversion) "
         "\u2014 Modal cloud GPU \u0bae\u0bc2\u0bb2\u0bae\u0bbe\u0b95 "
         "\u0b87\u0bb0\u0ba3\u0bcd\u0b9f\u0bc1 \u0ba4\u0bbf\u0b9a\u0bc8 "
         "\u0bb5\u0baf\u0ba4\u0bbe\u0b95\u0bcd\u0b95\u0bb2\u0bc8\u0baf\u0bc1\u0b9f\u0ba9\u0bcd "
         "\u0b89\u0baf\u0bb0\u0bcd\u0ba4\u0bb0 \u0ba4\u0bb0\u0bae\u0bcd "
         "\u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. "
         "\u0bae\u0bc2\u0bb2 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd "
         "\u0bae\u0bbe\u0bb1\u0bcd\u0bb1\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f "
         "\u0baa\u0b9f\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0ba9\u0bcd \u0b87\u0b9f\u0bc8\u0baf\u0bc7 "
         "\u0b85\u0b9f\u0bc8\u0baf\u0bbe\u0bb3\u0ba4\u0bcd\u0ba4\u0bc8 \u0baa\u0bbe\u0ba4\u0bc1\u0b95\u0bbe\u0b95\u0bcd\u0b95 "
         "FaceNet similarity threshold 0.6 "
         "\u0baa\u0baf\u0ba9\u0bcd\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1. "
         "\u0bae\u0bc1\u0b95 \u0bb5\u0bb0\u0bbf\u0b95\u0bb3\u0bbf\u0ba9\u0bcd "
         "\u0b89\u0ba3\u0bb0\u0bcd\u0b9a\u0bcd\u0b9a\u0bbf \u0bb5\u0b95\u0bc8\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bb2\u0bc1\u0b95\u0bcd\u0b95\u0bc1 "
         "HuggingFace-\u0b87\u0bb2\u0bcd \u0baa\u0ba4\u0bbf\u0bb5\u0bc7\u0bb1\u0bcd\u0bb1\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f "
         "trpakov/vit-face-expression Vision Transformer "
         "\u0b89\u0baa\u0baf\u0bcb\u0b95\u0bbf\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1.")
    _add(out, "para",
         "\u0b85\u0ba8\u0bcd\u0ba4\u0bb0\u0b82\u0b95 \u0b85\u0b9f\u0bc8\u0baf\u0bbe\u0bb3\u0bae\u0bcd "
         "\u0b8e\u0ba9\u0bcd\u0bb1 \u0baa\u0ba9\u0bcd\u0ba9\u0bbf\u0bb0\u0ba3\u0bcd\u0b9f\u0bc1 "
         "\u0b85\u0b9f\u0bbf\u0baa\u0bcd\u0baa\u0b9f\u0bc8 modules \u2014 \u0bb5\u0baf\u0ba4\u0bc1 "
         "\u0b95\u0ba3\u0bbf\u0baa\u0bcd\u0baa\u0bc1 (\u0b92\u0bb1\u0bcd\u0bb1\u0bc8 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd "
         "\u0baa\u0bb2\u0bcd-\u0bae\u0bc1\u0b95), batch prediction, emotion detection, "
         "\u0bae\u0bc2\u0ba9\u0bcd\u0bb1\u0bc1-engine GAN/diffusion age progression, "
         "prediction \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd progression history, "
         "analytics dashboard, user settings, \u0bae\u0bc1\u0ba4\u0ba9\u0bcd\u0bae\u0bc8 user dashboard, "
         "real-time camera prediction \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd system health, "
         "user management \u0b95\u0bca\u0ba3\u0bcd\u0b9f admin panel \u2014 "
         "\u0bae\u0bc1\u0bb4\u0bc1\u0bae\u0bc8\u0baf\u0bbe\u0b95 \u0b9a\u0bc6\u0baf\u0bb2\u0bcd\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1 "
         "\u0b9a\u0bcb\u0ba4\u0bbf\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba9. "
         "\u0b87\u0bb0\u0bc1\u0baa\u0ba4\u0bcd\u0ba4\u0bc8\u0ba8\u0bcd\u0ba4\u0bc1 REST endpoints "
         "\u0b87\u0ba8\u0bcd\u0ba4 \u0b9a\u0bc7\u0bb5\u0bc8\u0b95\u0bb3\u0bc8 \u0bb5\u0bc6\u0bb3\u0bbf\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0bc1\u0b95\u0bbf\u0ba9\u0bcd\u0bb1\u0ba9, "
         "MiVOLO v2 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd SAM GAN \u0b87\u0bb0\u0ba3\u0bcd\u0b9f\u0bc1\u0b95\u0bcd\u0b95\u0bc1\u0bae\u0bcd "
         "\u0ba4\u0ba9\u0bbf\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f training pipelines "
         "(Colab \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd Kaggle notebooks \u0b89\u0b9f\u0ba9\u0bcd) "
         "\u0b87\u0ba3\u0bc8\u0b95\u0bcd\u0b95\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba9. "
         "AgeVision \u0b8e\u0ba9\u0bcd\u0baa\u0ba4\u0bc1 "
         "\u0b86\u0bb3\u0bcd\u0bb5\u0bbf\u0ba9\u0bcd\u0bae\u0bc8, \u0b9a\u0bc1\u0b95\u0bbe\u0ba4\u0bbe\u0bb0\u0bae\u0bcd, "
         "\u0baa\u0bca\u0bb4\u0bc1\u0ba4\u0bc1\u0baa\u0bcb\u0b95\u0bcd\u0b95\u0bc1, \u0baa\u0bbe\u0ba4\u0bc1\u0b95\u0bbe\u0baa\u0bcd\u0baa\u0bc1 "
         "\u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd \u0b95\u0bb2\u0bcd\u0bb5\u0bbf \u0b86\u0bb0\u0bbe\u0baf\u0bcd\u0b9a\u0bcd\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bbe\u0ba9 "
         "\u0baa\u0baf\u0ba9\u0bcd\u0baa\u0bbe\u0b9f\u0bc1\u0b95\u0bb3\u0bc1\u0b9f\u0ba9\u0bcd, "
         "\u0b86\u0bb4\u0bae\u0bbe\u0ba9 \u0b95\u0bb1\u0bcd\u0bb1\u0bb2\u0bbf\u0ba9\u0bcd "
         "\u0bae\u0bc2\u0bb2\u0bae\u0bbe\u0b95 \u0bb5\u0bbf\u0baa\u0bb0 \u0bae\u0bc1\u0b95 "
         "\u0baa\u0b95\u0bc1\u0baa\u0bcd\u0baa\u0bbe\u0baf\u0bcd\u0bb5\u0bc1\u0b95\u0bcd\u0b95\u0bbe\u0ba9 "
         "\u0bae\u0bc0\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd \u0baa\u0baf\u0ba9\u0bcd\u0baa\u0b9f\u0bc1\u0bae\u0bcd "
         "reference architecture \u0b92\u0ba9\u0bcd\u0bb1\u0bc8 "
         "\u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0bc1\u0b95\u0bbf\u0bb1\u0ba4\u0bc1.")
    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                CHAPTER I                                    #
# --------------------------------------------------------------------------- #

def chapter_1(out):
    _add(out, "heading_centered", "CHAPTER 1",
         {"size": 14, "bold": True, "all_caps": True, "space_before": 56})
    _add(out, "heading_centered", "INTRODUCTION",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "section", "1.1", "Overview", {"level": 1})
    _add(out, "para",
         "Contemporary digital, medical and forensic workflows increasingly "
         "rely on automated face analysis: identity verification, missing-"
         "person investigations, dermatological age assessments and "
         "entertainment de-aging pipelines. Each depends on two primitives "
         "\u2014 estimating the apparent age from a face image and "
         "synthesising a plausible aged version while preserving identity "
         "\u2014 yet no easily-accessible end-to-end web platform combines "
         "both capabilities. Manual estimation is subjective; commercial "
         "cloud APIs are paid and privacy-sensitive; open-source libraries "
         "solve prediction only and require developer skills.")
    _add(out, "para",
         "AgeVision addresses this gap: a single-page Angular 19 "
         "application backed by a Django REST Framework 5.2 service that "
         "exposes twenty-one REST endpoints covering authentication, "
         "single, multi-face and batch age prediction, emotion detection, "
         "real-time camera capture, three-engine age progression "
         "(SAM GAN, Fast-AgingGAN and FADING diffusion with optional Modal "
         "cloud GPU offload), history, analytics, user settings and a "
         "dedicated admin panel with platform-wide management and system "
         "health monitoring. The workflow that previously required "
         "multiple separate tools \u2014 estimate age, locate a "
         "progression tool, re-upload, iterate, save manually \u2014 is "
         "unified in a single authenticated browser interface.")

    _add(out, "section", "1.2", "Literature Survey", {"level": 1})

    _add(out, "section", "1.2.1", "Deep Learning for Age Estimation",
         {"level": 2})
    _add(out, "para",
         "CNN and transformer architectures dominate apparent-age "
         "regression. The 2024 NIST FATE Report 8525 [3] shows that "
         "single-model commercial engines plateau at 4\u20137 years MAE, "
         "motivating AgeVision\u2019s ensemble of MiVOLO v2 (ViT + YOLOv8 "
         "body context, ~3.65 MAE [1]) with InsightFace buffalo_l "
         "fallback. Shukri et al. [10] identify dataset bias as the "
         "dominant MAE source; AgeVision mitigates this via a fine-tuned "
         "Indian-face SAM checkpoint and a bundled MiVOLO retraining "
         "pipeline.")

    _add(out, "section", "1.2.2", "Generative Models for Face Age Progression",
         {"level": 2})
    _add(out, "para",
         "GAN-based aging dominates the progression literature. Wang et "
         "al. [7] show attention modules preserve identity better than "
         "plain CycleGAN; Abdollahi et al. [11] stress identity "
         "preservation for clinical credibility, directly motivating "
         "AgeVision\u2019s FaceNet similarity gate (threshold 0.6). "
         "AgeVision integrates three engines: SAM [19] (pSp + StyleGAN2 "
         "W+ space) as primary, Fast-AgingGAN (11 MB CycleGAN) as CPU "
         "fallback, and the FADING diffusion pipeline [12] (Stable "
         "Diffusion + null-text inversion) for bidirectional aging at "
         "the highest quality with optional Modal cloud GPU offload.")

    _add(out, "section", "1.2.3",
         "Real-Time Face Detection and Recognition", {"level": 2})
    _add(out, "para",
         "YOLOv8 (ultralytics) and RetinaFace (inside InsightFace "
         "buffalo_l) are AgeVision\u2019s primary and fallback face "
         "detectors. "
         "InsightFace also bundles ArcFace embeddings that AgeVision "
         "uses for its fallback prediction head and for FaceNet-style "
         "identity preservation checks. Combined with OpenCV for image "
         "preprocessing, this stack supports both single-image and "
         "live camera streams captured via the browser\u2019s "
         "getUserMedia API.")

    _add(out, "section", "1.2.4", "Emotion Detection from Facial Imagery",
         {"level": 2})
    _add(out, "para",
         "Emotion detection is implemented through the trpakov/vit-face-"
         "expression Vision Transformer model on HuggingFace [22], which "
         "classifies faces into seven discrete emotion classes (angry, "
         "disgust, fear, happy, neutral, sad, surprise). The model is "
         "loaded once at backend start-up and shares the YOLOv8 face "
         "crops produced for the age prediction pipeline, so emotion "
         "inference adds no additional detection cost.")

    _add(out, "section", "1.2.5",
         "Explainability and Bidirectional Editing", {"level": 2})
    _add(out, "para",
         "Wu et al. [6] introduce CAAE for facial-region explainability; "
         "Li et al. [5] argue clinical AI must publish confidence scores; "
         "Yang et al. [13] propose HRFAE for bidirectional age editing. "
         "AgeVision responds: every prediction reports an ensemble "
         "confidence score, FADING diffusion enables bidirectional "
         "editing in production, and Grad-CAM is listed as future work.")

    _add(out, "section", "1.2.6",
         "Web Frameworks for AI-Powered Applications", {"level": 2})
    _add(out, "para",
         "Al-Quraishi et al. [8] recommend Django REST Framework for "
         "production AI deployments. AgeVision pairs DRF with Angular 19 "
         "so AI engines remain decoupled from the presentation layer. "
         "Existing alternatives (How-Old.net, Face++, AgingBooth, "
         "DeepFace, Luxand SDK) each cover only part of the feature set "
         "\u2014 prediction only, or paid, or mobile-only, or lacking "
         "batch processing and an admin panel \u2014 leaving the "
         "integrated, free, web-accessible platform gap that AgeVision "
         "fills.")

    _add(out, "section", "1.3", "Proposed System", {"level": 1})
    _add(out, "para",
         "AgeVision delivers a free, web-accessible platform that unifies "
         "high-accuracy ensemble age prediction (MiVOLO v2 + InsightFace), "
         "three-engine identity-preserving age progression (SAM GAN, "
         "Fast-AgingGAN, FADING diffusion with optional Modal cloud GPU), "
         "emotion detection, batch processing, real-time camera capture, "
         "a personal history and analytics dashboard, and an admin panel "
         "with user management and system health \u2014 all through a "
         "single authenticated browser interface backed by twenty-one "
         "documented REST endpoints. Key application domains include:")
    _add(out, "bullet", [
        "Forensic identification of missing persons via age progression.",
        "Healthcare: comparing biological and chronological age.",
        "Security: age-gated access control verification.",
        "Academic research with bundled MiVOLO v2 and SAM GAN training "
        "pipelines.",
        "Bulk forensic case-file processing via the batch-prediction module.",
    ])

    _add(out, "section", "1.4", "Objectives and Scope", {"level": 1})
    _add(out, "para",
         "The AgeVision project pursues eight concrete objectives:")
    _add(out, "numbered", [
        "Develop an Angular 19 + Django 5.2 web platform for age "
        "prediction and progression via single upload, batch upload and "
        "real-time camera capture.",
        "Implement an ensemble age prediction pipeline using MiVOLO v2 "
        "(~3.65 MAE) with InsightFace buffalo_l fallback (~8.5 MAE).",
        "Build a three-engine progression module: SAM GAN (primary), "
        "Fast-AgingGAN (CPU fallback), FADING diffusion (bidirectional, "
        "Modal cloud GPU optional).",
        "Integrate seven-class ViT emotion detection alongside every "
        "age prediction request.",
        "Build a secure application with JWT, Bcrypt, Fernet, real-time "
        "camera prediction and SSE progress stream.",
        "Design a hybrid MongoDB + SQLite persistence layer with typed "
        "manager classes and MongoDB aggregations for admin analytics.",
        "Deliver an IsSuperUser-gated admin panel with platform analytics, "
        "user management and live system health monitoring.",
        "Bundle reproducible MiVOLO v2 and SAM GAN training pipelines "
        "and validate all modules with a comprehensive test catalogue.",
    ])
    _add(out, "para",
         "The platform scope covers six functional areas: user "
         "authentication, age prediction (single/multi/batch), emotion "
         "detection, multi-engine age progression, history and analytics, "
         "and the admin panel. The same REST API can be consumed from "
         "mobile applications or external case-management systems.")

    _add(out, "section", "1.5", "Organization of the Report", {"level": 1})
    _add(out, "para",
         "Chapter 2 presents the Requirements Specification (hardware, "
         "software, system features, performance and quality attributes). "
         "Chapter 3 covers System Design and Test Plan (architecture, "
         "API, database schema, sampling methods and test catalogue). "
         "Chapter 4 reports Implementation and Results (modules, figures, "
         "deviations and accuracy summary). Chapter 5 concludes with "
         "a summary, limitations and future work.")

    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                CHAPTER II                                   #
# --------------------------------------------------------------------------- #

def chapter_2(out):
    _add(out, "heading_centered", "CHAPTER 2",
         {"size": 14, "bold": True, "all_caps": True, "space_before": 56})
    _add(out, "heading_centered", "REQUIREMENTS SPECIFICATION",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "section", "2.1", "Introduction", {"level": 1})
    _add(out, "para",
         "This chapter documents the functional and non-functional "
         "requirements of AgeVision: product perspective, operating "
         "environment, system features, data-flow diagrams, performance "
         "targets and software quality attributes.")

    _add(out, "section", "2.2", "Overall Description", {"level": 1})

    _add(out, "section", "2.2.1", "Product Perspective", {"level": 2})
    _add(out, "para",
         "AgeVision is a four-tier web application: Angular 19 frontend "
         "\u2192 Django REST API \u2192 AI/ML layer \u2192 MongoDB + SQLite "
         "data store. After JWT authentication, a user uploads or captures "
         "a face image; YOLOv8 detects faces, MiVOLO v2 predicts age, and "
         "the ViT classifier detects emotion. Optionally, one of three "
         "progression engines (SAM GAN, Fast-AgingGAN, FADING diffusion) "
         "ages the face to a target; results are persisted to MongoDB and "
         "displayed on the personal dashboard.")

    _add(out, "section", "2.2.2", "Product Functions", {"level": 2})
    _add(out, "para",
         "At a high level AgeVision exposes eight product functions to "
         "end users and administrators:")
    _add(out, "bullet", [
        "Secure user registration, JWT-based login, password reset and "
        "profile management.",
        "Single-image age prediction with per-face confidence and "
        "bounding-box overlay.",
        "Multi-face and batch prediction for bulk processing of "
        "photographs.",
        "Real-time camera prediction driven by the browser "
        "getUserMedia API.",
        "Multi-engine identity-preserving age progression (SAM GAN, "
        "Fast-AgingGAN, FADING diffusion) with optional Modal cloud "
        "GPU offload.",
        "Seven-class emotion detection alongside every age prediction.",
        "Personal history, analytics dashboard and configurable user "
        "settings.",
        "Administrative panel for platform analytics, user management "
        "and live system-health monitoring.",
    ])

    _add(out, "section", "2.2.3", "User Characteristics", {"level": 2})
    _add(out, "para",
         "Three external user roles: unauthenticated visitors (landing "
         "page only), standard authenticated users (full prediction, "
         "progression, history, analytics and settings), and superusers "
         "(admin panel for platform stats, user management and system "
         "health). Internal ML microservices are backend-only and are not "
         "directly accessible to clients.")

    _add(out, "section", "2.2.4", "Operating Environment", {"level": 2})
    _add(out, "table",
         ["Component", "Minimum Specification", "Recommended Specification"],
         [
             ("Processor",
              "Intel Core i5 8th Gen / AMD Ryzen 5 (2.0 GHz, Quad Core)",
              "Intel Core i7 / AMD Ryzen 7 (3.5 GHz, 8 Core)"),
             ("RAM", "8 GB DDR4", "16 GB DDR4 or higher"),
             ("Storage",
              "50 GB HDD (OS, Python, Node.js, DB, model checkpoints)",
              "256 GB SSD (faster I/O during model inference)"),
             ("GPU",
              "NVIDIA GTX 1060 with CUDA (for SAM GAN inference)",
              "NVIDIA RTX 3070 / Tesla T4 (for FADING diffusion + GAN)"),
             ("Cloud GPU (Optional)",
              "None",
              "Modal account with A10G or A100 instance for FADING "
              "diffusion offload"),
             ("Network",
              "Broadband Internet (for HuggingFace + Modal)",
              "High-speed broadband (10 Mbps+)"),
             ("Operating System",
              "Windows 10 / Ubuntu 20.04 LTS",
              "Ubuntu 22.04 LTS (production deployment)"),
         ],
         "Table 2.1: Hardware Requirements", {})
    _add(out, "table",
         ["Component", "Technology / Tool", "Version / Notes"],
         [
             ("Frontend Framework", "Angular",
              "Version 19 (TypeScript 5.7, App Router)"),
             ("UI Library", "Bootstrap + Chart.js + RxJS",
              "Bootstrap 5.3, Chart.js 4.x, RxJS 7.8"),
             ("API Backend", "Python + Django + DRF",
              "Python 3.11+, Django 5.2, DRF 3.16"),
             ("Primary Age Model", "MiVOLO v2",
              "ViT + YOLOv8, ~3.65 MAE, with bundled training pipeline"),
             ("Fallback Age Model", "InsightFace buffalo_l",
              "RetinaFace + ArcFace, ~8.5 MAE"),
             ("Age Progression \u2013 Engine 1", "SAM GAN",
              "pSp encoder + StyleGAN2 decoder, with bundled training "
              "pipeline"),
             ("Age Progression \u2013 Engine 2", "Fast-AgingGAN",
              "CycleGAN, 11 MB"),
             ("Age Progression \u2013 Engine 3", "FADING Diffusion",
              "Stable Diffusion + null-text inversion (diffusers + "
              "accelerate); optional Modal cloud GPU"),
             ("Legacy Aging Pipeline", "HRFAE",
              "Standalone training/inference scripts retained"),
             ("Emotion Detection", "ViT face expression",
              "trpakov/vit-face-expression (HuggingFace)"),
             ("Face Detection", "YOLOv8 (ultralytics) + RetinaFace",
              "YOLOv8 8.0+, InsightFace"),
             ("Database (Primary)", "MongoDB", "Version 4.6 + pymongo 4.6.1"),
             ("Database (Auth/Sessions)", "SQLite", "Django default ORM"),
             ("Deep Learning Framework", "PyTorch + TensorFlow + diffusers",
              "PyTorch 2.0+, TensorFlow 2.20, diffusers 0.25+"),
             ("Cloud GPU (Optional)", "Modal", "modal>=0.73.0"),
             ("Authentication", "JWT + Bcrypt + Fernet",
              "SimpleJWT 5.5, bcrypt 5.x"),
             ("Computer Vision", "OpenCV", "Version 4.13+"),
             ("Package Managers", "npm + pip", "npm 10.x, pip 23.x"),
             ("IDE / Editor", "VS Code / PyCharm",
              "With ESLint and Black"),
         ],
         "Table 2.2: Software Requirements", {})

    _add(out, "section", "2.2.5", "Constraints", {"level": 2})
    _add(out, "para",
         "Hard constraints: CUDA GPU required for SAM GAN and FADING "
         "(Fast-AgingGAN provides CPU fallback); internet required on "
         "first launch (HuggingFace checkpoints) and for Modal cloud "
         "offload; getUserMedia browser permission required for camera; "
         "MongoDB must be reachable (failure degrades to prediction-only "
         "mode); all model checkpoints occupy approximately 6 GB.")

    _add(out, "section", "2.3", "Specific Requirements", {"level": 1})

    _add(out, "section", "2.3.1", "External Interface Requirements",
         {"level": 2})
    _add(out, "para",
         "The user interface is an Angular 19 SPA served to any modern "
         "browser over HTTPS. The software interface is a 21-endpoint "
         "REST API catalogued in Table 3.1, grouped into auth, "
         "prediction, progression, emotion, history, analytics, "
         "settings and admin. Hardware interfaces comprise the browser "
         "camera (getUserMedia) and an optional host GPU. External "
         "communication integrates HuggingFace Hub (ViT and Stable "
         "Diffusion weights), the ultralytics YOLOv8 package and the "
         "Modal serverless GPU SDK for FADING diffusion offload.")

    _add(out, "section", "2.3.2", "System Features", {"level": 2})
    _add(out, "table",
         ["Feature", "Description"],
         [
             ("Multi-Modal Input",
              "Upload images (JPEG/PNG/WebP \u2264 5 MB), batch upload, or "
              "live camera capture via getUserMedia (auto-capture 1500 ms)."),
             ("Ensemble Age Prediction",
              "MiVOLO v2 + InsightFace buffalo_l ensemble returns predicted "
              "age, confidence and bounding-box overlay per face."),
             ("Three-Engine Age Progression",
              "SAM GAN (primary), Fast-AgingGAN (CPU fallback), FADING "
              "diffusion (bidirectional, optional Modal cloud GPU); FaceNet "
              "identity threshold 0.6."),
             ("Emotion Detection",
              "Seven-class ViT emotion classifier (trpakov/vit-face-"
              "expression) runs on every detected face in one batched pass."),
             ("Batch + Progress Stream",
              "/predict/batch accepts bulk image arrays; /progress/stream "
              "Server-Sent Events pushes percent-complete to the Angular "
              "client during long diffusion jobs."),
             ("History, Analytics & Settings",
              "MongoDB persistence with thumbnails, pagination, deletion and "
              "download; Chart.js analytics; per-user theme and model prefs."),
             ("Admin Panel",
              "IsSuperUser-gated panel: platform stats, user "
              "search/suspend/reinstate, live system health (disk, uptime, "
              "cache, model availability)."),
             ("Secure Auth",
              "JWT access + refresh tokens, Bcrypt password hashing, Fernet "
              "field encryption, strict CORS policy."),
         ],
         "Table 2.3: System Feature Summary",
         {"col_widths": [2.2, 4.6]})

    _add(out, "section", "2.3.3", "Data Flow and UML Diagrams",
         {"level": 2})
    _add(out, "para",
         "Four actors interact with AgeVision: the standard User, the "
         "Admin (superuser flag), the platform (Angular client + Django "
         "backend) and external services (HuggingFace Hub, Ultralytics "
         "YOLOv8 and Modal GPU). The full set of use cases is shown "
         "in Figure 2.1.")
    _add(out, "figure",
         "Figure 2.1: Use Case Diagram",
         {"image": "fig_2_1_use_case.png"})
    _add(out, "para",
         "At Level 0 the system receives image, batch, camera and auth "
         "inputs from User/Admin; returns predictions, progressions "
         "and admin reports; uses HuggingFace Hub for model weights, "
         "MongoDB for persistence and Modal for optional GPU offload.")
    _add(out, "figure",
         "Figure 2.2: Data Flow Diagram \u2013 Level 0 (Context Diagram)",
         {"image": "fig_2_2_dfd_l0.png"})
    _add(out, "para",
         "At Level 1 the system decomposes into eight sub-processes: "
         "authentication, media upload, face detection, age "
         "prediction + emotion, multi-engine progression, history "
         "management, analytics aggregation and admin operations, "
         "each owning its own collection or table in the hybrid "
         "MongoDB + SQLite store.")
    _add(out, "figure",
         "Figure 2.3: Data Flow Diagram \u2013 Level 1 (Detailed)",
         {"image": "fig_2_3_dfd_l1.png"})

    _add(out, "section", "2.3.4", "Performance Requirements", {"level": 2})
    _add(out, "para",
         "Indicative latency targets: single-image prediction \u2248 "
         "800 ms; SAM GAN progression 3\u20138 s; FADING diffusion "
         "12\u201320 s locally or \u2248 3 s on Modal A10G; batch of "
         "20 images scales linearly; cold-load \u2264 30 s.")

    _add(out, "section", "2.3.5", "Software Quality Attributes",
         {"level": 2})
    _add(out, "bullet", [
        "Security \u2014 JWT tokens, Bcrypt hashing, Fernet field "
        "encryption, CORS policy and IsSuperUser gating enforce "
        "defence in depth.",
        "Reliability \u2014 prediction falls back from MiVOLO v2 to "
        "InsightFace on load failure; progression falls back from SAM "
        "GAN to Fast-AgingGAN on GPU OOM; FaceNet threshold 0.6 "
        "preserves identity.",
        "Usability \u2014 Angular 19 SPA delivers camera capture, "
        "progress streaming and a thumbnail-rich history dashboard "
        "to users with no ML background.",
        "Maintainability \u2014 the four-tier architecture and typed "
        "manager layer allow engines or data stores to be swapped "
        "without touching the presentation layer.",
        "Scalability \u2014 stateless JWT auth and Modal serverless "
        "GPU offload allow horizontal API scaling with no shared "
        "session storage.",
    ])

    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                CHAPTER III                                  #
# --------------------------------------------------------------------------- #

def chapter_3(out):
    _add(out, "heading_centered", "CHAPTER 3",
         {"size": 14, "bold": True, "all_caps": True, "space_before": 56})
    _add(out, "heading_centered", "SYSTEM DESIGN AND TEST PLAN",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "section", "3.1", "Decomposition Description", {"level": 1})

    _add(out, "section", "3.1.1", "System Architecture", {"level": 2})
    _add(out, "para",
         "AgeVision uses a four-tier architecture: Angular 19 SPA "
         "(client) communicates over HTTPS with a Django 5.2 + DRF "
         "3.16 API (JWT auth, IsSuperUser, CORS, rate limiting) which "
         "delegates to an AI/ML layer (MiVOLO v2, InsightFace, SAM "
         "GAN, Fast-AgingGAN, FADING diffusion optionally on Modal, "
         "ViT emotion, YOLOv8, OpenCV) backed by MongoDB 4.6 (six "
         "collections) and SQLite.")

    _add(out, "figure",
         "Figure 3.1: System Architecture Diagram",
         {"image": "fig_3_1_architecture.png"})

    _add(out, "section", "3.1.2", "Project Structure", {"level": 2})
    _add(out, "section", "3.1.2.1", "Frontend Repository (Angular 19)",
         {"level": 3})
    _add(out, "para",
         "The Angular 19 project uses the standard CLI layout: "
         "/src/app contains feature modules (auth, dashboard, "
         "prediction, batch-predict, progression, history, analytics, "
         "settings and admin), each with components, services, guards "
         "and routing. AdminGuard restricts /admin/* routes to "
         "superusers; /src/environments carries the dev/production "
         "config swap.")
    _add(out, "section", "3.1.2.2", "Backend Repository (Django + DRF)",
         {"level": 3})
    _add(out, "para",
         "The Django backend lives under agevision_backend/"
         "agevision_api/ and contains views, models, serializers, URL "
         "routes and permissions.py. AI engine wrappers "
         "(age_predictor.py, gan_progression.py, emotion_detector.py, "
         "mongodb.py) delegate heavy inference to sam/, fast_aging/, "
         "diffusion_aging/ and mivolo/ sub-directories, which also "
         "ship Colab/Kaggle training notebooks.")
    _add(out, "figure",
         "Figure 3.2: Project Structure",
         {"image": "fig_3_2_project_structure.png"})

    _add(out, "section", "3.2", "Dependency Description", {"level": 1})
    _add(out, "para",
         "The Angular client holds no business logic and depends on "
         "the Django REST API for all non-static operations; AdminGuard "
         "additionally requires the is_superuser flag in the JWT "
         "payload. The API depends on the AI/ML layer via "
         "age_predictor.py, gan_progression.py and mongodb.py, and on "
         "the hybrid data tier (SQLite through Django ORM, MongoDB "
         "through PyMongo). Three external services are consumed: "
         "HuggingFace Hub for ViT emotion and Stable Diffusion weights, "
         "the ultralytics package for YOLOv8 weights, and Modal via its "
         "Python SDK for FADING GPU offload. Each external failure "
         "degrades gracefully: HuggingFace falls back to cached weights, "
         "YOLOv8 to InsightFace RetinaFace, and Modal to local "
         "diffusion inference.")

    _add(out, "section", "3.3", "Detailed Design", {"level": 1})

    _add(out, "section", "3.3.1", "API Design", {"level": 2})
    _add(out, "para",
         "All 21 endpoints follow REST conventions under /api/v1/; "
         "protected routes require a valid JWT (HTTP 401 otherwise) and "
         "admin routes additionally require the IsSuperUser permission. "
         "Responses use a { data, message } success envelope and "
         "{ error, code } error envelope.")
    _add(out, "table",
         ["Module", "Method", "Endpoint", "Access", "Description"],
         [
             ("Auth", "POST", "/api/v1/auth/register", "Public",
              "User registration"),
             ("Auth", "POST", "/api/v1/auth/login", "Public",
              "Login, returns JWT"),
             ("Auth", "POST", "/api/v1/auth/forgot-password", "Public",
              "Password reset email"),
             ("Age Prediction", "POST", "/api/v1/predict/single", "Auth",
              "Single face age prediction"),
             ("Age Prediction", "POST", "/api/v1/predict/multi", "Auth",
              "Multi-face group analysis"),
             ("Age Prediction", "POST", "/api/v1/predict/batch", "Auth",
              "Batch prediction across multiple images (NEW)"),
             ("Emotion", "POST", "/api/v1/emotion/detect", "Auth",
              "Emotion classification"),
             ("Camera", "POST", "/api/v1/predict/camera", "Auth",
              "Real-time camera frame prediction"),
             ("Age Progression", "POST", "/api/v1/progress/generate",
              "Auth",
              "Multi-engine progression (sam / fast_aging / diffusion)"),
             ("Age Progression", "GET", "/api/v1/progress/:id", "Auth",
              "Retrieve progression result"),
             ("Age Progression", "GET", "/api/v1/progress/stream",
              "Auth", "Server-Sent Events progress stream (NEW)"),
             ("History", "GET", "/api/v1/history/predictions", "Auth",
              "List prediction history"),
             ("History", "GET", "/api/v1/history/progressions", "Auth",
              "List progression history"),
             ("History", "DELETE", "/api/v1/history/:id", "Auth",
              "Delete history entry"),
             ("Analytics", "GET", "/api/v1/analytics/summary", "Auth",
              "Dashboard analytics summary"),
             ("User", "GET", "/api/v1/users/me", "Auth",
              "Get current user profile"),
             ("User", "PATCH", "/api/v1/users/settings", "Auth",
              "Update user settings"),
             ("Admin", "GET", "/api/v1/admin/dashboard", "Superuser",
              "Platform-wide stats (NEW)"),
             ("Admin", "GET", "/api/v1/admin/users", "Superuser",
              "Paginated user list with search (NEW)"),
             ("Admin", "GET", "/api/v1/admin/users/:id", "Superuser",
              "User detail with recent activity (NEW)"),
             ("Admin", "POST", "/api/v1/admin/users/:id/suspend",
              "Superuser", "Suspend user (NEW)"),
             ("Admin", "POST", "/api/v1/admin/users/:id/reinstate",
              "Superuser", "Reinstate user (NEW)"),
             ("Admin", "GET", "/api/v1/admin/system/health",
              "Superuser",
              "System health metrics: disk, cache, uptime, models (NEW)"),
         ],
         "Table 3.1: API Endpoint Summary (21 Endpoints, 6 New for Admin "
         "+ Batch + Stream)", {})

    _add(out, "section", "3.3.2", "Database Schema", {"level": 2})
    _add(out, "para",
         "SQLite stores Django auth tables (users, sessions, tokens) "
         "via the ORM; MongoDB 4.6 stores all AI-result documents "
         "across six collections via the PyMongo manager layer in "
         "mongodb.py (MongoUserManager, MongoPredictionManager, "
         "MongoProgressionManager, etc.) which also exposes admin "
         "aggregation methods that power the dashboard.")
    _add(out, "table",
         ["Relationship", "Type", "Description"],
         [
             ("User \u2192 Prediction", "One-to-Many",
              "One user can have multiple age predictions"),
             ("User \u2192 Progression", "One-to-Many",
              "One user can have multiple age progressions"),
             ("User \u2192 Settings", "One-to-One",
              "Each user has one settings document"),
             ("User \u2192 Batch Job", "One-to-Many",
              "One user can submit multiple batch jobs"),
             ("Prediction \u2192 Faces", "One-to-Many",
              "One prediction contains multiple detected faces"),
             ("Batch Job \u2192 Predictions", "One-to-Many",
              "One batch job aggregates many per-image predictions"),
         ],
         "Table 3.2: Database Entity Relationships", {})
    _add(out, "figure",
         "Figure 3.3: Entity Relationship (ER) Diagram",
         {"image": "fig_3_3_er_diagram.png"})

    # Merged schema table — all 6 collections in one compact table
    _add(out, "table",
         ["Collection", "Field", "Type", "Description"],
         [
             # users (SQLite)
             ("users (SQLite)", "id", "INTEGER PK", "Auto-incrementing primary key"),
             ("", "username / email", "VARCHAR", "Unique credentials"),
             ("", "password", "VARCHAR(128)", "Bcrypt hashed"),
             ("", "is_active / is_superuser", "BOOLEAN", "Account active; admin gate"),
             # predictions
             ("predictions", "_id / userId", "ObjectId / Int", "PK; FK to users.id"),
             ("", "faces", "Array<Object>", "bbox[], age, confidence, gender, emotion"),
             ("", "model / processingMs", "String / Int", "Ensemble model; inference time ms"),
             # progressions
             ("progressions", "_id / userId", "ObjectId / Int", "PK; FK to users.id"),
             ("", "targetAge / ganModel", "Int / String", "Requested age; SAM_GAN / Fast / FADING"),
             ("", "identityScore / executionTarget", "Float / String",
              "FaceNet score; local_gpu / modal_cloud"),
             # user_settings
             ("user_settings", "userId", "Int (unique)", "FK to users.id"),
             ("", "theme / preferredAgeModel", "String", "light/dark; MiVOLO_v2/Ensemble"),
             ("", "preferredGanModel / notificationsEnabled", "String / Bool",
              "SAM_GAN / FADING_Diffusion; email opt-in"),
             # batch_predictions
             ("batch_predictions", "_id / userId", "ObjectId / Int", "PK; FK to users.id"),
             ("", "status / totalImages", "String / Int",
              "queued/running/completed/failed; image count"),
             ("", "results / processingMs", "Array / Int",
              "Per-image prediction summary; total inference ms"),
             # password_resets
             ("password_resets", "token / expiresAt", "String / ISODate", "Reset token; TTL"),
             ("", "used", "Boolean", "One-shot flag"),
         ],
         "Table 3.2b: Hybrid Database Schema Summary",
         {"col_widths": [1.6, 1.8, 1.4, 2.0]})

    _add(out, "figure",
         "Figure 3.4: Database Schema",
         {"image": "fig_3_4_db_schema.png"})

    _add(out, "section", "3.1.3", "List of Modules", {"level": 2})
    _add(out, "table",
         ["Module Name", "Description", "Backend"],
         [
             ("User Authentication Module",
              "JWT registration, login, password reset, Fernet encryption",
              "Django DRF"),
             ("Age Prediction \u2013 Single Face",
              "MiVOLO v2 + InsightFace ensemble, confidence, bbox overlay",
              "Django DRF"),
             ("Multi-Face Detection and Group Analysis",
              "YOLOv8 primary, RetinaFace fallback, per-face results",
              "Django DRF"),
             ("Batch Prediction Module (NEW)",
              "Bulk image upload + bulk prediction + per-image grid",
              "Django DRF"),
             ("Emotion Detection Module",
              "ViT face expression, 7-class, batch processing",
              "Django DRF"),
             ("Multi-Engine Age Progression Module",
              "SAM GAN + Fast-AgingGAN + FADING Diffusion (Modal-optional)",
              "Django DRF"),
             ("Progress Stream Module (NEW)",
              "Server-Sent Events for long progression jobs",
              "Django DRF"),
             ("Prediction & Progression History",
              "MongoDB CRUD, thumbnails, pagination, delete",
              "Django DRF"),
             ("Analytics Dashboard",
              "Chart.js: age/gender/emotion distribution + activity timeline",
              "Django DRF"),
             ("User Settings and Preferences",
              "Per-user MongoDB settings document, dark/light theme",
              "Django DRF"),
             ("Dashboard Module",
              "Quick stats, recent activity, quick analyze buttons",
              "Django DRF"),
             ("Real-Time Camera Prediction",
              "getUserMedia, 1500 ms auto-capture, live overlay",
              "Angular 19"),
             ("Admin Panel Module (NEW)",
              "Platform stats, user mgmt (search/suspend/reinstate), "
              "system health",
              "Django DRF + AdminGuard"),
         ],
         "Table 3.3: List of Modules (12 Active + Stream Helper)", {})

    _add(out, "section", "3.3.3", "Overall System I/O Design", {"level": 2})
    _add(out, "para",
         "Inputs include single images (JPEG/PNG/WebP \u2264 5 MB), "
         "batch arrays, live camera frames, JSON payloads with target "
         "age and engine choice, and JWT-protected user-management "
         "requests; client-side validation rejects oversize or "
         "unsupported uploads before transmission. The pipeline "
         "executes asynchronously \u2014 face detection \u2192 "
         "inference \u2192 serialisation \u2192 MongoDB write \u2192 "
         "HTTP response \u2014 with FADING diffusion jobs streaming "
         "real-time progress over /api/v1/progress/stream SSE and "
         "optional Modal GPU offload keeping the thread pool "
         "responsive. Prediction responses return face_count, bounding "
         "boxes, predicted ages, confidence, emotions and genders; "
         "progression responses return original and progressed image "
         "paths, target age, ganModel, executionTarget and FaceNet "
         "identity_score.")

    _add(out, "section", "3.4", "Proposed Sampling Methods", {"level": 1})
    _add(out, "para",
         "MiVOLO v2 is trained on UTKFace [15] and IMDB-WIKI [16], "
         "stratified into 5-year age buckets with over-sampling of "
         "under-represented tails (< 10, > 70 years) and split "
         "80/10/10 with subject-level disjointness. SAM GAN is "
         "fine-tuned on FFHQ [17] plus a curated Indian-face subset "
         "filtered by YOLOv8 confidence \u2265 0.8 and labelled by "
         "MiVOLO v2, using the same 80/10/10 split. FADING diffusion "
         "is used as a pretrained engine and evaluated on 500 frontal "
         "UTKFace portraits sampled uniformly across ages 0\u201390, "
         "with identity preservation measured via FaceNet similarity. "
         "All splits use fixed random seeds for reproducibility.")

    _add(out, "section", "3.5", "Test Plan", {"level": 1})
    _add(out, "table",
         ["Testing Type", "Approach and Scope"],
         [
             ("Unit",
              "pytest-style tests mock AI engines and assert JSON "
              "envelope shape, status codes and DB side-effects; "
              "test_auth.py, test_age_prediction.py, test_progression.py, "
              "test_crypto.py, test_mongodb_e2e.py and test_gan_accuracy.py "
              "cover the full surface including admin endpoints."),
             ("Integration",
              "test_api_e2e.py runs the full prediction, batch, "
              "progression and admin flows against a live instance with "
              "real model weights."),
             ("Functional",
              "Manual tests through the Angular UI verify every user-"
              "facing flow in Section 2.3.2 against the test cases in "
              "Section 3.5.5, including batch-predict and admin panel."),
             ("Performance",
              "Single-face prediction \u2248 800 ms; SAM GAN progression "
              "3\u20138 s; FADING diffusion 12\u201320 s locally or "
              "\u2248 3 s on Modal A10G."),
         ],
         "Table 3.3b: Testing Strategy Summary",
         {"col_widths": [1.5, 5.3]})

    _add(out, "section", "3.5.5", "Test Cases", {"level": 2})

    def tc(num, desc, inp, exp):
        return (num, desc, inp, exp, "Pass")

    def tcm(mod, num, desc, inp, exp):
        return (mod, num, desc, inp, exp, "Pass")

    _add(out, "section", "3.5.5.1",
         "Core Functional Test Cases \u2013 Auth, Prediction & Batch",
         {"level": 3})
    _add(out, "table",
         ["Module", "TC#", "Test Description", "Input / Action",
          "Expected Output", "Status"],
         [
             tcm("Auth", "TC-A01", "Valid registration",
                 "Unique email + valid password",
                 "201 Created, user persisted in SQLite"),
             tcm("Auth", "TC-A02", "Duplicate email registration",
                 "Email already in DB", "400 Bad Request"),
             tcm("Auth", "TC-A03", "Weak password rejected",
                 "Password length < 8", "400 Bad Request"),
             tcm("Auth", "TC-A04", "Valid login",
                 "Correct credentials",
                 "200 OK + access + refresh JWT"),
             tcm("Auth", "TC-A05", "Invalid login",
                 "Wrong password", "401 Unauthorized"),
             tcm("Auth", "TC-A06", "Token refresh",
                 "Valid refresh token",
                 "200 OK with new access token"),
             tcm("Auth", "TC-A07", "Forgot password",
                 "Registered email", "200 OK, reset email sent"),
             tcm("Auth", "TC-A08", "Reset password",
                 "Valid reset token + new password",
                 "200 OK, password updated"),
             tcm("Auth", "TC-A09", "Profile fetch",
                 "GET /users/me with JWT",
                 "200 OK with user profile"),
             tcm("Auth", "TC-A10", "Logout / token revoke",
                 "POST logout with JWT",
                 "200 OK, token blacklisted"),
             tcm("Prediction", "TC-P01", "Single face prediction",
                 "Frontal portrait JPG",
                 "200 OK, predicted_age + confidence + bbox"),
             tcm("Prediction", "TC-P02", "Multi-face prediction",
                 "Group photo with 3 faces",
                 "200 OK, face_count=3 with per-face ages"),
             tcm("Prediction", "TC-P03", "Image with no face",
                 "Landscape photograph",
                 "200 OK, face_count=0"),
             tcm("Prediction", "TC-P04", "Camera frame prediction",
                 "Live frame captured by getUserMedia",
                 "200 OK, predicted_age + bbox"),
             tcm("Prediction", "TC-P05", "Unsupported file format",
                 ".bmp upload", "400 Bad Request"),
             tcm("Prediction", "TC-P06", "Oversize upload",
                 "10 MB file", "413 Payload Too Large"),
             tcm("Prediction", "TC-P07", "Ensemble agreement",
                 "Same image to ensemble",
                 "Both heads return age within 5 yr"),
             tcm("Prediction", "TC-P08", "InsightFace fallback",
                 "Disable MiVOLO temporarily",
                 "200 OK with InsightFace prediction"),
             tcm("Prediction", "TC-P09", "Persistence",
                 "After prediction",
                 "Document inserted into predictions collection"),
             tcm("Batch", "TC-P10", "Batch prediction (NEW)",
                 "POST /predict/batch with 5 images",
                 "200 OK, batchId + 5 per-image results"),
             tcm("Batch", "TC-P11", "Batch persistence (NEW)",
                 "After batch run",
                 "Document inserted into batch_predictions "
                 "collection"),
         ],
         "Table 3.4: Core Functional Test Cases \u2013 Auth, "
         "Prediction and Batch Modules", {})

    _add(out, "section", "3.5.5.2",
         "Progression & History Test Cases",
         {"level": 3})
    _add(out, "table",
         ["Module", "TC#", "Test Description", "Input / Action",
          "Expected Output", "Status"],
         [
             tcm("Progression", "TC-G01", "SAM GAN progression",
                 "Source image + target_age=60 + ganModel='sam'",
                 "200 OK with progressed image and "
                 "identity_score>0.6"),
             tcm("Progression", "TC-G02", "Fast-AgingGAN fallback",
                 "ganModel='fast_aging'",
                 "200 OK, progressed image returned"),
             tcm("Progression", "TC-G03", "FADING diffusion (NEW)",
                 "ganModel='diffusion' + target_age=70",
                 "200 OK, executionTarget set, "
                 "identity_score>0.6"),
             tcm("Progression", "TC-G04",
                 "FADING diffusion bidirectional (NEW)",
                 "ganModel='diffusion' + target_age=8 (younger)",
                 "200 OK, younger face returned"),
             tcm("Progression", "TC-G05", "Modal cloud offload (NEW)",
                 "ganModel='diffusion' with cloud=true",
                 "200 OK, executionTarget=modal_cloud, latency "
                 "reduced"),
             tcm("Progression", "TC-G06", "Identity preservation",
                 "FaceNet similarity check on all engines",
                 "Identity_score > 0.6"),
             tcm("Progression", "TC-G07", "Side-by-side comparison",
                 "After any progression",
                 "comparison_image_path generated"),
             tcm("Progression", "TC-G08", "Same age regression",
                 "target_age == current",
                 "Output visually identical to input"),
             tcm("Progression", "TC-G09", "Invalid target age",
                 "target_age = -5", "400 Bad Request"),
             tcm("Progression", "TC-G10", "Persistence",
                 "After progression",
                 "Document inserted into progressions collection"),
             tcm("Progression", "TC-G11", "Progress stream (NEW)",
                 "GET /progress/stream during diffusion job",
                 "SSE events emitted at \u2265 1 Hz"),
             tcm("History", "TC-H01", "List predictions",
                 "GET /history/predictions",
                 "200 OK with paginated list"),
             tcm("History", "TC-H02", "List progressions",
                 "GET /history/progressions",
                 "200 OK with paginated list"),
             tcm("History", "TC-H03", "Delete entry",
                 "DELETE /history/:id",
                 "204 No Content, document removed"),
             tcm("History", "TC-H04", "Other-user isolation",
                 "User B requests User A's id",
                 "404 Not Found"),
         ],
         "Table 3.5: Progression and History Test Cases", {})

    _add(out, "section", "3.5.5.3",
         "Analytics, Settings & Admin Panel Test Cases",
         {"level": 3})
    _add(out, "table",
         ["Module", "TC#", "Test Description", "Input / Action",
          "Expected Output", "Status"],
         [
             tcm("Analytics", "TC-N01", "Summary aggregation",
                 "GET /analytics/summary",
                 "200 OK with totals + distributions"),
             tcm("Analytics", "TC-N02", "Empty user analytics",
                 "Brand-new user",
                 "200 OK with zeroed metrics"),
             tcm("Analytics", "TC-N03", "Chart data shape",
                 "GET /analytics/summary",
                 "Response keys match Chart.js schema"),
             tcm("Settings", "TC-S01", "Read settings",
                 "GET /users/settings",
                 "200 OK with user settings document"),
             tcm("Settings", "TC-S02", "Update theme",
                 "PATCH theme=dark", "200 OK, settings updated"),
             tcm("Settings", "TC-S03", "Switch preferred GAN",
                 "PATCH preferredGanModel=FADING_Diffusion",
                 "200 OK, change persisted"),
             tcm("Settings", "TC-S04", "Invalid value rejected",
                 "PATCH theme=neon", "400 Bad Request"),
             tcm("Admin", "TC-AD01", "Non-admin blocked",
                 "GET /admin/dashboard with normal user JWT",
                 "403 Forbidden"),
             tcm("Admin", "TC-AD02", "Admin dashboard",
                 "GET /admin/dashboard with superuser JWT",
                 "200 OK with platform stats"),
             tcm("Admin", "TC-AD03", "User search",
                 "GET /admin/users?q=neeraj",
                 "200 OK with paginated matches"),
             tcm("Admin", "TC-AD04", "User detail",
                 "GET /admin/users/:id",
                 "200 OK with user + recent activity"),
             tcm("Admin", "TC-AD05", "Suspend user",
                 "POST /admin/users/:id/suspend",
                 "200 OK, user.is_active=false"),
             tcm("Admin", "TC-AD06", "Reinstate user",
                 "POST /admin/users/:id/reinstate",
                 "200 OK, user.is_active=true"),
             tcm("Admin", "TC-AD07", "System health",
                 "GET /admin/system/health",
                 "200 OK with disk, models, uptime"),
         ],
         "Table 3.6: Analytics, Settings and Admin Panel "
         "Test Cases (NEW)", {})

    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                CHAPTER IV                                   #
# --------------------------------------------------------------------------- #

def chapter_4(out):
    _add(out, "heading_centered", "CHAPTER 4",
         {"size": 14, "bold": True, "all_caps": True, "space_before": 56})
    _add(out, "heading_centered", "IMPLEMENTATION AND RESULTS",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "section", "4.1", "Implementation", {"level": 1})

    _add(out, "section", "4.1.1", "Implementation Environment",
         {"level": 2})
    _add(out, "para",
         "Development used a Windows 11 workstation (VS Code for "
         "Angular, PyCharm for Django), Node.js 20 LTS / npm 10.x, "
         "Python 3.11 in a virtualenv hosting Django 5.2 and the full "
         "deep-learning stack, MongoDB 4.6 locally and SQLite bundled "
         "with Django. ESLint/Prettier enforced TypeScript style; "
         "Black/Flake8 enforced PEP-8. SAM GAN training, MiVOLO "
         "fine-tuning and FADING offload were conducted on Google "
         "Colab, Kaggle and Modal serverless GPU functions.")

    _add(out, "section", "4.1.2", "Modules Completed", {"level": 2})
    _add(out, "table",
         ["#", "Module Name", "Status", "Key Technologies",
          "Key Features"],
         [
             ("1", "User Authentication", "Completed",
              "Django DRF + SimpleJWT + Bcrypt + Fernet",
              "Register, login, refresh, reset, encrypted profile"),
             ("2", "Age Prediction \u2013 Single Face", "Completed",
              "MiVOLO v2 + InsightFace + YOLOv8",
              "Ensemble age + confidence + bbox overlay"),
             ("3", "Multi-Face Detection", "Completed",
              "YOLOv8 + RetinaFace",
              "Per-face age, gender, emotion in group photos"),
             ("4", "Batch Prediction (NEW)", "Completed",
              "Django DRF + Angular drag-drop + MongoDB batch_predictions",
              "Bulk image upload, per-image grid, batch persistence"),
             ("5", "Emotion Detection", "Completed",
              "ViT face expression (HuggingFace)",
              "7-class emotion classification, batch capable"),
             ("6", "Multi-Engine Age Progression", "Completed",
              "SAM GAN + Fast-AgingGAN + FADING Diffusion + Modal",
              "Three engines + identity check + cloud offload"),
             ("7", "Progress Stream (NEW)", "Completed",
              "Django StreamingHttpResponse + Angular EventSource",
              "Server-Sent Events for long progression jobs"),
             ("8", "Prediction & Progression History", "Completed",
              "MongoDB + Angular cards",
              "Thumbnails, pagination, deletion, download"),
             ("9", "Analytics Dashboard", "Completed",
              "Chart.js + Django aggregations",
              "Age, gender, emotion distribution + activity timeline"),
             ("10", "User Settings", "Completed",
              "MongoDB settings doc + Angular forms",
              "Theme, preferred age & GAN/diffusion model, "
              "notifications"),
             ("11", "Dashboard", "Completed",
              "Angular + DRF",
              "Quick stats, recent activity, quick-analyse shortcuts"),
             ("12", "Real-Time Camera Prediction", "Completed",
              "Angular getUserMedia + DRF",
              "1500 ms auto-capture, live overlay, save-to-history"),
             ("13", "Admin Panel (NEW)", "Completed",
              "Django DRF IsSuperUser + Angular AdminGuard",
              "Platform stats, user mgmt, system health"),
         ],
         "Table 4.1: Modules Completed Summary (13 Modules)", {})

    # 13 module narratives
    modules = [
        ("4.1.2.1", "User Authentication Module",
         "User Authentication Flow", "4.1",
         "Delivers sign-up, sign-in and self-service password reset "
         "on Django REST Framework with SimpleJWT; Bcrypt hashes "
         "passwords, Fernet encrypts sensitive profile fields, and "
         "short-lived access tokens plus rotating refresh tokens "
         "protect against replay."),
        ("4.1.2.2", "Age Prediction \u2013 Single Face Module",
         "Age Prediction Interface", "4.2",
         "The most-exercised pathway: YOLOv8 (with RetinaFace "
         "fallback) locates the face, MiVOLO v2 and InsightFace are "
         "invoked in parallel, and weighted-ensemble averaging "
         "produces an annotated age plus a confidence score on the "
         "returned image."),
        ("4.1.2.3", "Multi-Face Detection and Group Analysis Module",
         "MiVOLO v2 Prediction Pipeline", "4.3",
         "Sweeps YOLOv8 over entire group photos, falls back to "
         "RetinaFace for small or non-frontal faces, and returns "
         "per-face age, gender and emotion in a single JSON payload "
         "that the frontend renders as a scrollable list of cropped "
         "faces."),
        ("4.1.2.4", "Batch Prediction Module (NEW)",
         "Batch Prediction Module Output", "4.7",
         "A new POST /api/v1/predict/batch endpoint runs every "
         "uploaded image through the ensemble and persists a "
         "batch_predictions document in MongoDB; the Angular drag-"
         "drop component renders a progress indicator and a "
         "thumbnail results grid for forensic and academic "
         "workflows."),
        ("4.1.2.5", "Emotion Detection Module",
         "Emotion Detection Output", "4.3",
         "A trpakov/vit-face-expression Vision Transformer, loaded "
         "once at start-up, reuses the YOLOv8 face crops from age "
         "prediction and returns one of seven canonical emotion "
         "classes merged into each per-face result without extra "
         "detection cost."),
        ("4.1.2.6", "Multi-Engine Age Progression Module",
         "SAM GAN Progression Flow", "4.4",
         "Three selectable engines \u2014 SAM GAN (pSp + StyleGAN2 "
         "in W+ latent space), Fast-AgingGAN (CPU-friendly "
         "CycleGAN), and FADING (Stable Diffusion with null-text "
         "inversion, optionally offloaded to Modal) \u2014 each "
         "gated by a 0.6 FaceNet identity similarity threshold."),
        ("4.1.2.7", "Progress Stream Module (NEW)",
         "FADING Diffusion Progression Flow", "4.5",
         "Long diffusion jobs expose percent-complete updates over "
         "GET /api/v1/progress/stream using Django "
         "StreamingHttpResponse; the Angular client consumes the "
         "Server-Sent Events via the browser EventSource API and "
         "renders a live progress bar inside the progression card."),
        ("4.1.2.8",
         "Prediction and Progression History Module",
         "Side-by-Side Age Progression Output", "4.6",
         "Predictions, progressions and batch results are stored in "
         "three MongoDB collections; the Angular history view "
         "serves a paginated, userId-filtered slice with thumbnails, "
         "a download button and per-entry deletion, ensuring strict "
         "inter-account data isolation."),
        ("4.1.2.9", "Analytics Dashboard Module",
         "User Dashboard Interface", "4.8",
         "A single aggregation endpoint returns per-user counts, "
         "age distributions, gender ratios, emotion histograms and "
         "an activity timeline, which Chart.js renders as four "
         "dashboard visualisations in well under one second per "
         "request."),
        ("4.1.2.10", "User Settings and Preferences Module",
         "User Settings Page", "4.8",
         "A per-user MongoDB settings document captures theme, "
         "preferred age model (MiVOLO v2 / InsightFace / Ensemble), "
         "preferred progression engine (SAM_GAN / Fast_AgingGAN / "
         "FADING_Diffusion) and notification opt-in; a JSON schema "
         "validates every update server-side before persistence."),
        ("4.1.2.11", "Dashboard Module",
         "User Dashboard Interface", "4.8",
         "The post-login landing page summarises lifetime "
         "prediction and progression totals, lists the five most "
         "recent results with thumbnails and exposes \u201Cquick "
         "analyse\u201D shortcuts that deep-link into the "
         "prediction and progression flows."),
        ("4.1.2.12", "Real-Time Camera Prediction Module",
         "Real-Time Camera Prediction Interface", "4.9",
         "An Angular getUserMedia stream copies frames to an off-"
         "screen canvas every 1500 ms and posts each to "
         "predict/camera; the response renders live as an overlay "
         "of bounding boxes and ages, doubling as a demo and a "
         "quality-control tool."),
        ("4.1.2.13", "Admin Panel Module (NEW)",
         "Admin Panel \u2013 User Management & System Health", "4.10",
         "Gated by IsSuperUser (backend) and AdminGuard "
         "(frontend), the admin panel exposes six endpoints \u2014 "
         "stats, user search, detail, suspend, reinstate, system "
         "health \u2014 and renders Dashboard, Users and Health "
         "tabs backed by MongoDB aggregations."),
    ]
    fig_image_map = {
        "4.1": "fig_4_1_auth_flow.png",
        "4.2": "fig_4_2_age_prediction.png",
        "4.3": "fig_4_3_mivolo_pipeline.png",
        "4.4": "fig_4_4_sam_gan_flow.png",
        "4.5": "fig_4_5_fading_flow.png",
        "4.6": "fig_4_6_side_by_side.png",
        "4.7": "fig_4_7_batch_predict.png",
        "4.8": "fig_4_8_dashboard.png",
        "4.9": "fig_4_9_camera.png",
        "4.10": "fig_4_10_admin_panel.png",
    }
    for sec_num, title, screenshot, fig_num, body in modules:
        _add(out, "section", sec_num, title, {"level": 3})
        _add(out, "para", body)
        _add(out, "figure",
             f"Figure {fig_num}: {screenshot}",
             {"image": fig_image_map.get(fig_num, "")})

    _add(out, "section", "4.2", "Results", {"level": 1})

    _add(out, "section", "4.2.1", "Deviations and Justifications",
         {"level": 2})
    _add(out, "table",
         ["#", "Planned Design", "Final Implementation", "Justification"],
         [
             ("D-01",
              "EfficientNet-B0 as primary age model",
              "MiVOLO v2 ensembled with InsightFace buffalo_l",
              "MiVOLO v2 reduces MAE from ~6 yr to ~3.65 yr by "
              "leveraging body context (YOLOv8) alongside face features."),
             ("D-02",
              "OpenCV Haar Cascade + DNN face detection",
              "YOLOv8 (ultralytics) primary with RetinaFace fallback",
              "YOLOv8 detects small/non-frontal faces with significantly "
              "higher recall."),
             ("D-03",
              "Single GAN engine (Age-cGAN / StarGAN)",
              "Three engines: SAM GAN + Fast-AgingGAN + FADING Diffusion",
              "Diffusion (FADING) provides bidirectional aging at "
              "higher visual quality; SAM and Fast-AgingGAN remain "
              "for low-latency and CPU-only cases."),
             ("D-04",
              "MongoDB-only persistence",
              "Hybrid MongoDB + SQLite, 6 MongoDB collections, "
              "MongoUserManager / MongoPredictionManager / "
              "MongoProgressionManager / MongoPasswordResetManager",
              "Django auth integrates cleanly with SQLite, leaving "
              "MongoDB for flexible AI-result documents and admin "
              "aggregations."),
             ("D-05",
              "Single emotion model from FER2013",
              "trpakov/vit-face-expression Vision Transformer",
              "Pretrained ViT outperforms CNN baselines on the same "
              "FER2013 split and avoids local re-training."),
             ("D-06",
              "Built-in batch processing (deferred at second review)",
              "Delivered as the new /predict/batch/ endpoint plus "
              "Angular /pages/batch-predict component with batch_"
              "predictions MongoDB collection",
              "Resolved before the final review; the batch flow is "
              "now production-ready."),
             ("D-07",
              "Admin panel with platform analytics (deferred at "
              "second review)",
              "Delivered as 6 admin endpoints (dashboard, users, "
              "user detail, suspend, reinstate, system health) gated "
              "by IsSuperUser + AdminGuard",
              "Resolved before the final review; superusers now have "
              "full visibility and user-management tools."),
             ("D-08",
              "Cloud-deployed production instance",
              "Local development deployment for review; Modal cloud "
              "GPU integrated for FADING diffusion offload",
              "Hybrid local + serverless cloud architecture validated; "
              "full cloud deployment remains the next roadmap step."),
             ("D-09",
              "Full PDF export of reports",
              "Project final report itself is exportable as DOCX and "
              "PDF; in-app PDF report deferred to future work",
              "PDF export of the academic report (this document) is "
              "covered; per-result PDF export remains future work."),
             ("D-10",
              "Single training pipeline (only inference initially)",
              "Two reproducible training pipelines bundled: MiVOLO "
              "v2 (mivolo/train.py + Colab) and SAM GAN (sam/train.py "
              "+ Colab + Kaggle)",
              "Researchers can now retrain both engines on local "
              "datasets without leaving the repo."),
         ],
         "Table 4.2: Deviations and Justifications", {})

    _add(out, "section", "4.2.2", "Project Roadmap and Status",
         {"level": 2})
    _add(out, "table",
         ["Phase", "Activity", "Status"],
         [
             ("Phase 1",
              "Requirement gathering, literature review, tech-stack "
              "selection", "Completed"),
             ("Phase 2",
              "Frontend skeleton (Angular 19), backend skeleton "
              "(Django 5.2 + DRF), authentication module",
              "Completed"),
             ("Phase 3",
              "Age prediction pipeline (MiVOLO v2 + InsightFace "
              "ensemble), single + multi-face support, emotion "
              "detection", "Completed"),
             ("Phase 4",
              "GAN age progression (SAM GAN + Fast-AgingGAN), "
              "identity preservation, side-by-side rendering",
              "Completed"),
             ("Phase 5",
              "History, analytics, settings, dashboard, real-time "
              "camera prediction", "Completed"),
             ("Phase 6",
              "Test catalogue execution, second review submission",
              "Completed"),
             ("Phase 7",
              "Batch prediction module, FADING diffusion engine, "
              "Modal cloud GPU integration, admin panel with user "
              "management and system health, MongoDB manager refactor, "
              "MiVOLO + SAM training pipelines",
              "Completed"),
             ("Phase 8",
              "Final report preparation (DOCX + PDF), final review "
              "submission",
              "Completed"),
         ],
         "Table 4.3: Project Roadmap and Phase Status", {})

    _add(out, "section", "4.2.3", "Results Summary", {"level": 2})
    _add(out, "para",
         "The MiVOLO v2 + InsightFace ensemble achieves a mean "
         "absolute error of approximately 3.65 years on the held-out "
         "UTKFace + IMDB-WIKI test split, down from a 6\u20137 year "
         "EfficientNet-B0 baseline. All three progression engines "
         "preserve identity above the FaceNet 0.6 threshold on over "
         "95% of evaluation samples; SAM GAN meets the 3\u20138 s "
         "latency budget. Every module in Table 4.1 is complete; "
         "deviations in Table 4.2 are informed upgrades (D-01\u2013"
         "D-05) or resolved deferrals (D-06, D-07); full cloud "
         "deployment (D-08) and PDF export (D-09) remain as future "
         "work.")

    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                CHAPTER V                                    #
# --------------------------------------------------------------------------- #

def chapter_5(out):
    _add(out, "heading_centered", "CHAPTER 5",
         {"size": 14, "bold": True, "all_caps": True, "space_before": 56})
    _add(out, "heading_centered", "CONCLUSION AND FUTURE WORK",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "section", "5.1", "Summary", {"level": 1})
    _add(out, "para",
         "AgeVision delivers a complete, reproducible, full-stack "
         "reference implementation of an AI-powered face-analysis "
         "platform that unifies age prediction and identity-preserving "
         "age progression into a single web application. On the "
         "prediction side the MiVOLO v2 + InsightFace ensemble achieves "
         "a mean absolute error of approximately 3.65 years on standard "
         "benchmarks, comfortably below the 4\u20137 year plateau reported "
         "for single-model commercial systems. On the progression side, "
         "three engines now ship together: SAM GAN for fast identity-"
         "preserving aging, Fast-AgingGAN for CPU-only environments, "
         "and the newly added FADING diffusion pipeline for the highest-"
         "quality bidirectional results \u2014 with optional Modal "
         "cloud GPU offload.")
    _add(out, "para",
         "Between the second and final reviews, four significant "
         "modules were added: batch prediction (solving the forensic "
         "bulk-processing gap), the FADING diffusion progression engine "
         "with cloud offload, a Server-Sent Events progress stream for "
         "long-running jobs, and a full admin panel with user "
         "management and live system health. Two reproducible training "
         "pipelines \u2014 MiVOLO v2 and SAM GAN \u2014 were also "
         "bundled, each with Colab and Kaggle notebooks, turning the "
         "repository into a ready-to-extend research artefact. In "
         "total, twenty-one REST endpoints, six MongoDB collections and "
         "thirteen functional modules have been implemented, documented "
         "and validated against a test catalogue that now includes a "
         "dedicated Admin Panel test table (TC-AD01 \u2192 TC-AD07) "
         "alongside the existing authentication, prediction, batch, "
         "progression, history, analytics and settings cases.")
    _add(out, "para",
         "Known limitations of the current implementation are:")
    _add(out, "bullet", [
        "Dependency on HuggingFace cloud for ViT and diffusers weights "
        "on first run; an offline-first packaging step is needed for "
        "air-gapped deployments.",
        "FADING diffusion inference on CPU or a consumer GPU takes "
        "12\u201320 seconds per image; Modal cloud GPU offload is "
        "optional and requires an account.",
        "Image quality sensitivity \u2014 low-resolution or heavily "
        "occluded faces reduce prediction accuracy.",
        "Local file storage for uploaded images limits horizontal "
        "scalability and forces sticky sessions.",
        "Language and ethnicity bias: SAM GAN was trained primarily on "
        "FFHQ (Western faces); the Indian-face SAM checkpoint and the "
        "diffusion engine mitigate this but do not eliminate it.",
        "No native mobile application; mobile usage relies on the "
        "responsive web UI.",
        "Admin panel operates on per-request queries today \u2014 "
        "pre-aggregated caches will be needed at higher user counts.",
    ])

    _add(out, "section", "5.2", "Future Work", {"level": 1})
    _add(out, "bullet", [
        "Full production cloud deployment (AWS EC2 G4dn / GCP T4) "
        "with a Celery worker queue for batch and diffusion jobs.",
        "Per-result PDF export and mobile application (React Native / "
        "Flutter) consuming the existing REST API.",
        "Visual explainability via Grad-CAM heatmaps and multi-language "
        "UI support (Tamil, Hindi, Spanish).",
        "Integration adapters for forensic case-management platforms "
        "and pre-aggregated admin analytics caches.",
        "Model registry with automated retraining triggered by new "
        "uploaded batches, using the bundled MiVOLO and SAM "
        "training pipelines.",
    ])

    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                REFERENCES                                   #
# --------------------------------------------------------------------------- #

def references(out):
    _add(out, "heading_centered", "REFERENCES",
         {"size": 14, "bold": True, "all_caps": True})
    _add(out, "references", REFERENCES)
    _add(out, "pagebreak")


# --------------------------------------------------------------------------- #
#                                APPENDICES                                   #
# --------------------------------------------------------------------------- #

def appendices(out):
    _add(out, "heading_centered", "APPENDICES",
         {"size": 14, "bold": True, "all_caps": True})

    _add(out, "sub_title", "APPENDIX 1: SYSTEM INSTALLATION AND SETUP GUIDE")
    _add(out, "para",
         "Prerequisites: Python 3.11+, Node.js 20 LTS, MongoDB 4.6, "
         "Git; optionally a CUDA-capable GPU and a Modal account for "
         "FADING diffusion cloud offload.")
    _add(out, "sub_heading", "Backend Setup")
    _add(out, "numbered", [
        "Clone the backend repository: git clone <repository-url> "
        "agevision_backend",
        "Create a Python virtual environment: python -m venv venv",
        "Activate it: source venv/bin/activate (Linux/Mac) or "
        "venv\\Scripts\\activate (Windows)",
        "Install dependencies: pip install -r requirements.txt",
        "Configure environment variables: MONGODB_URI, JWT_SECRET, "
        "MIVOLO_MODEL_PATH, SAM_CHECKPOINT_PATH, DIFFUSION_MODEL_PATH, "
        "MODAL_TOKEN_ID (optional), MODAL_TOKEN_SECRET (optional)",
        "Apply migrations: python manage.py migrate",
        "Create a superuser for the admin panel: python manage.py "
        "createsuperuser",
        "Start the server: python manage.py runserver 0.0.0.0:8000",
    ])
    _add(out, "sub_heading", "Frontend Setup")
    _add(out, "numbered", [
        "Clone the frontend repository: git clone <repository-url> "
        "agevision-frontend",
        "Install dependencies: npm install",
        "Configure src/environments/environment.ts with apiUrl: "
        "http://localhost:8000/api/v1",
        "Start the development server: ng serve",
    ])
    _add(out, "sub_heading", "Optional: Modal Cloud GPU Setup")
    _add(out, "numbered", [
        "Install Modal CLI: pip install modal>=0.73.0",
        "Authenticate: modal token new",
        "Deploy the FADING diffusion function from agevision_backend/"
        "agevision_api/diffusion_aging/",
        "Set cloud=true when calling /api/v1/progress/generate with "
        "ganModel='diffusion'.",
    ])

    _add(out, "sub_title",
         "APPENDIX 2: ALGORITHM IMPLEMENTATION DETAILS")

    _add(out, "section", "A2.1",
         "MiVOLO v2 + InsightFace Ensemble", {"level": 2})
    _add(out, "para",
         "Each face crop is forwarded to MiVOLO v2 and InsightFace in "
         "parallel. The final age is a weighted average where MiVOLO "
         "carries weight 0.7 (lower MAE on benchmark) and InsightFace "
         "carries weight 0.3. The ensemble confidence is 1 \u2212 (\u03C3 / "
         "max_age) where \u03C3 is the standard deviation of the two "
         "predictions and max_age is set to 100. The same ensemble "
         "feeds into the per-face confidence reported in the response "
         "payload.")

    _add(out, "section", "A2.2",
         "SAM GAN Pipeline", {"level": 2})
    _add(out, "para",
         "The pSp encoder maps the source image into the W+ latent "
         "space of a pretrained StyleGAN2 generator. An age-condition "
         "vector corresponding to the user-specified target age is "
         "concatenated and decoded through the StyleGAN2 decoder to "
         "produce the aged face. FaceNet similarity is then computed "
         "between the source and progressed embeddings; outputs with a "
         "score below 0.6 are flagged as low identity preservation.")

    _add(out, "section", "A2.3",
         "FADING Diffusion Pipeline (NEW)", {"level": 2})
    _add(out, "para",
         "The FADING (FAce age-editing via Diffusion-based INversion "
         "and Guidance) pipeline first computes a null-text inversion "
         "of the source image against the Stable Diffusion v1.5 "
         "backbone, producing a sequence of noise latents that "
         "reconstruct the original when paired with an empty prompt. "
         "A classifier-free guidance step then applies an age-specific "
         "text prompt (e.g. \u201Cphoto of a 60-year-old person\u201D) "
         "during the denoising loop, pushing the image toward the "
         "requested target age while retaining the inverted identity. "
         "When cloud offload is enabled the entire loop runs on a Modal "
         "A10G function; otherwise it executes locally on the first "
         "available CUDA device, falling back to CPU if none is "
         "detected. Identity preservation is verified by the same "
         "FaceNet threshold (0.6) that guards the GAN engines.")

    _add(out, "section", "A2.4",
         "Admin Aggregations and Permission Gate (NEW)", {"level": 2})
    _add(out, "para",
         "Admin endpoints are gated by the IsSuperUser DRF permission "
         "(agevision_api/permissions.py). Aggregation methods on "
         "MongoPredictionManager and MongoProgressionManager \u2014 "
         "platform_stats, platform_detector_breakdown, "
         "platform_gender_distribution, platform_model_breakdown and "
         "platform_daily_counts \u2014 run lightweight MongoDB "
         "aggregation pipelines ($match + $group + $project) to "
         "compute the dashboard metrics with a single round-trip each.")


# --------------------------------------------------------------------------- #
#                                    BUILD                                    #
# --------------------------------------------------------------------------- #

def build() -> List[Tuple]:
    out: List[Tuple] = []
    cover_page(out)
    bonafide(out)
    viva_voce(out)
    acknowledgement(out)
    table_of_contents(out)
    list_of_tables(out)
    list_of_figures(out)
    abstract(out)
    abstract_tamil(out)
    # Transition from Roman-numbered prelims to Arabic-numbered main text.
    _add(out, "section_break")
    chapter_1(out)
    chapter_2(out)
    chapter_3(out)
    chapter_4(out)
    chapter_5(out)
    references(out)
    appendices(out)
    return out


if __name__ == "__main__":
    cmds = build()
    print(f"Content tree: {len(cmds)} instructions")
