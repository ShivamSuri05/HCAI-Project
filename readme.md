# Human Centric Artificial Intelligence PBL

This repository contains **5 Django projects** developed as part of the **Human Centric AI** course.  
All projects are contained in a single Django setup and can be accessed through different URLs.

---

## 📂 Repository Structure

```
 
Human-Centric-AI/
 ├── demos/      # Demo files and examples
 ├── home/       # Home Page assets
 ├── pbl/        # Project Settings
 ├── project1/   # Project 1: Supervised Learning Interface
 ├── project2/   # Project 2: Active Learning for Text Classification
 ├── project3/   # Project 3: Decision trees and Counterfactual explanations
 ├── project4/   # Project 4: Influence of future predictions over active learning of users’ tastes for recommender systems
 ├── project5/   # Project 5: Reinforcement learning with human feedback
 ├── static/     # Shared static files (CSS, JS, images)
 ├── templates/  # Shared HTML templates
 ├── media/      # Uploaded media files
 ├── .gitignore  # Git ignore configuration
 ├── install_packages.py  # Helper script for package installation
 ├── manage.py   # Django project management script
 ├── readme.md   # readme documentation
 ├── requirements.txt  # Python dependencies
 
```

---

## ⚡ Prerequisites

- Python **3.9.x** or **3.10.x**  
- pip  
- virtualenv (recommended)  
- Django **4.x**  

---

## 🚀 Setup Instructions

1. **Clone the repository**  
   - Run: `git clone https://github.com/ShivamSuri05/HCAI-Project.git`  
   - Change directory: `cd HCAI-Project`  

2. **Create a Virtual Environment**  
   - Run: `python -m venv venv`  
   - Activate (Linux/Mac): `source venv/bin/activate`  
   - Activate (Windows): `venv\Scripts\activate`  

> ⚠️ **Attention – Graphviz Required**  
> Some projects (e.g., decision tree visualizations) require Graphviz.  
> - Download Graphviz (64-bit) from [https://graphviz.org/download/](https://graphviz.org/download/)  
> - Recommended version: **graphviz-13.0.1 (64-bit EXE installer)**  
> - Add `graphviz/bin` to your system's environment variable PATH  

> ⚠️ **Warning – Antivirus / Windows Defender**  
> Graphviz may require whitelisting:  
> - Allow `graphviz/bin/dot.exe` in Windows Defender / Antivirus software  

3. **Install Dependencies**  
   - Run: `python install_packages.py`  

4. **Apply Database Migrations**  
   - Run: `python manage.py makemigrations`  
   - Run: `python manage.py migrate`  

5. **Run the Server**  
   - Run: `python manage.py runserver`  

6. **Access the Projects**
- HomePage: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
- Project 1: [http://127.0.0.1:8000/project1/index](http://127.0.0.1:8000/project1/index)  
- Project 2: [http://127.0.0.1:8000/project2/index](http://127.0.0.1:8000/project2/index)  
- Project 3: [http://127.0.0.1:8000/project3/index](http://127.0.0.1:8000/project3/index)  
- Project 4: [http://127.0.0.1:8000/project4/index](http://127.0.0.1:8000/project4/index)  
- Project 5: [http://127.0.0.1:8000/project5/index](http://127.0.0.1:8000/project5/index)  

---

## 📚 Project Descriptions

- **Project 1:** Supervised Learning Interface  
- **Project 2:** Active Learning for Text Classification  
- **Project 3:** Decision trees and Counterfactual explanations  
- **Project 4:** Influence of future predictions over active learning of users’ tastes for recommender systems  
- **Project 5:** Reinforcement learning with human feedback  

---

## 📝 Notes

- All projects share static files (`/static`) and templates (`/templates`)  
- Uploaded files are stored in `/media`  
- Activate the virtual environment before running the server (recommended) 
- If decision tree visualizations don't work, re-check Graphviz installation and PATH setup  

---

## 👨‍💻 Author

**Shivam Suri**  
Master's in Data Science, TUHH  
Course: Human-Centric Artificial Intelligence
