## Repository Content

### Software and Platform
- Programming language: Python
- Software: Jupyter Notebook (for exploratory analysis), Python scripts
- Platforms: Works on Windows, MacOS, and Linux
- See requirements.txt for all required dependencies

### Repository Mapping

PlantEdibilityClassification-/  
│── backend/  
│ ├── data/class_mapping/  
│ │ ├── plantnet300k_species_names.json    
│ ├── scripts/  
│ │ ├── __ init __.py  
│ │ ├── classifier.py  
│ │ ├── describe_plant.py  
│ │ ├── fetch_data.py  
│ │ ├── main.py  
│ │ ├── utils.py  
│ ├── Dockerfile  
│ ├── LICENSE  
│ ├── README.md  
│ ├── docker-compose.yml  
│ ├── package-lock.json  
│ ├── requirements.txt  
│── data/  
│ ├── data appendix.pdf    
│── frontend/public/  
│ ├── index.html  
│ ├── index.js  
│ ├── style.css  
│── output/  
│ ├── genus_counts.cvs  
│ ├── model_comparison.png  
│ ├── species_counts.csv  
│ ├── ui_screenshot.png  
│── scripts/  
│ ├── Analysis_EDA.ipynb  
│ ├── Test_Data_EDA.ipynb  
│ ├── data_appendix_scripts.ipynb  
│── README.md  
│── package.json  
│── server.js  

### UI Setup Instructions

1. **Git clone the repository**
   ```bash
   git clone https://github.com/your-username/PlantEdibilityClassification-.git
   cd PlantEdibilityClassification-
   ```

2. **Open the project in VS Code**

3. **Install frontend dependencies**
   ```bash
   npm install
   ```

4. **Start the frontend**
   ```bash
   npm start
   ```

5. **Run the backend**

   ### Option A: Using Docker
   ```bash
   cd backend
   docker-compose up --build
   ```

   ### Option B: Without Docker
   ```bash
   cd backend
   python3 -m venv myvenv
   source myvenv/bin/activate  # On Windows use: myvenv\Scripts\activate
   pip3 install -r requirements.txt
   cd scripts
   python3 main.py
   ```
6. **Open http://localhost:3000 in browser and play around!**


## (Optional) Add OpenAI Key for Edibility Description 
   1. create .env inside backend directory
   2. add OPENAI_KEY='KEY'

