## Repository Content

PlantEdibilityClassification-/  
├── backend/  
│   ├── Dockerfile  
│   ├── requirements.txt  
│   └── server.py  
├── data/  
│   ├── edible_plants/  
│   └── non_edible_plants/  
├── frontend/  
│   └── public/  
│       ├── index.html  
│       └── styles.css  
├── output/  
│   ├── predictions/  
│   └── logs/  
├── scripts/  
│   └── main.py  
├── .gitignore  
├── README.md  
├── package.json  
└── server.js  

## UI Setup Instructions

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

