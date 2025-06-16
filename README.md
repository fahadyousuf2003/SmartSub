# Football Substitution Recommendation System

This project is a football substitution recommendation system that analyzes football match videos and provides substitution recommendations based on player performance metrics.

## Project Structure

The project consists of two main components:

1. **Backend**: A Python FastAPI server that handles video processing, analysis, and recommendation generation
2. **Frontend**: A React TypeScript web application that provides the user interface

## Setup and Installation

### Backend Setup

1. Navigate to the Backend directory:
   ```
   cd Backend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```
   python main.py
   ```
   
   The backend server will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the required dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```
   
   The frontend application will run on http://localhost:5173

## Usage

1. Open the web application in your browser at http://localhost:5173
2. Navigate to the home page by clicking "Get Started" or going to /home
3. Input your team name and opponent team name
4. Select the parameters for analysis (Speed, Distance Covered, Pass Accuracy)
5. Upload your match video
6. Click "Start Analysis" to process the video
7. Wait for processing to complete
8. View the processed video and substitution recommendations

## System Workflow

1. User uploads a video file through the frontend
2. User inputs team names and selects analysis parameters
3. Frontend sends the data to the backend
4. Backend processes the video:
   - Track player movements
   - Assign players to teams
   - Calculate performance metrics
   - Generate substitution recommendations
5. Backend returns a processed video and recommendations
6. Frontend displays the results to the user

## Technologies Used

- **Backend**:
  - Python
  - FastAPI
  - OpenCV
  - NumPy
  - MongoDB
  - Computer Vision & AI algorithms

- **Frontend**:
  - React
  - TypeScript
  - Vite
  - CSS/HTML

## Important Notes

- Processing large video files may take a significant amount of time
- The system works best with high-quality, stable camera footage
- For optimal results, use videos where teams have distinctly colored uniforms 