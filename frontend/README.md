# Time-Series Forecasting Frontend

React frontend application for the Time-Series Forecasting API.

## Features

- 🎯 Model selection (LSTM or XGBoost)
- 📊 Real-time predictions
- 💚 Health and readiness status indicators
- 🎨 Modern, responsive UI

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Backend API running on `http://localhost:8000`

## Installation

```bash
cd frontend
npm install
```

## Development

Start the development server:

```bash
npm start
```

The app will open at `http://localhost:3000`.

## Configuration

The frontend connects to the API at `http://localhost:8000` by default. To change this, create a `.env` file in the `frontend` directory:

```
REACT_APP_API_URL=http://your-api-url:8000
```

## Production Build

Build the production bundle:

```bash
npm run build
```

The build folder will contain the optimized production build.

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── PredictionCard.jsx
│   │   ├── PredictionCard.css
│   │   ├── StatusIndicator.jsx
│   │   └── StatusIndicator.css
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── App.css
│   ├── index.jsx
│   └── index.css
├── package.json
└── README.md
```
