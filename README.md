# BIOF3003 Digital Health Technology - Frontend Assignment

This is the frontend application for the BIOF3003 Digital Health Technology course assignment. It includes a user panel, camera feed, and data visualization features.

## Features

- User panel for managing user information
- Camera feed integration
- Historical data visualization using charts
- MongoDB integration for data storage
- Backend API for data processing and storage

## Prerequisites

- Node.js (v18 or later)
- MongoDB (local or Atlas)
- npm or yarn
- Python 3.8 or later (for backend)
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heartlens-frontend
```

2. Install frontend dependencies:
```bash
npm install
# or
yarn install
```

3. Install backend dependencies:
```bash
cd heartlens-frontend/biof3003-backend
# Clean any existing node_modules
rm -rf node_modules package-lock.json
# Install dependencies
npm install
# or
yarn install
```

4. Create a `.env.local` file in the frontend root directory with:
```
MONGODB_URI=your_mongodb_connection_string
```

5. Create a `.env` file in the backend directory with:
```
MONGODB_URI=your_mongodb_connection_string
PORT=3003
```

## Running the Application

### Backend Setup

1. Navigate to the backend directory:
```bash
cd heartlens-frontend/biof3003-backend
```

2. Start the backend server:
```bash
npm run dev
# or
yarn dev
```

The backend will run on http://localhost:3003

### Frontend Setup

1. Open a new terminal window
2. Navigate to the frontend directory:
```bash
cd heartlens-frontend
```

3. Start the frontend development server:
```bash
npm run dev
# or
yarn dev
```

The frontend will run on http://localhost:3002

## Troubleshooting

If the application doesn't start properly, check the following:

### Port Already in Use
If you see "Error: listen EADDRINUSE: address already in use" or similar:
1. Find the process using the port:
   ```bash
   lsof -i :3002  # For frontend
   lsof -i :3003  # For backend
   ```
2. Kill the process:
   ```bash
   kill -9 <PID>
   ```
   where `<PID>` is the process ID from the previous command

### Backend Issues
1. Verify you're in the correct directory (`heartlens-frontend/biof3003-backend`)
2. If you see dependency errors:
   ```bash
   cd heartlens-frontend/biof3003-backend
   rm -rf node_modules package-lock.json
   npm install
   ```
3. Ensure all backend dependencies are installed
4. Check that the MongoDB connection string in `.env` is correct
5. Verify port 3003 is not being used by another application
6. Check the terminal for any backend error messages

### Frontend Issues
1. Verify you're in the correct directory (`heartlens-frontend`)
2. Ensure all frontend dependencies are installed
3. Check that the MongoDB connection string in `.env.local` is correct
4. Verify port 3002 is not being used by another application
5. Check the terminal for any frontend error messages

### Common Issues
1. Make sure both frontend and backend are running simultaneously
2. Verify that MongoDB is running and accessible
3. Check that Node.js v18 or later is installed
4. Ensure all environment variables are properly set
5. Check network connectivity between frontend and backend

## Project Structure

```
heartlens-frontend/
├── biof3003-backend/     # Backend server
│   ├── src/             # Backend source code
│   ├── models/          # Data models
│   └── routes/          # API routes
├── src/                 # Frontend source code
│   ├── app/            # Next.js app directory
│   ├── components/     # React components
│   ├── lib/           # Utility functions
│   └── types/         # TypeScript definitions
└── public/            # Static assets
```

## Technologies Used

### Frontend
- Next.js 14
- React 18
- TypeScript
- MongoDB
- Mongoose
- Chart.js
- Tailwind CSS

### Backend
- Node.js
- Express
- MongoDB
- Mongoose
- TypeScript
- ml-random-forest (for machine learning)
- fft-js (for signal processing)
- Custom peak detection algorithm

## License

This project is licensed under the MIT License. 