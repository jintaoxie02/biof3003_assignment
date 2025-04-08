import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import recordRoutes from './routes/records.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

console.log('Starting server with configuration:');
console.log('Port:', port);
console.log('MongoDB URI:', process.env.MONGODB_URI || 'mongodb://localhost:27017/biof3003');

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/records', recordRoutes);

// Basic route for testing
app.get('/', (req, res) => {
  res.json({ message: 'Server is running' });
});

// MongoDB Connection
console.log('Attempting to connect to MongoDB...');
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/biof3003')
  .then(() => {
    console.log('Successfully connected to MongoDB at:', process.env.MONGODB_URI || 'mongodb://localhost:27017/biof3003');
    app.listen(port, () => {
      console.log(`Server is successfully running on port ${port}`);
    });
  })
  .catch((error) => {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  });

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Error:', err);
  res.status(500).json({ error: 'Internal server error', details: err.message });
}); 