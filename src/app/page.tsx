'use client';

import { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Home() {
  const [userId, setUserId] = useState('');
  const [heartRate, setHeartRate] = useState<number | null>(null);
  const [hrv, setHrv] = useState<number | null>(null);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);

  const chartData: ChartData<'line'> = {
    labels: historicalData.map(record => new Date(record.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Heart Rate',
        data: historicalData.map(record => record.heartRate),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'HRV',
        data: historicalData.map(record => record.hrv),
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  const handleUserSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Start fetching historical data
    fetchHistoricalData();
    // Start camera if available
    startCamera();
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/records?userId=${userId}`);
      const data = await response.json();
      setHistoricalData(data);
    } catch (error) {
      console.error('Error fetching historical data:', error);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  // Simulate heart rate and HRV updates
  useEffect(() => {
    if (!userId) return;

    const interval = setInterval(() => {
      // Simulate heart rate between 60-100
      const newHeartRate = Math.floor(Math.random() * (100 - 60) + 60);
      // Simulate HRV between 20-80
      const newHrv = Math.floor(Math.random() * (80 - 20) + 20);

      setHeartRate(newHeartRate);
      setHrv(newHrv);

      // Send data to backend
      fetch('http://localhost:3000/api/records', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          heartRate: newHeartRate,
          hrv: newHrv,
        }),
      }).then(() => {
        // Update historical data
        fetchHistoricalData();
      }).catch(error => {
        console.error('Error saving record:', error);
      });
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [userId]);

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">BIOF3003 Digital Health Technology</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* User Panel */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">User Panel</h2>
            <form onSubmit={handleUserSubmit} className="space-y-4">
              <div>
                <label htmlFor="userId" className="block text-sm font-medium text-gray-700">
                  User ID
                </label>
                <input
                  type="text"
                  id="userId"
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  required
                />
              </div>
              <button
                type="submit"
                className="w-full bg-primary text-white py-2 px-4 rounded-md hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
              >
                Start Monitoring
              </button>
            </form>
          </div>

          {/* Camera Feed */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-semibold mb-4">Camera Feed</h2>
            <div className="aspect-video bg-gray-200 rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
            </div>
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-gray-100 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Heart Rate</p>
                <p className="text-2xl font-bold">{heartRate || '--'} BPM</p>
              </div>
              <div className="bg-gray-100 p-4 rounded-lg">
                <p className="text-sm text-gray-600">HRV</p>
                <p className="text-2xl font-bold">{hrv || '--'} ms</p>
              </div>
            </div>
          </div>
        </div>

        {/* Historical Data */}
        <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-semibold mb-4">Historical Data</h2>
          <div className="h-96">
            <Line data={chartData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
      </div>
    </main>
  );
} 