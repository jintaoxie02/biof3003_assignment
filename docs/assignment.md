# BIOF3003 Digital Health Technology - Assignment Documentation

## Project Overview

HeartLens is a web-based application that uses photoplethysmography (PPG) through a webcam to measure heart rate and heart rate variability (HRV) in real-time. The application provides a user-friendly interface for monitoring vital signs and storing historical data.

## Technical Implementation

### Frontend Implementation

The frontend is built using Next.js 14 with TypeScript and includes the following components:

1. **User Panel**
   - Input field for subject ID
   - Confirmation button
   - Real-time status display

2. **Camera Feed**
   - Webcam integration using React Webcam
   - Real-time video display
   - Frame processing for PPG analysis

3. **Data Visualization**
   - Real-time heart rate display
   - HRV monitoring
   - Historical data chart using Chart.js

4. **UI Design**
   - Responsive layout using Tailwind CSS
   - Modern and clean interface
   - Error handling and user feedback

### Backend Implementation

The backend is built using Express.js with TypeScript and includes:

1. **API Endpoints**
   - GET /api/records - Fetch historical data
   - POST /api/records - Store new measurements

2. **Database Integration**
   - MongoDB for data storage
   - Mongoose for data modeling
   - Efficient querying and data retrieval

3. **Data Processing**
   - Real-time data validation
   - Error handling
   - Data formatting and normalization

## Features

### 1. User Panel
- Subject ID input and validation
- User confirmation system
- Session management

### 2. Real-time Monitoring
- Webcam-based PPG measurement
- Heart rate calculation
- HRV analysis
- Real-time data display

### 3. Historical Data
- Data storage in MongoDB
- Historical data retrieval
- Data visualization
- Time-series analysis

### 4. User Interface
- Responsive design
- Intuitive navigation
- Error handling
- Loading states

## Technical Challenges

1. **PPG Signal Processing**
   - Noise reduction
   - Signal filtering
   - Peak detection

2. **Real-time Data Processing**
   - Efficient frame processing
   - Data synchronization
   - Performance optimization

3. **Data Storage and Retrieval**
   - Efficient database queries
   - Data validation
   - Error handling

## Future Improvements

1. **Enhanced Signal Processing**
   - Advanced filtering algorithms
   - Machine learning for better accuracy
   - Multi-signal analysis

2. **User Features**
   - User authentication
   - Multiple subject support
   - Data export functionality

3. **Visualization**
   - Advanced charting options
   - Customizable views
   - Export capabilities

## Conclusion

HeartLens demonstrates the successful implementation of a web-based PPG monitoring system. The application provides real-time heart rate and HRV monitoring with historical data storage and visualization. The project showcases modern web development practices and biomedical signal processing techniques.

## References

1. Photoplethysmography (PPG) Signal Processing
2. WebRTC and Webcam Integration
3. MongoDB Documentation
4. Next.js Documentation
5. Chart.js Documentation 