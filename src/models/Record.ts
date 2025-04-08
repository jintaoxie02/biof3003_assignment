import mongoose from 'mongoose';

const RecordSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
  },
  heartRate: {
    type: Number,
    required: true,
  },
  hrv: {
    type: Number,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const Record = mongoose.models.Record || mongoose.model('Record', RecordSchema);

export default Record; 