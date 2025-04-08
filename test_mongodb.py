import pymongo
import pandas as pd

# Connect to MongoDB Atlas
client = pymongo.MongoClient("mongodb+srv://biof3003digitalhealth01:qoB38jemj4U5E7ZL@cluster0.usbry.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Connect to the test database and records collection
db = client["test"]
collection = db["records"]

# Get the data
data = list(collection.find({}))
df = pd.DataFrame(data)

print(f"Successfully connected to MongoDB Atlas!")
print(f"Found {len(df)} records in the database")
print("\nColumns in the data:")
print(df.columns.tolist())
print("\nFirst record:")
print(df.iloc[0]) 