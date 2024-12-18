from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure

def watch_changes():
    uri = "mongodb+srv://ghitaoudrhiri02:ghitaoudrhiri02@cluster0.fllnb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["AGILE"]
    collection = db["ratings"]

    print("Listening for changes...")
    try:
        with collection.watch() as stream:
            for change in stream:
                print("Change detected:", change)
                # Call Azure Function here
                trigger_azure_function(change)
    except ConnectionFailure as e:
        print(f"Error: {e}")

def trigger_azure_function(change_data):
    import requests
    azure_function_url = "https://<your-azure-function-url>"
    response = requests.post(azure_function_url, json=change_data)
    print("Azure Function triggered:", response.status_code, response.text)

if __name__ == "__main__":
    watch_changes()