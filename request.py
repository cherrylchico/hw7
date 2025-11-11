import requests
import json

if __name__ == "__main__":
    url = "http://127.0.0.1:8000/predict"
    try:
        with open('sample.json') as f:
            sample_data = json.load(f)
        ## Uncomment below to see the data being sent
        # print("Sending data:", json.dumps(sample_data, indent=2))
        response = requests.post(url, json=sample_data)

        ## Uncomment below to see the full response details
        # print(f"Status Code: {response.status_code}")
        print("Response:", response.text)
        if response.status_code != 200:
            print("Error details:", response.json().get('detail', 'No detail provided'))
    except Exception as e:
        print(f"Error: {e}")
