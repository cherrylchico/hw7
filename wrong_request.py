import requests
import json

if __name__ == "__main__":
    url = "http://127.0.0.1:8000/predict"
    try:
        # Load input JSON
        with open('wrong_sample.json') as f:
            sample_data = json.load(f)

        ## Uncomment below to see the data being sent
        # print("Sending data:", json.dumps(sample_data, indent=2))

        # Send POST request with timeout and handle response
        try:
            response = requests.post(url, json=sample_data, timeout=10)
        except requests.exceptions.ConnectionError:
            print("[ERROR] Could not connect to the server. Is it running?")
            raise
        except requests.exceptions.Timeout:
            print("[ERROR] Request timed out after 10 seconds.")
            raise
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            raise


        # Print server response
        print("Response:", response.text)

        # Handle non-200 responses (as discussed in class)
        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", "No detail provided")
            except ValueError:
                error_detail = "Server returned non-JSON error response."
            print(f"[ERROR] Status code {response.status_code}. Details:", error_detail)
        else:
            # Successful case: parse JSON and print prediction
            try:
                data = response.json()
                prediction = data.get("prediction", None)
                if prediction is not None:
                    print("YES! Prediction from API:", prediction)
                else:
                    print("NO! Response JSON missing 'prediction' field.")
            except ValueError:
                print("[ERROR] Could not parse JSON response.")

    except FileNotFoundError:
        print("[ERROR] Input file 'sample.json' not found.")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format in input file: {e}")
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}")
