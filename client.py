# client.py - Client to demonstrate token refresh mechanism
import requests
import time
import json
import sys

API_BASE_URL = "http://localhost:5000"

class TokenRefreshClient:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.access_token_expiry = None
        self.refresh_token_expiry = None
        
    def login(self, username, password):
        """Authenticate and get initial tokens"""
        response = requests.post(
            f"{API_BASE_URL}/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data['access_token']
            self.refresh_token = data['refresh_token']
            
            # Set expiry times
            current_time = time.time()
            self.access_token_expiry = current_time + data['access_token_expires_in']
            self.refresh_token_expiry = current_time + data['refresh_token_expires_in']
            
            print("Login successful!")
            return True
        else:
            print(f"Login failed: {response.text}")
            return False
    
    def refresh_access_token(self):
        """Get a new access token using the refresh token"""
        if not self.refresh_token:
            print("No refresh token available. Please login first.")
            return False
            
        if time.time() > self.refresh_token_expiry:
            print("Refresh token has expired. Please login again.")
            return False
            
        response = requests.post(
            f"{API_BASE_URL}/refresh-token",
            headers={"Authorization": f"Bearer {self.refresh_token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data['access_token']
            
            # Update access token expiry time
            self.access_token_expiry = time.time() + data['expires_in']
            
            print("Access token refreshed successfully!")
            return True
        else:
            print(f"Token refresh failed: {response.text}")
            return False
    
    def ensure_valid_token(self):
        """Check if access token is valid, refresh if needed"""
        # Add a small buffer (30 seconds) to ensure we refresh before expiration
        if not self.access_token or time.time() > (self.access_token_expiry - 30):
            return self.refresh_access_token()
        return True
    
    def make_authenticated_request(self, endpoint, method="GET", data=None, files=None):
        """Make an authenticated request to the API"""
        if not self.ensure_valid_token():
            print("Failed to get a valid token. Please login again.")
            return None
            
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        if method.upper() == "GET":
            response = requests.get(f"{API_BASE_URL}/{endpoint}", headers=headers)
        elif method.upper() == "POST":
            response = requests.post(f"{API_BASE_URL}/{endpoint}", headers=headers, json=data, files=files)
        else:
            print(f"Unsupported method: {method}")
            return None
            
        return response
    
    def get_protected_data(self):
        """Get data from a protected endpoint"""
        response = self.make_authenticated_request("protected-data")
        
        if response and response.status_code == 200:
            return response.json()
        else:
            status = response.status_code if response else "No response"
            print(f"Failed to get protected data. Status: {status}")
            return None
    
    def upload_file(self, file_path):
        """Upload a file for processing"""
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = self.make_authenticated_request("upload-file", method="POST", files=files)
                
                if response and response.status_code == 200:
                    return response.json()
                else:
                    status = response.status_code if response else "No response"
                    print(f"Failed to upload file. Status: {status}")
                    return None
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
    
    def check_job_status(self, job_id):
        """Check the status of a processing job"""
        response = self.make_authenticated_request(f"job-status/{job_id}")
        
        if response and response.status_code == 200:
            return response.json()
        else:
            status = response.status_code if response else "No response"
            print(f"Failed to check job status. Status: {status}")
            return None

def demo():
    client = TokenRefreshClient()
    
    # Step 1: Login
    if not client.login("testuser", "password123"):
        return
    
    # Step 2: Get protected data
    print("\nFetching protected data...")
    data = client.get_protected_data()
    if data:
        print(f"Protected data: {json.dumps(data, indent=2)}")
    
    # Step 3: Upload a file (if specified)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nUploading file: {file_path}")
        upload_result = client.upload_file(file_path)
        
        if upload_result:
            job_id = upload_result['job_id']
            print(f"File uploaded successfully. Job ID: {job_id}")
            
            # Step 4: Monitor job status with token refresh
            print("\nMonitoring job status...")
            completed = False
            
            while not completed:
                time.sleep(10)  # Check every 10 seconds
                
                status = client.check_job_status(job_id)
                if not status:
                    print("Failed to get job status.")
                    break
                    
                print(f"Job status: {status['status']}")
                
                if status['status'] in ['completed', 'failed']:
                    completed = True
                    print(f"Final job result: {json.dumps(status, indent=2)}")
    else:
        print("\nNo file specified for upload. Pass a file path as an argument to test file upload.")
    
    # Demonstrate token refresh
    print("\nSimulating passage of time to trigger token refresh...")
    # Force token to expire for demonstration
    client.access_token_expiry = time.time() - 10
    
    print("Fetching protected data again (should trigger token refresh)...")
    data = client.get_protected_data()
    if data:
        print(f"Protected data after token refresh: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    demo()