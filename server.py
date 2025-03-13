# server.py - Authentication and File Processing API
from flask import Flask, request, jsonify
import jwt
import datetime
import uuid
import os
import time
import threading
import requests
from functools import wraps

app = Flask(__name__)

# In a real application, use a secure secret key stored in environment variables
SECRET_KEY = "this-is-my-super-secret-key"

# Mock database for demonstration
tokens_db = {}
file_processing_jobs = {}

# Token expiration settings (in seconds)
ACCESS_TOKEN_EXPIRY = 600  # 10 minutes
REFRESH_TOKEN_EXPIRY = 3600  # 1 hour

# Decorator to verify the access token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            
            # Check if token is expired
            if datetime.datetime.utcnow().timestamp() > payload['exp']:
                return jsonify({'message': 'Token has expired!'}), 401
                
            # Check if token is valid in our database
            if payload['token_id'] not in tokens_db:
                return jsonify({'message': 'Invalid token!'}), 401
                
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
            
        return f(payload, *args, **kwargs)
    
    return decorated

# Login endpoint to generate initial tokens
@app.route('/login', methods=['POST'])
def login():
    auth = request.json
    
    # In a real app, validate username/password against a database
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify!'}), 401
        
    # For demonstration, accepting any username/password
    # In a real app, validate credentials from a database
    
    # Generate tokens
    token_id = str(uuid.uuid4())
    refresh_token_id = str(uuid.uuid4())
    
    # Create access token
    access_token_payload = {
        'token_id': token_id,
        'username': auth.get('username'),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=ACCESS_TOKEN_EXPIRY),
        'iat': datetime.datetime.utcnow()
    }
    
    # Create refresh token
    refresh_token_payload = {
        'token_id': refresh_token_id,
        'username': auth.get('username'),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=REFRESH_TOKEN_EXPIRY),
        'iat': datetime.datetime.utcnow()
    }
    
    access_token = jwt.encode(access_token_payload, SECRET_KEY, algorithm="HS256")
    refresh_token = jwt.encode(refresh_token_payload, SECRET_KEY, algorithm="HS256")
    
    # Store tokens in our database
    tokens_db[token_id] = {
        'username': auth.get('username'),
        'refresh_token_id': refresh_token_id,
        'expires_at': access_token_payload['exp']
    }
    
    tokens_db[refresh_token_id] = {
        'username': auth.get('username'),
        'is_refresh_token': True,
        'expires_at': refresh_token_payload['exp']
    }
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'access_token_expires_in': ACCESS_TOKEN_EXPIRY,
        'refresh_token_expires_in': REFRESH_TOKEN_EXPIRY
    })

# Refresh token endpoint
@app.route('/refresh-token', methods=['POST'])
def refresh_token():
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return jsonify({'message': 'Refresh token is missing!'}), 401
        
    refresh_token = auth_header.split(' ')[1]
    
    try:
        # Decode the refresh token
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=["HS256"])
        refresh_token_id = payload['token_id']
        
        # Check if refresh token exists and is valid
        if refresh_token_id not in tokens_db or 'is_refresh_token' not in tokens_db[refresh_token_id]:
            return jsonify({'message': 'Invalid refresh token!'}), 401
            
        # Check if token is expired
        if datetime.datetime.utcnow().timestamp() > tokens_db[refresh_token_id]['expires_at']:
            # Remove expired token
            tokens_db.pop(refresh_token_id, None)
            return jsonify({'message': 'Refresh token has expired!'}), 401
            
        username = tokens_db[refresh_token_id]['username']
        
        # Generate new access token
        new_token_id = str(uuid.uuid4())
        
        # Create new access token
        new_access_token_payload = {
            'token_id': new_token_id,
            'username': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=ACCESS_TOKEN_EXPIRY),
            'iat': datetime.datetime.utcnow()
        }
        
        new_access_token = jwt.encode(new_access_token_payload, SECRET_KEY, algorithm="HS256")
        
        # Store new token in our database
        tokens_db[new_token_id] = {
            'username': username,
            'refresh_token_id': refresh_token_id,
            'expires_at': new_access_token_payload['exp']
        }
        
        return jsonify({
            'access_token': new_access_token,
            'expires_in': ACCESS_TOKEN_EXPIRY
        })
        
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid refresh token!'}), 401

# Protected route example
@app.route('/protected-data', methods=['GET'])
@token_required
def protected_data(payload):
    return jsonify({
        'message': f"Hello, {payload['username']}! This is protected data.",
        'data': {
            'secret_info': 'This is some confidential information',
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
    })

# File upload endpoint
@app.route('/upload-file', methods=['POST'])
@token_required
def upload_file(payload):
    # Check if file was provided
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request!'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No file selected!'}), 400
        
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save file to a temporary location
    filename = f"temp_{job_id}_{file.filename}"
    file_path = os.path.join('/tmp', filename)
    file.save(file_path)
    
    # Get the refresh token ID associated with this user's access token
    refresh_token_id = tokens_db[payload['token_id']]['refresh_token_id']
    
    # Store job information
    file_processing_jobs[job_id] = {
        'username': payload['username'],
        'file_path': file_path,
        'status': 'queued',
        'refresh_token_id': refresh_token_id,
        'submitted_at': datetime.datetime.utcnow().isoformat(),
        'result': None
    }
    
    # Start the file processing in a background thread
    processing_thread = threading.Thread(
        target=process_file_with_token_refresh,
        args=(job_id, refresh_token_id)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({
        'message': 'File upload successful. Processing started.',
        'job_id': job_id
    })

# Job status endpoint
@app.route('/job-status/<job_id>', methods=['GET'])
@token_required
def job_status(payload, job_id):
    if job_id not in file_processing_jobs:
        return jsonify({'message': 'Job not found!'}), 404
        
    job = file_processing_jobs[job_id]
    
    # Ensure user can only access their own jobs
    if job['username'] != payload['username']:
        return jsonify({'message': 'Unauthorized access to job!'}), 403
        
    response = {
        'job_id': job_id,
        'status': job['status'],
        'submitted_at': job['submitted_at']
    }
    
    if job['status'] == 'completed':
        response['result'] = job['result']
        
    return jsonify(response)

# Function to handle file processing with token refresh
def process_file_with_token_refresh(job_id, refresh_token_id):
    job = file_processing_jobs[job_id]
    job['status'] = 'processing'
    
    # In a real application, this might involve more complex processing
    # that requires multiple API calls with valid tokens
    
    try:
        # Simulate a long-running process
        total_processing_time = 180  # 3 minutes for demo purposes
        elapsed_time = 0
        
        while elapsed_time < total_processing_time:
            # Every 8 minutes (just before access token expires), get a new access token
            # For demo purposes, we'll do this every 50 seconds
            if elapsed_time % 50 == 0 and elapsed_time > 0:
                # Get a new access token using the refresh token
                new_access_token = refresh_access_token(refresh_token_id)
                
                if not new_access_token:
                    # If refresh failed, mark job as failed
                    job['status'] = 'failed'
                    job['result'] = 'Authentication failure during processing'
                    return
                
                # Use the new access token to make authenticated requests
                # Here we'll just simulate this
                print(f"Using new access token for job {job_id}")
            
            # Simulate work
            time.sleep(10)
            elapsed_time += 10
            
            # Simulate progress updates
            progress = min(100, int((elapsed_time / total_processing_time) * 100))
            print(f"Job {job_id} progress: {progress}%")
        
        # Simulate saving results to database using the current valid token
        # In a real scenario, you'd make an authenticated API call
        job['status'] = 'completed'
        job['result'] = {
            'processed_data': f"Processed content of file {os.path.basename(job['file_path'])}",
            'processing_time': f"{total_processing_time} seconds",
            'completed_at': datetime.datetime.utcnow().isoformat()
        }
        
        # Clean up
        try:
            os.remove(job['file_path'])
        except:
            pass
            
    except Exception as e:
        job['status'] = 'failed'
        job['result'] = f"Error during processing: {str(e)}"

# Function to refresh the access token
def refresh_access_token(refresh_token_id):
    try:
        # Check if refresh token is still valid
        if refresh_token_id not in tokens_db:
            print(f"Refresh token {refresh_token_id} not found")
            return None
            
        refresh_token_data = tokens_db[refresh_token_id]
        
        # Check if token is expired
        if datetime.datetime.utcnow().timestamp() > refresh_token_data['expires_at']:
            print(f"Refresh token {refresh_token_id} has expired")
            tokens_db.pop(refresh_token_id, None)
            return None
            
        # Generate new access token
        new_token_id = str(uuid.uuid4())
        
        # Create access token
        access_token_payload = {
            'token_id': new_token_id,
            'username': refresh_token_data['username'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=ACCESS_TOKEN_EXPIRY),
            'iat': datetime.datetime.utcnow()
        }
        
        access_token = jwt.encode(access_token_payload, SECRET_KEY, algorithm="HS256")
        
        # Store new token in our database
        tokens_db[new_token_id] = {
            'username': refresh_token_data['username'],
            'refresh_token_id': refresh_token_id,
            'expires_at': access_token_payload['exp']
        }
        
        return access_token
        
    except Exception as e:
        print(f"Error refreshing token: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, port=5000)