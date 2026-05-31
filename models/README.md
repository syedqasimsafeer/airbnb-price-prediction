# Models Folder

No large model file is required for deployment.

The Streamlit app trains the model automatically on first run and caches it during the session.

This avoids version mismatch errors that sometimes happen when uploading `.joblib` files trained in a different environment.
