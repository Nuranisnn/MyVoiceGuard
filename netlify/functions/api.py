import awsgi

from app import app


def handler(event, context):
    """
    Netlify Functions entrypoint.
    Reuses the Flask app and all existing API routes:
    - /health
    - /predict-file
    - /predict-url
    - /predict-live
    """
    return awsgi.response(app, event, context)
