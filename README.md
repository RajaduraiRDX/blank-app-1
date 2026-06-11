# AIMS Research Console

A Flask-based Aggregate Image Measurement System for civil engineering research.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Deploy as a Web Link

The easiest handoff is to deploy it as a Flask web service so collaborators can open a normal URL.

### Render

1. Put this project in a GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Use these settings:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Deploy. Render will give you an `onrender.com` URL.

The included `Procfile` also contains the same production start command.

## Send as a ZIP

If the other person will run it on their own computer, zip the project folder and ask them to run:

```bash
pip install -r requirements.txt
python app.py
```

Then they can open `http://127.0.0.1:5000`.

## Notes for Research Use

- Uploaded files are stored in `work/uploads`.
- Generated analysis results are stored in `outputs/results`.
- For public deployments, add authentication before sharing sensitive lab images.
