# Deploying MAORS on Streamlit

Local:   pip install -r requirements.txt && streamlit run streamlit_app.py

Streamlit Community Cloud:
1. Push this folder to a GitHub repo, with streamlit_app.py, maors/, models/,
   maternal_ai_dataset.xlsx and requirements.txt at the repo root.
2. share.streamlit.io -> Create app -> pick the repo, branch main, file streamlit_app.py.
3. Advanced settings -> Python 3.12. No secrets are needed.

scikit-learn and imbalanced-learn are pinned to the versions the saved models were tested
with (1.8.0 / 0.14.2); keep them pinned or retrain with build_models.py after upgrading.
The FastAPI app (app.py, index.html, Docker files) is unchanged and still works on its own.
