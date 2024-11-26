FROM --platform=linux/amd64 python:3.12-slim

RUN pip install pipenv
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock",  "./"]
RUN pipenv install --system --deploy

COPY ["service.py", "./"]
COPY ["dict_vectorizer.pkl", "./"]
COPY ["standard_scaler.pkl", "./"]
COPY ["xgb_model.pkl", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "service:app"]