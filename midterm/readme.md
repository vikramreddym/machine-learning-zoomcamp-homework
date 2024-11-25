Prerequisites:
Docker Desktop

Steps to run the model:
docker build -t adpred .
docker run -d -p 9696:9696 adpred
python predict_client.py