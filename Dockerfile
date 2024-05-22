FROM python:3.9

   WORKDIR /src 
   
   # Copy the local code to the container
   COPY requirements.txt requirements.txt
   COPY datasets datasets
   COPY models models

   RUN pip install --upgrade pip
   RUN pip install fastapi uvicorn
   RUN pip install -r requirements.txt

   COPY ["app.py", "./"]

   EXPOSE 80

   CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "80"]