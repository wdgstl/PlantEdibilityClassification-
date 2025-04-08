FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./scripts ./scripts

COPY ./data ./data

EXPOSE 8000

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]