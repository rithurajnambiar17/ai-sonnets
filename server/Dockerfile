# 1 
FROM python:3.8.10

# 2
RUN pip install Flask gunicorn pickle numpy tensorflow

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app