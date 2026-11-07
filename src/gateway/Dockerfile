FROM python:3.9-slim-bookworm
WORKDIR /app
COPY src/gateway/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY src/gateway/app.py ./
USER 65534:65534
CMD ["python", "app.py"]
