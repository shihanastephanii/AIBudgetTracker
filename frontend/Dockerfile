FROM python:3.11-slim

WORKDIR /app

COPY . .

EXPOSE 8080

CMD sh -c "python3 -m http.server ${PORT:-8080}"
