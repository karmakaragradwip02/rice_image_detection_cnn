# Use the official FastAPI image from Tiangolo
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory inside the container
WORKDIR /app

# Copy only the essential files
COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
COPY ./src /app/src
COPY ./README.md /app/README.md
COPY ./fastapi.py /app/
COPY ./templates /app/templates
COPY ./dvc.yaml /app/dvc.yaml
COPY ./params.yaml /app/params.yaml
COPY ./config /app/config
COPY ./main.py /app/main.py


# Install pip and project dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Install the package
RUN pip install --no-cache-dir -e .

# Command to run the FastAPI application
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "80"]