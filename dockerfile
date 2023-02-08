FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Install dependencies from requirements.txt
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . /home

# Set the working directory to /app
WORKDIR /home

# install requirements
RUN apt-get update && apt-get install -y git && pip install -r requirements.txt
