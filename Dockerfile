FROM python:3.7

# Install utils
RUN apt-get update

# Check our python environment
RUN python3 --version
RUN pip3 --version

# Set the working directory for containers
WORKDIR  /usr/src/<app-name>

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from project root to the working dir
COPY src/ /src/
RUN ls -la /src/*

# Run python application
CMD ["python3", "/src/main.py"]
