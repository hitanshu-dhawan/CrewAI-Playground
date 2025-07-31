# Use Python 3 as the base image
FROM python:3

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase (dockerignore will exclude .md files)
COPY . .

# Set Python to run in unbuffered mode (recommended for Docker)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]