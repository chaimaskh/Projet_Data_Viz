# Use an official Python runtime as a parent image
FROM python:3.10.7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["panel", "serve", "dashboardEDA.py", "--address", "0.0.0.0", "--port", "80", "--allow-websocket-origin", "*"]
