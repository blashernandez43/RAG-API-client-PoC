# Use the official Selenium Standalone Chrome image as the base image
FROM registry.access.redhat.com/ubi8/python-311:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container and install dependencies
COPY requirements.txt /app/requirements.txt
USER 0
RUN pip3 install -r requirements.txt

# Copy your streamlit Python script to the container
COPY . .

EXPOSE 4050

# Set the command to run app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=4050", "--server.address=0.0.0.0"]
