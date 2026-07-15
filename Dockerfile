FROM python:3.11-slim
# set working directory
WORKDIR /app 

# copy all project files
COPY . /app

# install dependencies for inference
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 \
    flask \
    pillow

# expose port for flask   
EXPOSE 8000

# run flask app
CMD ["python", "app.py"]
