FROM python:3.10-bullseye

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

RUN mkdir /app
RUN mkdir /app/assets
RUN mkdir /app/utils
COPY *.py /app/
COPY requirements.txt /app/
Copy assets/* /app/assets/


WORKDIR /app
RUN pip3 install -r requirements.txt

COPY javascript /app/javascript
WORKDIR /app/javascript

RUN npm install
RUN npm run build

EXPOSE 7860

CMD ["python3", "server.py"]