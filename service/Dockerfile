FROM python:3.6.5-slim

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev python3 build-essential

RUN mkdir -p /usr/src/app
COPY ./ /usr/src/app

WORKDIR /usr/src/app

RUN pip3 install -r requirements.txt


EXPOSE 5000

ENTRYPOINT ["flask"]

CMD ["run", "--host=0.0.0.0"]

