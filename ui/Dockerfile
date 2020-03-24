FROM node:latest

#create app dir

RUN mkdir -p usr/src/app
WORKDIR /usr/src/app

COPY package.json /usr/src/app
COPY . /usr/src/app

EXPOSE 5000

RUN yarn install

RUN npm install -g serve

RUN npm run build

CMD ["npm", "start"]