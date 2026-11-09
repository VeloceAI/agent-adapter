FROM node:24-slim
WORKDIR /app
COPY package.json .
RUN npm install --no-audit --no-fund --loglevel=error
COPY client.ts .
CMD ["node", "client.ts"]
