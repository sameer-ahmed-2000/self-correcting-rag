# syntax=docker/dockerfile:1
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage — serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config to support React Router (SPA fallback)
RUN echo 'server { \
  listen 80; \
  root /usr/share/nginx/html; \
  index index.html; \
  location / { try_files $uri $uri/ /index.html; } \
  location /api/ { proxy_pass http://backend:8000; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection keep-alive; } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
