# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Accept build arguments
ARG VITE_SOCKET_URL=http://localhost:3001
ARG VITE_API_URL=http://localhost:3001/api

# Set environment variables for Vite
ENV VITE_SOCKET_URL=$VITE_SOCKET_URL
ENV VITE_API_URL=$VITE_API_URL

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
