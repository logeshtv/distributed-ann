FROM node:18-alpine

WORKDIR /app

# Install Python, pip, and timezone data
RUN apk add --no-cache python3 py3-pip tzdata

# Set timezone
ENV TZ=America/New_York

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Install Python packages needed for data download
# Install in stages to avoid memory issues
RUN pip3 install --break-system-packages --no-cache-dir numpy pandas && \
    pip3 install --break-system-packages --no-cache-dir yfinance pyarrow && \
    pip3 install --break-system-packages --no-cache-dir requests loguru python-dotenv tqdm

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001
USER nodejs

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3001/health || exit 1

# Start server
CMD ["node", "src/server.js"]
