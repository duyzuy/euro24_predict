#!/bin/bash

# Euro 2024 Prediction API Deployment Script
# Usage: ./deploy.sh [development|production]

set -e

ENVIRONMENT=${1:-development}
PROJECT_NAME="euro24-prediction-api"

echo "🚀 Deploying Euro 2024 Prediction API in $ENVIRONMENT mode..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build the Docker image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t $PROJECT_NAME:latest .
    echo "✅ Docker image built successfully!"
}

# Function to run in development mode
run_development() {
    echo "🏃 Starting in development mode..."
    docker-compose up --build -d euro24-api
    
    echo "⏳ Waiting for API to be ready..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is running and healthy!"
        echo "📖 API Documentation: http://localhost:8000/docs"
        echo "🔍 Health Check: http://localhost:8000/health"
    else
        echo "❌ API health check failed!"
        docker-compose logs euro24-api
        exit 1
    fi
}

# Function to run in production mode
run_production() {
    echo "🏭 Starting in production mode with nginx..."
    docker-compose --profile production up --build -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 15
    
    # Health check through nginx
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ Production deployment successful!"
        echo "🌐 API (via nginx): http://localhost"
        echo "📖 API Documentation: http://localhost/docs"
        echo "🔍 Health Check: http://localhost/health"
    else
        echo "❌ Production deployment health check failed!"
        docker-compose logs
        exit 1
    fi
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping services..."
    docker-compose --profile production down
    echo "✅ All services stopped!"
}

# Function to show logs
show_logs() {
    echo "📋 Showing logs..."
    docker-compose logs -f
}

# Main deployment logic
check_docker

case $ENVIRONMENT in
    "development" | "dev")
        run_development
        ;;
    "production" | "prod")
        run_production
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        show_logs
        ;;
    "build")
        build_image
        ;;
    *)
        echo "Usage: $0 [development|production|stop|logs|build]"
        echo ""
        echo "Commands:"
        echo "  development  - Run API in development mode (port 8000)"
        echo "  production   - Run API with nginx in production mode (port 80)"
        echo "  stop         - Stop all running services"
        echo "  logs         - Show service logs"
        echo "  build        - Build Docker image only"
        exit 1
        ;;
esac

echo "🎉 Deployment completed!"