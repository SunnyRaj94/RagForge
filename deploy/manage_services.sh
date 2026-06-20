#!/usr/bin/env bash

# deploy/manage_services.sh
# Script to orchestrate starting, stopping, and checking the status of RagForge services.

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;37m' # No Color

# Determine directories relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# Create logs directory
mkdir -p "$PROJECT_ROOT/data/logs"

show_status() {
    echo -e "${BLUE}=== Service Status ===${NC}"
    
    # 1. Check Docker Containers
    echo -e "\n${YELLOW}[Docker Containers]${NC}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Docker daemon is not running.${NC}"
    else
        docker compose -p ragforge -f "$COMPOSE_FILE" ps
    fi

    # 2. Check Ollama
    echo -e "\n${YELLOW}[Ollama Server]${NC}"
    if lsof -i :11434 >/dev/null 2>&1; then
        echo -e "${GREEN}Ollama is running on port 11434.${NC}"
    else
        echo -e "${RED}Ollama is not running.${NC}"
    fi

    # 3. Check Temporal Worker
    echo -e "\n${YELLOW}[Temporal Worker]${NC}"
    WORKER_PID=$(pgrep -f "src/ragforge/worker.py")
    if [ -n "$WORKER_PID" ]; then
        echo -e "${GREEN}Temporal Worker is running (PID: $WORKER_PID).${NC}"
    else
        echo -e "${RED}Temporal Worker is not running.${NC}"
    fi

    # 4. Check Streamlit App
    echo -e "\n${YELLOW}[Streamlit UI]${NC}"
    STREAMLIT_PID=$(pgrep -f "streamlit run app.py")
    if [ -n "$STREAMLIT_PID" ]; then
        echo -e "${GREEN}Streamlit UI is running (PID: $STREAMLIT_PID).${NC}"
    else
        echo -e "${RED}Streamlit UI is not running.${NC}"
    fi
}

start_services() {
    echo -e "${BLUE}=== Starting RagForge Services ===${NC}"

    # 1. Docker Compose
    echo -e "\n${YELLOW}Starting Docker containers...${NC}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker daemon is not running. Please start Docker first.${NC}"
        exit 1
    fi
    docker compose -p ragforge -f "$COMPOSE_FILE" up -d
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Docker containers started successfully.${NC}"
    else
        echo -e "${RED}Failed to start Docker containers.${NC}"
        exit 1
    fi

    # 2. Ollama
    echo -e "\n${YELLOW}Checking Ollama server...${NC}"
    if lsof -i :11434 >/dev/null 2>&1; then
        echo -e "${GREEN}Ollama is already running.${NC}"
    else
        echo -e "${YELLOW}Ollama is not running. Starting Ollama in background...${NC}"
        if command -v ollama >/dev/null 2>&1; then
            OLLAMA_CONTEXT_LENGTH=65536 OLLAMA_FLASH_ATTENTION=true OLLAMA_KEEP_ALIVE="-1" OLLAMA_MAX_LOADED_MODELS=2 nohup ollama serve > "$PROJECT_ROOT/data/logs/ollama.log" 2>&1 &
            sleep 3
            if lsof -i :11434 >/dev/null 2>&1; then
                echo -e "${GREEN}Ollama started successfully.${NC}"
            else
                echo -e "${RED}Failed to start Ollama. Check logs in data/logs/ollama.log${NC}"
            fi
        else
            echo -e "${RED}Error: 'ollama' command not found. Please install Ollama or start it manually.${NC}"
        fi
    fi

    # 3. Temporal Worker
    echo -e "\n${YELLOW}Checking Temporal Worker...${NC}"
    WORKER_PID=$(pgrep -f "src/ragforge/worker.py")
    if [ -n "$WORKER_PID" ]; then
        echo -e "${GREEN}Temporal Worker is already running.${NC}"
    else
        echo -e "${YELLOW}Starting Temporal Worker in background...${NC}"
        # Shift directory to project root to execute worker cleanly
        cd "$PROJECT_ROOT"
        PYTHONPATH=. nohup uv run python src/ragforge/worker.py > data/logs/temporal_worker.log 2>&1 &
        sleep 2
        WORKER_PID=$(pgrep -f "src/ragforge/worker.py")
        if [ -n "$WORKER_PID" ]; then
            echo -e "${GREEN}Temporal Worker started successfully (PID: $WORKER_PID).${NC}"
        else
            echo -e "${RED}Failed to start Temporal Worker. Check logs in data/logs/temporal_worker.log${NC}"
        fi
    fi

    echo -e "\n${GREEN}All background services running successfully!${NC}"
    echo -e "To launch the Streamlit chat UI, run:"
    echo -e "${BLUE}uv run streamlit run app.py${NC}\n"
}

stop_services() {
    echo -e "${BLUE}=== Stopping RagForge Services ===${NC}"

    # 1. Temporal Worker
    echo -e "\n${YELLOW}Stopping Temporal Worker...${NC}"
    WORKER_PID=$(pgrep -f "src/ragforge/worker.py")
    if [ -n "$WORKER_PID" ]; then
        kill $WORKER_PID
        echo -e "${GREEN}Temporal Worker stopped.${NC}"
    else
        echo -e "${GREEN}Temporal Worker is not running.${NC}"
    fi

    # 2. Ollama Server
    echo -e "\n${YELLOW}Stopping Ollama...${NC}"
    if lsof -i :11434 >/dev/null 2>&1; then
        OLLAMA_PID=$(lsof -t -i :11434)
        if [ -n "$OLLAMA_PID" ]; then
            kill $OLLAMA_PID
            echo -e "${GREEN}Ollama server stopped.${NC}"
        fi
    else
        echo -e "${GREEN}Ollama is not running.${NC}"
    fi

    # 3. Streamlit UI (if running in background)
    echo -e "\n${YELLOW}Stopping Streamlit UI...${NC}"
    STREAMLIT_PID=$(pgrep -f "streamlit run app.py")
    if [ -n "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID
        echo -e "${GREEN}Streamlit UI stopped.${NC}"
    else
        echo -e "${GREEN}Streamlit UI is not running.${NC}"
    fi

    # 4. Docker Containers
    echo -e "\n${YELLOW}Stopping Docker containers...${NC}"
    if docker info >/dev/null 2>&1; then
        docker compose -p ragforge -f "$COMPOSE_FILE" down
        echo -e "${GREEN}Docker containers stopped.${NC}"
    else
        echo -e "${RED}Docker daemon not running; skipping containers shutdown.${NC}"
    fi
}

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
