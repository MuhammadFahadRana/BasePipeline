# Command

docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main

# Breakdown of this command:

-p 3000:8080: Makes the interface accessible at http://localhost:3000.
--add-host=host.docker.internal:host-gateway: This is the "host" part you remembered. It allows the Docker container to "see" your Windows host, which is necessary if Ollama is running locally on your machine rather than inside Docker.
-v open-webui:/app/backend/data: Ensures your chats and settings are saved even if the container is stopped.
ghcr.io/open-webui/open-webui:main: The official image.

# Model Details

The model name is set to **ATLAS**. You can select it in the Open WebUI interface after connecting it to the API at `http://host.docker.internal:8000/v1`.
