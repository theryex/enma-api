import os
import uvicorn
import yaml
import aiohttp

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import Completion

app = FastAPI(
    title="Enma API Gateway"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

args = {
    "config": os.getenv("GATEWAY_CONF", "gateway-conf.yaml"),
    "port": int(os.getenv("GATEWAY_PORT", 9009)),
    "inference_url": os.getenv("INFERENCE_URL"),
    "inference_model_name": os.getenv("INFERENCE_MODEL_NAME"),
    "inference_author": os.getenv("INFERENCE_AUTHOR", "Unknown"),
    "inference_description": os.getenv("INFERENCE_DESCRIPTION", "Hosted Model")
}

# Configuration Logic
# If INFERENCE_URL and INFERENCE_MODEL_NAME are set (Docker mode), use them.
# Otherwise, fall back to loading the config file.

config_mode = "env" if args["inference_url"] and args["inference_model_name"] else "file"
config = {}

if config_mode == "env":
    print("Running in Environment Variable Mode (Single Model)")
    # Construct a config-like structure for compatibility
    # Force lowercase for the key to ensure case-insensitive matching
    model_key = args["inference_model_name"].lower()
    config = {
        "models": {
            model_key: {
                "author": args["inference_author"],
                "description": args["inference_description"],
                "url": args["inference_url"]
            }
        }
    }
else:
    print(f"Running in Config File Mode: Loading {args['config']}")
    if os.path.exists(args["config"]):
        with open(args["config"], "r") as f:
            config = yaml.safe_load(f)
            # Ensure keys from yaml are treated consistently (optional, but good practice if we enforce lowercase)
            # For backward compatibility, we'll assume the yaml is correct as-is, or we could lower() keys here too.
            # Let's keep file mode behavior close to original unless needed.
    else:
        print(f"Warning: Config file {args['config']} not found and env vars not set. Gateway may malfunction.")
        config = {"models": {}}

@app.get("/engines")
async def engines():
    all_engines = []
    for engine in config["models"].keys():
        all_engines.append({
            "name": engine,
            "author": config["models"][engine]["author"],
            "description": config["models"][engine]["description"]
        })
    return {"engines": all_engines}

@app.post("/completion")
async def completion(completion: Completion):
    if completion.engine is None:
        # If running in single-model env mode, we can default to the only available model
        if config_mode == "env" and len(config["models"]) == 1:
            completion.engine = list(config["models"].keys())[0]
        else:
            raise Exception("Engine not specified")

    # Normalize requested engine name to lowercase for lookup
    requested_engine = completion.engine.lower()

    if requested_engine not in config["models"]:
         raise Exception(f"Engine '{completion.engine}' not found")

    engine_endpoint = config["models"][requested_engine]["url"]
    async with aiohttp.ClientSession() as session:
        # Pass the original completion object (which might preserve the original case if that matters downstream,
        # though usually it doesn't).
        async with session.post(engine_endpoint, json=completion.dict()) as resp:
            return await resp.json()

@app.get("/")
async def root():
    return "Sometimes I dream about cheese."

if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=args["port"]
    )
