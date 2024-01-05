from __future__ import annotations

from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request

from langsuite.utils.logging import logger


def serve(env, *args):
    app = FastAPI()

    @app.get("/")
    async def root(req: Request):
        req_info = await req.body()

        if state := env.get_task_info():
            logger.info(state)
            return {
                "status_code": 200,
                "timestamp": datetime.timestamp(datetime.now()),
                "request": req_info,
                "data": [],
                "feedback": state,
            }

    @app.get("/fetch/scene")
    async def handle_request_fetch_scene(req: Request):
        req_info = await req.body()

        # action = req_info.get("action")
        # if action == "fetch_scene":
        figure = env.render(mode="webui")

        return {
            "status_code": 200,
            "timestamp": datetime.timestamp(datetime.now()),
            "request": req_info,
            "data": {"scene": figure.to_json(pretty=True, remove_uids=False)},
        }
        # raise HTTPException(status_code=500, detail=f"action {req_info.get('action')} is not defined.")

    @app.get("/fetch/config")
    async def handle_request_fetch_config(req: Request):
        req_info = await req.body()

        return {
            "status_code": 200,
            "timestamp": datetime.timestamp(datetime.now()),
            "request": req_info,
            "data": {
                "config": {
                    "env": env.cfg,
                    "agents": [agent.get_config() for (_, agent) in env.agents.items()],
                }
            },
        }

    @app.get("/update")
    async def handle_request_update(req: Request):
        req_info = await req.json()

        config = req_info.get("config")
        # if action == "fetch_scene":
        env.update_config(config)
        figure = env.render(mode="webui")

        return {
            "status_code": 200,
            "timestamp": datetime.timestamp(datetime.now()),
            "request": req_info,
            "data": {"scene": figure.to_json(pretty=True, remove_uids=False)},
        }

    @app.get("/action")
    async def handle_request_action(req: Request):
        req_info = await req.json()

        action = req_info.get("action")
        # if action == "fetch_scene":
        if state := env.step({"action": action}):
            figure = env.render(mode="webui")
            logger.info(state)

            return {
                "status_code": 200,
                "timestamp": datetime.timestamp(datetime.now()),
                "request": req_info,
                "feedback": state,
                "data": {
                    "scene": figure.to_json(pretty=True, remove_uids=False),
                },
            }
        else:
            logger.info(state)
            return {
                "status_code": 500,
                "timestamp": datetime.timestamp(datetime.now()),
                "request": req_info,
                "data": {},
            }

    @app.get("/message")
    async def handle_request_message(req: Request):
        req_info = await req.json()

        message = req_info.get("message")
        logger.info(message)
        # if action == "fetch_scene":
        
        if state := env.step({"action": message["content"]}):
            figure = env.render(mode="webui")
            logger.info(state)

            return {
                "status_code": 200,
                "timestamp": datetime.timestamp(datetime.now()),
                "request": req_info,
                "feedback": state,
                "data": {
                    "scene": figure.to_json(pretty=True, remove_uids=False),
                },
            }
        else:
            logger.info(state)
            return {
                "status_code": 500,
                "timestamp": datetime.timestamp(datetime.now()),
                "request": req_info,
                "data": {},
            }

    uvicorn.run(app, host="0.0.0.0", port=8022, log_level="error")
