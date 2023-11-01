# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations

import asyncio
import os
from asyncio import Semaphore

import aiohttp
import plotly.io
import streamlit as st
from streamlit import config as st_config, logger

limit = 10
st_logger = logger.get_logger(__name__)
st_config.set_option("server.headless", True)
app_path = os.path.abspath(__file__)
app_args = dict()
flag_options = {}


async def fetch_scene(_session):
    if not st.session_state.server_started:
        return

    async with _session.get("/fetch/scene") as response:
        # response = await requests.get(url_backend,

        if response.status == 200:
            response = await response.json()
            return response


async def fetch_config(_session):
    async with _session.get("/fetch/config") as response:
        # response = await requests.get(url_backend,

        if response.status == 200:
            response = await response.json()
            return response.get("data").get("config")


async def check_if_server_started(client):
    async with Semaphore(limit):
        async with client.get("/", json={"render": True}) as response:
            if response.status == 200:
                st_logger.info("Server started")
                # response = await response.json()
                # if response.get("status_code") == 200:
                st.session_state.server_started = True
                response = await response.json()
                st_logger.info(response)
                return response


async def action_callback(_session, action_name):
    async with _session.get(
        "/action", json={"action": action_name, "render": True}
    ) as response:
        # response = await requests.get(url_backend,

        if response.status == 200:
            response = await response.json()
            return response


async def update_config_callback(_session, config):
    async with _session.get(
        "/update", json={"config": config, "render": True}
    ) as response:
        # response = await requests.get(url_backend,

        if response.status == 200:
            response = await response.json()
            return response


async def send_chat_message(_session, message):
    async with _session.get("/message", json={"message": message}) as response:
        # response = await requests.get(url_backend,

        if response.status == 200:
            response = await response.json()
            return response


def render_scene(scene_data):
    figure = plotly.io.from_json(scene_data)
    figure.update_layout(height=800)
    st.plotly_chart(figure, theme="streamlit", use_container_width=True)


def submit_callback():
    if not st.session_state.submit_clicked:
        st.session_state.submit_clicked = True
        st_logger.info("Button clicked")


class StreamlitApp:
    def __init__(self, title="LangSuitE") -> None:
        self.title = title
        self.header = ""
        st.set_page_config(
            page_title=self.title,
            page_icon="🎈",
        )
        self._plot_figure = None

        if "scene" not in st.session_state:
            st.session_state.scene = dict(timestamp=None)

        if "submit_clicked" not in st.session_state:
            st.session_state.submit_clicked = False

        if "server_started" not in st.session_state:
            st.session_state.server_started = False

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if "chat_input" not in st.session_state:
            st.session_state.chat_input = False

    def _css_config(self):
        st.markdown(
            """
        <style>
        .appview-container .main .block-container {
            max-width: 1400px;
        }

        .stButton {
            margin-left: auto;
            margin-right: auto;
        }

        </style>
        """,
            unsafe_allow_html=True,
        )

    def restart(self):
        st.experimental_rerun()

    async def start(self):
        self._css_config()

        c30, c31, c32 = st.columns([2.5, 1, 3])

        with c30:
            st.title(self.title)
            st.header(self.header)

        with st.expander("Introduction", expanded=True):
            st.markdown(
                """
                Introductions here
                """
            )
        async with aiohttp.ClientSession(base_url="http://0.0.0.0:8022") as session:
            if not st.session_state.server_started:
                response = await check_if_server_started(session)
                if response and "feedback" in response:
                    st_logger.info(response["feedback"])
                    st.session_state.chat_messages.append(
                        {
                            "role": "user",
                            "content": response["feedback"]["state"]["feedback"],
                        }
                    )

            env_cfg = await fetch_config(session)
            st_logger.info(env_cfg)

            caption = "Map Viewer"
            st.markdown("")
            st.markdown(f"## **{caption}**")

            # with st.form(key="st_form"):
            _, c1, _, c2, _ = st.columns([0.07, 1, 0.07, 5, 0.07])
            with c1:
                game_mode = st.selectbox(
                    "Game Mode",
                    options=[
                        "Chat",
                        "Manual",
                    ],
                )
                with st.form(key="config"):
                    max_view_distance = st.slider(
                        "Max View Distance",
                        min_value=1.0,
                        max_value=5.0,
                        value=float(env_cfg["agents"][0]["max_view_distance"]),
                        help="max view distance of agent",
                    )

                    focal_length = st.slider(
                        "Focal Length",
                        min_value=1,
                        max_value=30,
                        value=env_cfg["agents"][0]["focal_length"],
                        help="max view distance of agent",
                    )

                    step_size = st.slider(
                        label="Step Size",
                        min_value=0.01,
                        max_value=2.0,
                        value=env_cfg["agents"][0]["step_size"],
                        help="agent step size",
                    )
                    submit_form = st.form_submit_button(
                        label="Update Config", use_container_width=True
                    )

                if submit_form:
                    agent_config = env_cfg["agents"][0]
                    agent_config.update(
                        {
                            "focal_length": focal_length,
                            "max_view_distance": max_view_distance,
                            "step_size": step_size,
                        }
                    )
                    env_cfg.update({"agents": [agent_config]})
                    scene_response = await update_config_callback(session, env_cfg)

            # Prompting Box
            scene_response = None
            if game_mode == "Chat":
                if prompt := st.chat_input("Prompt"):
                    # with st.chat_message("user"):
                    #     st.markdown(prompt)
                    new_message = {"role": "system", "content": prompt}
                    st.session_state.chat_messages.append(new_message)
                    scene_response = await send_chat_message(session, new_message)
                    if scene_response:
                        system_response = scene_response["state"]
                        if type(system_response) == dict:
                            system_response = [system_response]

                        message = []
                        st_logger.info(system_response)
                        for response in system_response:
                            message.append((response["agent"], response["feedback"]))

                        message = ", ".join([f"{m[0]}: {m[1]}" for m in message])

                        st.session_state.chat_messages.append(
                            {"role": "user", "content": message}
                        )

            with c2:
                tab1, tab2 = st.tabs(["Chat View", "Map View"])

                with tab2:
                    scene_plh = st.empty()

                    if game_mode == "Manual":
                        _, c20, c21, c22, _ = st.columns(
                            [0.07, 2, 2, 2, 0.07], gap="large"
                        )
                        with c20:
                            turn_left = st.button("TURN LEFT", key="L")
                        with c21:
                            turn_right = st.button("TURN RIGHT", key="R")
                        with c22:
                            move_forward = st.button("MOVE FOWARD", key="F")

                        if move_forward:
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": "move_ahead"}
                            )
                            scene_response = await action_callback(
                                session, "move_ahead"
                            )

                        if turn_left:
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": "turn_left"}
                            )
                            scene_response = await action_callback(session, "turn_left")

                        if turn_right:
                            st.session_state.chat_messages.append(
                                {"role": "assistant", "content": "turn_right"}
                            )
                            scene_response = await action_callback(
                                session, "turn_right"
                            )

                # submit_btn = st.button('submit', on_click=submit_callback, use_container_width=True)
                # submit_button = st.empty()

                # self.plot_figure(plot_placeholder)

                # check_if_server_started()

                with scene_plh:
                    if not scene_response:
                        scene_response = await fetch_scene(session)
                    # st_logger.info(scene_response)
                    # if scene_response and (st.session_state.scene['timestamp'] != scene_response.get("timestamp")):

                    #     render_scene(scene_response.get("data"))

                    #     st.session_state.scene['timestamp'] = scene_response.get("timestamp")

                    if scene_response:
                        if "feedback" in scene_response:
                            st_logger.info(scene_response["feedback"])
                            st.session_state.chat_messages.append(
                                {
                                    "role": "user",
                                    "content": scene_response["feedback"]["state"][
                                        "feedback"
                                    ],
                                }
                            )

                        if "scene" in scene_response["data"] and (
                            st.session_state.scene["timestamp"]
                            != scene_response.get("timestamp")
                        ):
                            render_scene(scene_response.get("data").get("scene"))

                            st.session_state.scene["timestamp"] = scene_response.get(
                                "timestamp"
                            )

                with tab1:
                    for message in st.session_state.chat_messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

        # if not submit_button:
        #     st.stop()


if __name__ == "__main__":
    app = StreamlitApp()

    asyncio.new_event_loop().run_until_complete(app.start())
