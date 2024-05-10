import enum
import logging
import os
import socket
import time

import requests

from determined.common import api
from determined.common.api import authentication, certs


class IdleType(enum.Enum):
    KERNELS_OR_TERMINALS = 1
    KERNEL_CONNECTIONS = 2
    ACTIVITY = 3


REPORT_IDLE_INTERVAL = 30

last_activity = None


def wait_for_jupyter(addr):
    """
    Avoid logging enormous stacktraces when the requests library attempts to connect to a server
    that isn't accepting connections yet.  This is expected as jupyter startup time might take
    longer than a second, and we don't want to generate scary logs for expected behavior.
    """
    i = 0
    while True:
        with socket.socket() as s:
            try:
                s.connect(addr)
                # Connection worked, we're done here.
                return
            except ConnectionError as e:
                if (i + 1) % 10 == 0:
                    # Every 10 seconds without reaching jupyter, start telling the user.
                    # This is beyond the range of expected startup times.
                    logging.warning(f"jupyter is still not reachable at {addr}")
            time.sleep(1)
            i += 1


def is_idle(request_address: str, token: str, mode: IdleType):
    auth_header = {"Authorization": f"token {token}"}
    try:
        kernels = requests.get(
            f"{request_address}/api/kernels", headers=auth_header, verify=False
        ).json()
        terminals = requests.get(
            f"{request_address}/api/terminals", headers=auth_header, verify=False
        ).json()
        sessions = requests.get(
            f"{request_address}/api/sessions", headers=auth_header, verify=False
        ).json()
    except Exception:
        logging.warning("Cannot get notebook kernel status", exc_info=True)
        return False

    if mode == IdleType.KERNELS_OR_TERMINALS:
        return len(kernels) == 0 and len(terminals) == 0 and len(sessions) == 0
    elif mode == IdleType.KERNEL_CONNECTIONS:
        # Unfortunately, the terminals API doesn't return a connection count.
        return all(k["connections"] == 0 for k in kernels)
    elif mode == IdleType.ACTIVITY:
        global last_activity

        old_last_activity = last_activity
        if kernels or terminals:
            last_activity = max(x["last_activity"] for x in kernels + terminals)
        no_busy_kernels = all(k["execution_state"] != "busy" for k in kernels)

        return no_busy_kernels and (last_activity == old_last_activity)


def main():
    requests.packages.urllib3.disable_warnings()
    port = os.environ["NOTEBOOK_PORT"]
    notebook_id = os.environ["DET_TASK_ID"]
    token = os.environ["DET_USER_TOKEN"]
    notebook_server = f"https://127.0.0.1:{port}/proxy/{notebook_id}"
    master_url = api.canonicalize_master_url(os.environ["DET_MASTER"])
    cert = certs.default_load(master_url)
    sess = authentication.login_with_cache(master_url, cert=cert)
    try:
        idle_type = IdleType[os.environ["NOTEBOOK_IDLE_TYPE"].upper()]
    except KeyError:
        logging.warning(
            "unknown idle type '%s', using default value",
            os.environ["NOTEBOOK_IDLE_TYPE"],
        )
        idle_type = IdleType.KERNELS_OR_TERMINALS

    wait_for_jupyter(("127.0.0.1", int(port)))

    while True:
        try:
            idle = is_idle(notebook_server, token, idle_type)
            sess.put(
                f"/api/v1/notebooks/{notebook_id}/report_idle",
                params={"notebook_id": notebook_id, "idle": idle},
            )
        except Exception:
            logging.warning("ignoring error communicating with master", exc_info=True)
        time.sleep(REPORT_IDLE_INTERVAL)


if __name__ == "__main__":
    main()
