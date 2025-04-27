import time
import random

# Servers represented as dictionary {server_id: request_count}
servers = {
    "Server1": 0,
    "Server2": 0,
    "Server3": 0
}

server_list = list(servers.keys())
round_robin_index = 0

def handle_request(server):
    servers[server] += 1
    print(f"{server} handling request. Total requests: {servers[server]}")

def round_robin():
    global round_robin_index
    server = server_list[round_robin_index]
    round_robin_index = (round_robin_index + 1) % len(server_list)
    return server

def least_connections():
    return min(servers, key=lambda s: servers[s])

def distribute_request(method):
    if method == "round_robin":
        server = round_robin()
    elif method == "least_connections":
        server = least_connections()
    else:
        raise ValueError("Unknown method")
    handle_request(server)

# Simulate client requests
methods = ["round_robin", "least_connections"]

for i in range(10):
    method = random.choice(methods)
    print(f"\nRequest {i+1} using {method} method:")
    distribute_request(method)
    time.sleep(1)
