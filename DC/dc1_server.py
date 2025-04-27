from xmlrpc.server import SimpleXMLRPCServer

# Define factorial function
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Create server
server = SimpleXMLRPCServer(("localhost", 8000))

print("Server is listening on port 8000...")

# Register the function
server.register_function(factorial, "factorial")

# Run the server
server.serve_forever()