import xmlrpc.client

# Connect to the server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Input number
num = int(input("Enter an integer to calculate factorial: "))

# Call the remote factorial function
result = proxy.factorial(num)

print(f"Factorial of {num} is: {result}")