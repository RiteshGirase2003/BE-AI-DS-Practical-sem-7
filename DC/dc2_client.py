import Pyro4

uri = input("Enter the server URI: ").strip()  # <<< use .strip() to clean input
string_concatenator = Pyro4.Proxy(uri)

s1 = input("Enter first string: ")
s2 = input("Enter second string: ")

result = string_concatenator.concatenate(s1, s2)
print("Concatenated string:", result)