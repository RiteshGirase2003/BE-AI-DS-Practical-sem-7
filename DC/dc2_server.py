# server1.py
import Pyro4

@Pyro4.expose
class StringConcat:
    def concatenate(self, s1, s2):
        return s1 + s2

daemon = Pyro4.Daemon()
uri = daemon.register(StringConcat)
print("Ready. Object URI =", uri)
daemon.requestLoop()