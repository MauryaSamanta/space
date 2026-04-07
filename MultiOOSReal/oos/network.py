class Network:
    def __init__(self, delay=200):
        self.delay = delay
        self.queue = []

    def broadcast(self, msg, current_time):
        deliver_time = current_time + self.delay

        self.queue.append({
            "msg": msg,
            "deliver_time": deliver_time
        })

    def deliver(self, current_time, fleet):

        delivered = []

        for item in self.queue:
            if item["deliver_time"] <= current_time:

                for oos in fleet:
                    oos.inbox.append(item["msg"])

                delivered.append(item)

        for item in delivered:
            self.queue.remove(item)