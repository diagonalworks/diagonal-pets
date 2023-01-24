from datetime import datetime, timedelta
import threading

class Progress:

    INTERVAL = timedelta(seconds = 30)

    def __init__(self, name, items):
        self.name = name
        self.items = items
        self.last = datetime.now()
        self.seen = 0

    def next(self):
        now = datetime.now()
        if self.seen == 0:
            self.start = now
        self.seen += 1
        if now - self.last > self.INTERVAL:
            self._report(now)
            self.last = now

    def finish(self):
        now = datetime.now()
        print("%s: done in %s" % (self.name, self._format_timedelta(now - self.start)))

    def _report(self, now):
        msg = "%s: %d/%d %d%%" % (self.name, self.seen, self.items, int(self.seen * 100.0 / self.items))
        if self.seen > 1:
            remaining = ((now - self.start) / (self.seen - 1)) * (self.items - self.seen)
            msg += " %s remaining" % self._format_timedelta(remaining)
        print(msg)

    def _format_timedelta(self, d):
        msg = ""
        if d.days > 0:
            msg += "%dd" % d.days
        if d.seconds >= 60*60:
            msg += "%dh" % (d.seconds / (60*60))
        if (d.seconds % (60*60)) >= 60:
            msg += "%dm" % ((d.seconds % (60*60)) / 60)
        else:
            msg += "%ds" % (d.seconds % (60*60))
        return msg

def track(name, items):
    return Progress(name, items)

class NullProgress:

    def next(self):
        pass

    def finish(self):
        pass

def dont_track(name, items):
    return NullProgress()